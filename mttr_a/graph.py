"""
LangGraph StateGraph for the MTTR-A multi-step reasoning pipeline.

Each episode runs a configurable number of reasoning steps. Every step injects
the retrieved grounding document as context and measures groundedness as
cos(step_output_emb, grounding_doc_emb) — the paper's signal applied to outputs.

Drift is detected when any step's groundedness drops below τ_drift, triggering
the recovery reflex mid-chain. Episode confidence = min(step_confidences).

For MockProvider: groundedness is simulated per-step from N(μ, σ) matching the
distribution of real retrieval-confidence scores. No embedding calls are made.
"""

from __future__ import annotations

import random
import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from mttr_a_simulation import REFLEX_PARAMS, Episode

from .config import BenchmarkConfig
from .providers import BaseLLMProvider, MockProvider, _load_corpus, get_top_doc, step_groundedness

# ── Graph state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    run_id: int
    query: str
    grounding_doc: str
    grounding_doc_emb: Any          # L2-normalized np.ndarray; None for mock
    step_index: int
    max_steps: int
    step_outputs: list              # list[str]
    step_confidences: list          # list[float]
    response: str
    confidence: float               # min(step_confidences)
    is_drift: bool
    drift_step: int                 # step index where drift first occurred (-1 = none)
    reflex_mode: str | None
    t_reason_start: float
    t_reason_end: float
    t_drift_check: float
    t_recovery_start: float
    t_recovery_end: float
    T_detect: float
    T_decide: float
    T_execute: float
    t_queued: float
    t_first_token: float
    tool_latency_s: float
    n_tool_calls: int


# ── Node factories ────────────────────────────────────────────────────────────

def _make_retrieve_node(is_mock: bool, rng: random.Random, corpus_embs=None, corpus_docs=None):
    """Finds the grounding document for the query and stores it in state."""
    def retrieve_node(state: AgentState) -> AgentState:
        state["t_reason_start"] = time.perf_counter()
        if is_mock:
            docs = corpus_docs or _load_corpus()
            state["grounding_doc"] = rng.choice(docs)
            state["grounding_doc_emb"] = None
        else:
            doc_text, doc_emb, _ = get_top_doc(
                state["query"], corpus_embs=corpus_embs, corpus_docs=corpus_docs
            )
            state["grounding_doc"] = doc_text
            state["grounding_doc_emb"] = doc_emb
        return state
    return retrieve_node


def _make_reasoning_step_node(provider: BaseLLMProvider, is_mock: bool, mock_rng: random.Random):
    """
    Runs one reasoning step.

    Real providers: injects the grounding doc as context, then measures
    groundedness = cos(step_output_emb, grounding_doc_emb).

    Mock: samples groundedness from the calibrated Gaussian directly —
    no embedding call needed.
    """
    def reasoning_step_node(state: AgentState) -> AgentState:
        step = state["step_index"]

        if is_mock:
            result = provider.invoke(state["query"])
            confidence = result.confidence  # Gaussian-sampled groundedness
        else:
            history = "\n".join(
                f"Step {i+1}: {s}" for i, s in enumerate(state["step_outputs"])
            )
            parts = [
                f"Context document:\n{state['grounding_doc']}",
                f"Task: {state['query']}",
            ]
            if history:
                parts.append(f"Reasoning so far:\n{history}")
            parts.append(
                f"Step {step + 1} — reason through this step. "
                "Stay grounded in the context document above:"
            )
            result = provider.invoke("\n\n".join(parts))
            confidence = step_groundedness(result.content, state["grounding_doc_emb"])

        new_outputs = list(state["step_outputs"]) + [result.content]
        new_confs = list(state["step_confidences"]) + [confidence]

        state["step_outputs"] = new_outputs
        state["step_confidences"] = new_confs
        state["response"] = result.content
        state["confidence"] = min(new_confs)
        state["step_index"] = step + 1
        state["t_first_token"] = result.t_first_token or 0.0
        state["tool_latency_s"] = (
            state["tool_latency_s"] + sum(tc.latency_s for tc in result.tool_calls)
        )
        state["n_tool_calls"] = state["n_tool_calls"] + len(result.tool_calls)
        state["t_reason_end"] = time.perf_counter()
        return state

    return reasoning_step_node


def _make_check_groundedness_node(config: BenchmarkConfig, rng: random.Random):
    """Flags drift if the latest step's groundedness is below τ_drift."""
    def check_groundedness_node(state: AgentState) -> AgentState:
        state["t_drift_check"] = time.perf_counter()
        latest = state["step_confidences"][-1] if state["step_confidences"] else 0.0
        drifted = (
            latest < config.drift_threshold
            or rng.random() < config.stochastic_fault_rate
        )
        state["is_drift"] = drifted
        if drifted and state["drift_step"] == -1:
            state["drift_step"] = state["step_index"] - 1
        return state
    return check_groundedness_node


def _make_recovery_node(provider: BaseLLMProvider, rng: random.Random):
    """Selects and executes a recovery reflex when drift is detected."""
    def recovery_node(state: AgentState) -> AgentState:
        state["t_recovery_start"] = time.perf_counter()

        if not state["is_drift"]:
            state.update({
                "reflex_mode": None,
                "T_detect": 0.0,
                "T_decide": 0.0,
                "T_execute": 0.0,
                "t_recovery_end": state["t_recovery_start"],
            })
            return state

        T_detect = state["t_drift_check"] - state["t_reason_end"]

        t_decide_start = time.perf_counter()
        modes = list(REFLEX_PARAMS.keys())
        weights = [REFLEX_PARAMS[m].weight for m in modes]
        (mode,) = rng.choices(modes, weights=weights, k=1)
        T_decide = time.perf_counter() - t_decide_start

        t_exec_start = time.perf_counter()
        if mode == "auto-replan":
            provider.invoke(
                f"Replan and improve your answer to: {state['query']}",
                context="recovery",
            )
        elif mode == "tool-retry":
            provider.invoke(state["query"], context="retry")
        elif mode == "rollback":
            provider.invoke(
                f"Answer simply and concisely: {state['query']}",
                context="rollback",
            )
        elif mode == "human-approve":
            pass
        T_execute = time.perf_counter() - t_exec_start

        state.update({
            "reflex_mode": mode,
            "T_detect": T_detect,
            "T_decide": T_decide,
            "T_execute": T_execute,
            "t_recovery_end": time.perf_counter(),
        })
        return state

    return recovery_node


# ── Routing ───────────────────────────────────────────────────────────────────

def _route_after_groundedness_check(state: AgentState) -> str:
    if state["is_drift"]:
        return "recovery"
    if state["step_index"] >= state["max_steps"]:
        return END
    return "reasoning_step"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(
    provider: BaseLLMProvider,
    config: BenchmarkConfig,
    seed: int,
    corpus_embs=None,
    corpus_docs: list[str] | None = None,
):
    """
    Compile the MTTR-A multi-step reasoning graph.

    Each episode: retrieve → reasoning_step (×N) → check_groundedness
                         → [drift → recovery | done → END]
    """
    is_mock = isinstance(provider, MockProvider)
    rng = random.Random(seed)

    g: StateGraph = StateGraph(AgentState)

    g.add_node("retrieve",           _make_retrieve_node(is_mock, rng, corpus_embs, corpus_docs))
    g.add_node("reasoning_step",     _make_reasoning_step_node(provider, is_mock, rng))
    g.add_node("check_groundedness", _make_check_groundedness_node(config, rng))
    g.add_node("recovery",           _make_recovery_node(provider, rng))

    g.add_edge(START,                "retrieve")
    g.add_edge("retrieve",           "reasoning_step")
    g.add_edge("reasoning_step",     "check_groundedness")
    g.add_conditional_edges(
        "check_groundedness",
        _route_after_groundedness_check,
        {"recovery": "recovery", "reasoning_step": "reasoning_step", END: END},
    )
    g.add_edge("recovery",           END)

    return g.compile()


# ── State → Episode conversion ────────────────────────────────────────────────

def state_to_episode(state: AgentState, wall_clock: float) -> Episode:
    delta_t = state["T_detect"] + state["T_decide"] + state["T_execute"]
    return Episode(
        run_id=state["run_id"],
        query=state["query"],
        confidence=round(state["confidence"], 4),
        drift_detected=state["is_drift"],
        reflex_mode=state["reflex_mode"],
        t_detect=round(state["T_detect"], 4),
        t_decide=round(state["T_decide"], 4),
        t_execute=round(state["T_execute"], 4),
        delta_t=round(delta_t, 4),
        t_fault=round(wall_clock, 4),
        t_recovered=round(wall_clock + delta_t, 4),
        step_confidences=tuple(round(c, 4) for c in state["step_confidences"]),
    )


def state_to_extended(state: AgentState) -> dict:
    queue_s = (
        max(0.0, state["t_reason_start"] - state["t_queued"]) if state["t_queued"] > 0.0 else 0.0
    )
    ttft_s = (
        round(state["t_first_token"] - state["t_reason_start"], 6)
        if state["t_first_token"] > 0.0 and state["t_reason_start"] > 0.0
        else None
    )
    return {
        "queue_latency_s": round(queue_s, 6),
        "ttft_s": ttft_s,
        "tool_latency_s": round(state["tool_latency_s"], 6),
        "n_tool_calls": state["n_tool_calls"],
        "step_confidences": [round(c, 4) for c in state["step_confidences"]],
        "drift_step": state["drift_step"],
    }
