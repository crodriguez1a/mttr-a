"""
LangGraph StateGraph for the MTTR-A three-node pipeline.

Replaces the simulation's mocked Pipeline with a real LangGraph workflow
backed by whichever provider is configured. All timing uses time.perf_counter()
against real call durations — no artificial sleep.

Node responsibilities
---------------------
  reasoning_node    call the LLM, record confidence and timing
  check_drift_node  compare confidence to τ_drift, set is_drift flag
  recovery_node     select reflex, re-invoke LLM, record latency components

The Episode returned maps 1-to-1 with the simulation's Episode dataclass so
MetricsComputer works without modification.
"""

from __future__ import annotations

import random
import time
from typing import Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from mttr_a_simulation import QUERY_POOL, REFLEX_PARAMS, Episode

from .config import BenchmarkConfig
from .providers import BaseLLMProvider


# ── Graph state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    run_id: int
    query: str
    response: str
    confidence: float
    is_drift: bool
    reflex_mode: Optional[str]
    t_reason_start: float
    t_reason_end: float
    t_drift_check: float
    t_recovery_start: float
    t_recovery_end: float
    T_detect: float
    T_decide: float
    T_execute: float
    # Extended instrumentation (not used by MetricsComputer; written to telemetry only)
    t_queued: float        # Gap 3: perf_counter when runner queued this episode
    t_first_token: float   # Gap 2: perf_counter of first streaming token (0.0 = not streaming)
    tool_latency_s: float  # Gap 4: total time spent in tool calls during reasoning
    n_tool_calls: int      # Gap 4: number of tool invocations


# ── Node factories ────────────────────────────────────────────────────────────

def _make_reasoning_node(provider: BaseLLMProvider):
    """Returns a node function that calls the LLM and records confidence + timing."""

    def reasoning_node(state: AgentState) -> AgentState:
        state["t_reason_start"] = time.perf_counter()
        result = provider.invoke(state["query"])
        state["response"] = result.content
        state["confidence"] = result.confidence
        state["t_first_token"] = result.t_first_token or 0.0
        state["tool_latency_s"] = sum(tc.latency_s for tc in result.tool_calls)
        state["n_tool_calls"] = len(result.tool_calls)
        state["t_reason_end"] = time.perf_counter()
        return state

    return reasoning_node


def _make_check_drift_node(config: BenchmarkConfig, rng: random.Random):
    """Returns a node function that flags drift based on confidence threshold."""

    def check_drift_node(state: AgentState) -> AgentState:
        state["t_drift_check"] = time.perf_counter()
        state["is_drift"] = (
            state["confidence"] < config.drift_threshold
            or rng.random() < config.stochastic_fault_rate
        )
        return state

    return check_drift_node


def _make_recovery_node(provider: BaseLLMProvider, rng: random.Random):
    """
    Returns a node function that selects and executes a recovery reflex.

    Reflex actions:
      auto-replan   re-invoke with an orchestration-focused prompt
      tool-retry    re-invoke the original query (simulates retrying a tool)
      rollback      invoke a simplified fallback query
      human-approve emit an escalation record; no LLM call (requires human gate)
    """

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

        # Policy selection
        t_decide_start = time.perf_counter()
        modes = list(REFLEX_PARAMS.keys())
        weights = [REFLEX_PARAMS[m].weight for m in modes]
        (mode,) = rng.choices(modes, weights=weights, k=1)
        T_decide = time.perf_counter() - t_decide_start

        # Reflex execution
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
            # In production: emit to escalation queue (PagerDuty, ServiceNow, etc.)
            # Measuring elapsed time here captures the gate latency when integrated
            # with a real approval workflow.
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


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(
    provider: BaseLLMProvider,
    config: BenchmarkConfig,
    seed: int,
):
    """
    Compile the LangGraph StateGraph for one benchmark configuration.
    Returns a compiled graph that accepts AgentState dicts via .invoke().
    """
    rng = random.Random(seed)

    g: StateGraph = StateGraph(AgentState)
    g.add_node("reasoning",   _make_reasoning_node(provider))
    g.add_node("check_drift", _make_check_drift_node(config, rng))
    g.add_node("recovery",    _make_recovery_node(provider, rng))

    g.add_edge(START,         "reasoning")
    g.add_edge("reasoning",   "check_drift")
    g.add_edge("check_drift", "recovery")
    g.add_edge("recovery",    END)

    return g.compile()


# ── State → Episode conversion ────────────────────────────────────────────────

def state_to_episode(state: AgentState, wall_clock: float) -> Episode:
    """
    Convert a completed AgentState into the Episode dataclass that
    MetricsComputer and TelemetryLogger expect.
    """
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
    )


def state_to_extended(state: AgentState) -> dict:
    """
    Extract the instrumentation fields that live outside the paper's Episode model.

    Returns a flat dict suitable for merging into the telemetry JSONL record.
    These fields address the four latency gaps not covered by the core metric:

      queue_latency_s  — Gap 3: pre-LLM routing / queue time
      t_first_token_s  — Gap 2: time-to-first-token (None when not streaming)
      tool_latency_s   — Gap 4: total time spent in tool calls
      n_tool_calls     — Gap 4: number of tool invocations
    """
    queue_s = max(0.0, state["t_reason_start"] - state["t_queued"]) if state["t_queued"] > 0.0 else 0.0
    ttft_s = (
        round(state["t_first_token"] - state["t_reason_start"], 6)
        if state["t_first_token"] > 0.0 and state["t_reason_start"] > 0.0
        else None
    )
    return {
        "queue_latency_s": round(queue_s, 6),
        "ttft_s": ttft_s,               # time from reasoning start to first token
        "tool_latency_s": round(state["tool_latency_s"], 6),
        "n_tool_calls": state["n_tool_calls"],
    }
