from __future__ import annotations

import time
from dataclasses import dataclass, field

from mttr_a.providers import step_groundedness
from mttr_a_simulation import Episode

TAU_DRIFT = 0.6


@dataclass
class StepRecord:
    turn: int
    reasoning_excerpt: str
    confidence: float | None  # None until the agent has retrieved evidence
    latency_s: float
    tool_called: str | None
    drift: bool
    recovered: bool
    t_wall: float = field(default_factory=time.perf_counter)


@dataclass
class ToolRecord:
    turn: int
    tool_name: str
    latency_s: float
    t_wall: float = field(default_factory=time.perf_counter)


class MTTRSession:
    def __init__(self, task: str, run_id: int = 0, drift_threshold: float = TAU_DRIFT):
        self._task = task
        self._run_id = run_id
        self._tau = drift_threshold
        self._steps: list[StepRecord] = []
        self._tool_calls: list[ToolRecord] = []
        self._evidence_emb = None  # set when the agent retrieves its first document
        self._drift_detected = False
        self._recovered = False
        self._t_fault: float | None = None
        self._t_recovered: float | None = None
        self._t_start = time.perf_counter()

    def record_step(
        self,
        reasoning_text: str,
        latency_s: float = 0.0,
        tool_called: str | None = None,
    ) -> StepRecord:
        # Groundedness can only be measured against evidence the agent has retrieved.
        # Before any retrieval tool fires, there is nothing to score against.
        if self._evidence_emb is not None:
            confidence = step_groundedness(reasoning_text, self._evidence_emb)
            drift = confidence < self._tau

            if drift and not self._drift_detected:
                self._drift_detected = True
                self._t_fault = time.perf_counter()

            recovered_this_step = False
            if not drift and self._drift_detected and not self._recovered:
                self._recovered = True
                self._t_recovered = time.perf_counter()
                recovered_this_step = True
        else:
            confidence = None
            drift = False
            recovered_this_step = False

        record = StepRecord(
            turn=len(self._steps),
            reasoning_excerpt=reasoning_text[:200],
            confidence=confidence,
            latency_s=latency_s,
            tool_called=tool_called,
            drift=drift,
            recovered=recovered_this_step,
        )
        self._steps.append(record)
        return record

    def record_tool_call(self, turn: int, tool_name: str, latency_s: float) -> ToolRecord:
        record = ToolRecord(turn=turn, tool_name=tool_name, latency_s=latency_s)
        self._tool_calls.append(record)
        return record

    def set_evidence(self, doc_text: str) -> None:
        """Embed the document the agent just retrieved.
        Subsequent steps are scored against this embedding until the next retrieval."""
        import numpy as np

        from mttr_a.providers import _get_embed_model
        model = _get_embed_model()
        emb = model.encode([doc_text], convert_to_numpy=True)
        norm = np.linalg.norm(emb)
        self._evidence_emb = emb[0] / (norm + 1e-9)

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    @property
    def recovered(self) -> bool:
        return self._recovered

    @property
    def latest_confidence(self) -> float | None:
        for s in reversed(self._steps):
            if s.confidence is not None:
                return s.confidence
        return None

    @property
    def step_confidences(self) -> list[float | None]:
        return [s.confidence for s in self._steps]

    @property
    def steps(self) -> list[StepRecord]:
        return list(self._steps)

    @property
    def tool_calls(self) -> list[ToolRecord]:
        return list(self._tool_calls)

    def episode(self) -> Episode:
        scored = [c for c in self.step_confidences if c is not None]
        episode_confidence = min(scored) if scored else 0.0

        t_execute = 0.0
        delta_t = 0.0
        t_fault = self._t_fault or self._t_start
        t_recovered = self._t_recovered or t_fault

        if self._drift_detected and self._t_fault is not None and self._t_recovered is not None:
            # T_execute: wall-clock time spanning the inference calls made while drifting.
            # No proxy — this is measured from real API latency.
            t_execute = max(0.0, self._t_recovered - self._t_fault)
            delta_t = t_execute

        return Episode(
            run_id=self._run_id,
            query=self._task[:80],
            confidence=round(episode_confidence, 4),
            drift_detected=self._drift_detected,
            reflex_mode="natural" if self._recovered else None,
            t_detect=0.0,
            t_decide=0.0,
            t_execute=round(t_execute, 4),
            delta_t=round(delta_t, 4),
            t_fault=round(t_fault, 4),
            t_recovered=round(t_recovered, 4),
            step_confidences=tuple(round(c, 4) for c in scored),
        )
