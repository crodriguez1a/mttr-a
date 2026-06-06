"""
ProductionRunner — orchestrates N LangGraph episodes and returns SystemMetrics.

Replaces the simulation's Orchestrator. Key differences from the mock:
  - Episodes are built from real LangGraph state (real timestamps, real confidence)
  - Stable intervals are still sampled (no continuous wall-clock monitoring needed
    for batch benchmarks; swap to event-driven collection for live systems)
  - Telemetry is emitted per-episode via the injected TelemetrySink
"""

from __future__ import annotations

import random
import time

from mttr_a_simulation import QUERY_POOL, MetricsComputer, SystemMetrics

from .config import BenchmarkConfig
from .graph import AgentState, build_graph, state_to_episode, state_to_extended
from .providers import BaseLLMProvider
from .sinks import TelemetrySink


class ProductionRunner:
    """
    Drives the LangGraph workflow for N runs, collects episodes, and returns
    a fully computed SystemMetrics.

    Dependency injection for provider and sink means tests can supply a
    MockProvider and an in-memory sink without touching the runner logic.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        provider: BaseLLMProvider,
        sink: TelemetrySink,
    ) -> None:
        self._config = config
        self._provider = provider
        self._sink = sink

    def run(self) -> SystemMetrics:
        cfg = self._config
        rng = random.Random(cfg.seed)
        graph = build_graph(self._provider, cfg, seed=cfg.seed)

        episodes = []
        stable_intervals: list[float] = []
        wall_clock = 0.0

        if cfg.verbose:
            print(f"\n{'─'*64}")
            print("  Run    Query excerpt                 Conf   Drift  Reflex")
            print(f"{'─'*64}")

        for run_id in range(cfg.n_runs):
            query = rng.choice(QUERY_POOL)

            initial_state: AgentState = {
                "run_id": run_id,
                "query": query,
                "response": "",
                "confidence": 0.0,
                "is_drift": False,
                "reflex_mode": None,
                "t_reason_start": 0.0,
                "t_reason_end": 0.0,
                "t_drift_check": 0.0,
                "t_recovery_start": 0.0,
                "t_recovery_end": 0.0,
                "T_detect": 0.0,
                "T_decide": 0.0,
                "T_execute": 0.0,
                "t_queued": time.perf_counter(),  # Gap 3: queue entry timestamp
                "t_first_token": 0.0,             # Gap 2: set by reasoning node
                "tool_latency_s": 0.0,            # Gap 4: set by reasoning node
                "n_tool_calls": 0,                # Gap 4: set by reasoning node
            }

            final_state: AgentState = graph.invoke(initial_state)
            episode = state_to_episode(final_state, wall_clock)
            extended = state_to_extended(final_state)

            stable = rng.expovariate(1.0 / cfg.mtbf_mean)
            wall_clock = episode.t_recovered + stable

            episodes.append(episode)
            stable_intervals.append(stable)
            self._sink.emit(episode, extended)

            if cfg.verbose and (run_id < 20 or run_id % 50 == 0):
                drift_marker = "YES" if episode.drift_detected else "  -"
                reflex = episode.reflex_mode or "—"
                print(
                    f"  {run_id:>4}   {episode.query[:32]:<32} "
                    f"{episode.confidence:.3f}  {drift_marker}    {reflex}"
                )

        if cfg.verbose:
            print(f"{'─'*64}")
            print(f"  ... ({cfg.n_runs} total runs completed)")

        self._sink.flush()

        return MetricsComputer(alpha=cfg.alpha).compute(
            episodes, stable_intervals, n_runs=cfg.n_runs
        )
