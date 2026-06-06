"""
Comprehensive test suite for mttr_a_simulation.py.

Coverage targets: unit (pure functions), functional (classes), integration (end-to-end).
All tests are hermetic — no shared mutable state between cases.
"""

from __future__ import annotations

import io
import json
import math
import random
import runpy
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from mttr_a_simulation import (
    DEFAULT_ALPHA,
    DEFAULT_MTBF_MEAN,
    QUERY_POOL,
    REFLEX_PARAMS,
    ROLLING_WINDOW,
    TAU_DRIFT,
    _PAPER_METRICS,
    _load_query_pool,
    _load_reflex_params,
    _load_mock_config,
    _load_paper_benchmarks,
    Episode,
    LatencyDecomposition,
    MetricsComputer,
    ModeStats,
    Orchestrator,
    Pipeline,
    RecoveryResult,
    ReflexConfig,
    Reporter,
    ResultsSaver,
    SystemMetrics,
    TelemetryLogger,
    check_drift_node,
    compute_k_alpha,
    compute_latency_decomposition,
    compute_med_ttr_a,
    compute_mtbf,
    compute_mttr_a,
    compute_nrr,
    compute_nrr_alpha,
    compute_per_mode_stats,
    compute_percentile,
    compute_pi_up,
    compute_rolling_median,
    compute_std,
    main,
    reasoning_node,
    recovery_node,
)

# ── Shared fixtures ───────────────────────────────────────────────────────────


def _episode(
    run_id: int = 0,
    drift_detected: bool = True,
    reflex_mode: str | None = "auto-replan",
    t_detect: float = 0.5,
    t_decide: float = 0.15,
    t_execute: float = 5.29,
    delta_t: float = 5.94,
    t_fault: float = 0.0,
    t_recovered: float = 5.94,
) -> Episode:
    """Build an Episode with sensible defaults for testing."""
    return Episode(
        run_id=run_id,
        query="test query",
        confidence=0.55,
        drift_detected=drift_detected,
        reflex_mode=reflex_mode,
        t_detect=t_detect,
        t_decide=t_decide,
        t_execute=t_execute,
        delta_t=delta_t,
        t_fault=t_fault,
        t_recovered=t_recovered,
    )


def _metrics(**overrides: Any) -> SystemMetrics:
    """Build a SystemMetrics with all valid defaults, optionally overriding fields."""
    defaults: dict[str, Any] = dict(
        n_runs=200,
        drift_events=200,
        drift_rate=1.0,
        mttr_a_sys=6.5,
        med_ttr_a_sys=6.08,
        std_sys=2.18,
        p90_sys=11.47,
        mtbf_sys=6.45,
        nrr_sys=0.058,
        pi_up_sys=0.515,
        nrr_alpha=-0.055,
        alpha=DEFAULT_ALPHA,
        k_alpha=0.3333,
        latency_decomposition=LatencyDecomposition(
            t_detect_mean=0.496,
            t_decide_mean=0.172,
            t_execute_mean=5.864,
        ),
        per_mode={
            "auto-replan": ModeStats(count=99, median=5.94, std=0.74, p90=6.91),
            "tool-retry":  ModeStats(count=40, median=4.57, std=0.51, p90=5.29),
        },
        rolling_med_ttr_a=[5.8, 6.0, 6.2, 6.1, 5.9],
    )
    defaults.update(overrides)
    return SystemMetrics(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — reasoning_node
# ═══════════════════════════════════════════════════════════════════════════════


class TestReasoningNode:

    def test_output_in_unit_interval(self) -> None:
        rng = random.Random(0)
        for _ in range(100):
            result = reasoning_node("any query", rng)
            assert 0.0 <= result <= 1.0

    def test_deterministic_with_same_seed(self) -> None:
        result_a = reasoning_node("q", random.Random(7))
        result_b = reasoning_node("q", random.Random(7))
        assert result_a == result_b

    def test_clamps_to_zero_on_extreme_low_gauss(self) -> None:
        rng = MagicMock()
        rng.gauss.side_effect = [-100.0, 0.0]
        assert reasoning_node("q", rng) == 0.0

    def test_clamps_to_one_on_extreme_high_gauss(self) -> None:
        rng = MagicMock()
        rng.gauss.side_effect = [100.0, 0.0]
        assert reasoning_node("q", rng) == 1.0

    def test_query_argument_is_accepted(self) -> None:
        rng = random.Random(1)
        for q in QUERY_POOL:
            result = reasoning_node(q, rng)
            assert isinstance(result, float)


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — check_drift_node
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckDriftNode:

    def test_below_threshold_always_true(self) -> None:
        rng = MagicMock()
        rng.random.return_value = 0.99  # no stochastic trigger
        assert check_drift_node(TAU_DRIFT - 0.01, rng) is True

    def test_above_threshold_no_stochastic_returns_false(self) -> None:
        rng = MagicMock()
        rng.random.return_value = 0.99  # well above 0.05 stochastic threshold
        assert check_drift_node(TAU_DRIFT + 0.1, rng) is False

    def test_stochastic_trigger_above_threshold(self) -> None:
        rng = MagicMock()
        rng.random.return_value = 0.04  # < 0.05 → stochastic fault
        assert check_drift_node(TAU_DRIFT + 0.3, rng) is True

    def test_force_true_always_returns_true(self) -> None:
        rng = MagicMock()
        rng.random.return_value = 0.99
        assert check_drift_node(0.99, rng, force=True) is True

    def test_force_true_does_not_call_rng(self) -> None:
        rng = MagicMock()
        check_drift_node(0.99, rng, force=True)
        rng.random.assert_not_called()

    def test_exactly_at_threshold_is_not_drift(self) -> None:
        rng = MagicMock()
        rng.random.return_value = 0.99
        assert check_drift_node(TAU_DRIFT, rng) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — recovery_node
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecoveryNode:

    def test_mode_is_a_known_reflex(self) -> None:
        rng = random.Random(99)
        for _ in range(20):
            result = recovery_node(rng)
            assert result.mode in REFLEX_PARAMS

    def test_all_latencies_are_positive(self) -> None:
        rng = random.Random(5)
        for _ in range(30):
            result = recovery_node(rng)
            assert result.t_detect > 0
            assert result.t_decide > 0
            assert result.t_execute >= 0.1  # enforced minimum

    def test_t_execute_minimum_enforced(self) -> None:
        rng = MagicMock()
        rng.choices.return_value = ["auto-replan"]
        rng.expovariate.side_effect = [10.0, 10.0]  # t_detect+t_decide >> median
        rng.gauss.return_value = -999.0              # target_exec very negative
        result = recovery_node(rng)
        assert result.t_execute == 0.1

    def test_deterministic_with_same_seed(self) -> None:
        a = recovery_node(random.Random(42))
        b = recovery_node(random.Random(42))
        assert a == b

    def test_returns_recovery_result_namedtuple(self) -> None:
        result = recovery_node(random.Random(1))
        assert isinstance(result, RecoveryResult)
        assert hasattr(result, "mode")
        assert hasattr(result, "t_detect")
        assert hasattr(result, "t_decide")
        assert hasattr(result, "t_execute")

    def test_weighted_sampling_distribution(self) -> None:
        rng = random.Random(0)
        counts: dict[str, int] = {m: 0 for m in REFLEX_PARAMS}
        n = 2000
        for _ in range(n):
            counts[recovery_node(rng).mode] += 1
        total_weight = sum(c.weight for c in REFLEX_PARAMS.values())
        for mode, cfg in REFLEX_PARAMS.items():
            expected = cfg.weight / total_weight
            observed = counts[mode] / n
            assert abs(observed - expected) < 0.05, (
                f"{mode}: expected ≈{expected:.2%}, got {observed:.2%}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — compute_mttr_a
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeMttrA:

    def test_basic_mean(self) -> None:
        assert compute_mttr_a([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_single_value(self) -> None:
        assert compute_mttr_a([7.5]) == pytest.approx(7.5)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_mttr_a([])


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — compute_med_ttr_a
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeMedTtrA:

    def test_odd_count(self) -> None:
        assert compute_med_ttr_a([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_even_count(self) -> None:
        assert compute_med_ttr_a([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)

    def test_single_value(self) -> None:
        assert compute_med_ttr_a([9.0]) == pytest.approx(9.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_med_ttr_a([])


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — compute_std
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeStd:

    def test_basic_known_std(self) -> None:
        # [0, 1, 2] → sample std = 1.0 (ddof=1)
        assert compute_std([0.0, 1.0, 2.0]) == pytest.approx(1.0)

    def test_single_value_returns_zero(self) -> None:
        assert compute_std([5.0]) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert compute_std([]) == 0.0

    def test_identical_values_returns_zero(self) -> None:
        assert compute_std([3.0, 3.0, 3.0]) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — compute_percentile
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputePercentile:

    def test_p90_basic(self) -> None:
        # sorted[int(0.9 * 10)] = sorted[9] = 10 (implementation uses int-floor indexing)
        values = list(range(1, 11))  # 1..10
        result = compute_percentile(values, 0.9)
        assert result == 10

    def test_p50_is_median_for_odd_list(self) -> None:
        result = compute_percentile([1.0, 2.0, 3.0], 0.5)
        assert result == pytest.approx(2.0)

    def test_all_same_values(self) -> None:
        result = compute_percentile([5.0] * 10, 0.9)
        assert result == pytest.approx(5.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_percentile([], 0.9)


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — compute_mtbf
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeMtbf:

    def test_basic_mean_of_intervals(self) -> None:
        assert compute_mtbf([5.0, 7.0, 9.0]) == pytest.approx(7.0)

    def test_single_interval(self) -> None:
        assert compute_mtbf([6.73]) == pytest.approx(6.73)

    def test_empty_returns_zero(self) -> None:
        assert compute_mtbf([]) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — compute_nrr
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeNrr:

    def test_basic_formula(self) -> None:
        assert compute_nrr(6.0, 7.0) == pytest.approx(1.0 - 6.0 / 7.0)

    def test_zero_mtbf_returns_nan(self) -> None:
        assert math.isnan(compute_nrr(6.0, 0.0))

    def test_zero_mttr_perfect_reliability(self) -> None:
        assert compute_nrr(0.0, 7.0) == pytest.approx(1.0)

    def test_equal_mttr_and_mtbf(self) -> None:
        assert compute_nrr(5.0, 5.0) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — compute_k_alpha
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeKAlpha:

    def test_ninety_percent(self) -> None:
        expected = math.sqrt(0.1 / 0.9)
        assert compute_k_alpha(0.90) == pytest.approx(expected)

    def test_fifty_percent(self) -> None:
        assert compute_k_alpha(0.50) == pytest.approx(1.0)

    def test_ninety_five_percent(self) -> None:
        expected = math.sqrt(0.05 / 0.95)
        assert compute_k_alpha(0.95) == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — compute_nrr_alpha
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeNrrAlpha:

    def test_basic_formula(self) -> None:
        med, std, mtbf, alpha = 6.0, 2.0, 7.0, 0.90
        k = math.sqrt(0.1 / 0.9)
        expected = 1.0 - (1.0 / mtbf) * (med + k * std)
        assert compute_nrr_alpha(med, std, mtbf, alpha) == pytest.approx(expected)

    def test_zero_mtbf_returns_nan(self) -> None:
        assert math.isnan(compute_nrr_alpha(6.0, 2.0, 0.0, 0.90))

    def test_zero_std_degrades_to_nrr(self) -> None:
        med, mtbf = 6.0, 7.0
        nrr = compute_nrr(med, mtbf)
        nrr_a = compute_nrr_alpha(med, 0.0, mtbf, 0.90)
        assert nrr_a == pytest.approx(nrr)


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — compute_pi_up
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputePiUp:

    def test_basic_formula(self) -> None:
        assert compute_pi_up(7.0, 6.0) == pytest.approx(7.0 / 13.0)

    def test_zero_denominator_returns_zero(self) -> None:
        assert compute_pi_up(0.0, 0.0) == 0.0

    def test_zero_recovery_time(self) -> None:
        assert compute_pi_up(5.0, 0.0) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "mtbf,med_ttr_a",
        [(6.73, 6.21), (10.0, 2.0), (4.0, 1.0), (100.0, 50.0), (1.0, 0.5)],
    )
    def test_theorem1_pi_up_ge_nrr(self, mtbf: float, med_ttr_a: float) -> None:
        """Theorem 1: π_up ≥ NRR for all valid (mtbf, med_ttr_a) pairs."""
        pi_up = compute_pi_up(mtbf, med_ttr_a)
        nrr = compute_nrr(med_ttr_a, mtbf)
        assert pi_up >= nrr, f"Theorem 1 violated: π_up={pi_up}, NRR={nrr}"


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — compute_rolling_median
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeRollingMedian:

    def test_basic_window_of_three(self) -> None:
        result = compute_rolling_median([1.0, 2.0, 3.0, 4.0, 5.0], window=3)
        assert result == [2.0, 3.0, 4.0]

    def test_insufficient_data_returns_empty(self) -> None:
        assert compute_rolling_median([1.0, 2.0], window=3) == []

    def test_exact_window_size_returns_one_value(self) -> None:
        result = compute_rolling_median([1.0, 2.0, 9.0], window=3)
        assert result == [2.0]

    def test_window_of_one(self) -> None:
        result = compute_rolling_median([3.0, 5.0, 7.0], window=1)
        assert result == [3.0, 5.0, 7.0]

    def test_empty_input_returns_empty(self) -> None:
        assert compute_rolling_median([], window=20) == []


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — compute_latency_decomposition
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeLatencyDecomposition:

    def test_basic_means(self) -> None:
        eps = [_episode(t_detect=0.5, t_decide=0.1, t_execute=5.0)] * 3
        result = compute_latency_decomposition(eps)
        assert result.t_detect_mean == pytest.approx(0.5)
        assert result.t_decide_mean == pytest.approx(0.1)
        assert result.t_execute_mean == pytest.approx(5.0)

    def test_returns_latency_decomposition_type(self) -> None:
        result = compute_latency_decomposition([_episode()])
        assert isinstance(result, LatencyDecomposition)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_latency_decomposition([])

    def test_values_are_rounded_to_three_decimals(self) -> None:
        eps = [_episode(t_detect=0.1234567, t_decide=0.0987654, t_execute=5.9876543)]
        result = compute_latency_decomposition(eps)
        assert result.t_detect_mean == pytest.approx(0.123, abs=1e-3)
        assert result.t_decide_mean == pytest.approx(0.099, abs=1e-3)
        assert result.t_execute_mean == pytest.approx(5.988, abs=1e-3)


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — compute_per_mode_stats
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputePerModeStats:

    def test_empty_episodes_returns_empty_dict(self) -> None:
        assert compute_per_mode_stats([]) == {}

    def test_single_episode_std_is_zero(self) -> None:
        eps = [_episode(reflex_mode="auto-replan", delta_t=5.94)]
        result = compute_per_mode_stats(eps)
        assert result["auto-replan"].std == 0.0

    def test_multiple_episodes_count(self) -> None:
        eps = [_episode(reflex_mode="tool-retry", delta_t=4.0 + i * 0.1) for i in range(5)]
        result = compute_per_mode_stats(eps)
        assert result["tool-retry"].count == 5

    def test_only_modes_with_data_appear(self) -> None:
        eps = [_episode(reflex_mode="rollback", delta_t=7.0)]
        result = compute_per_mode_stats(eps)
        assert set(result.keys()) == {"rollback"}
        assert "auto-replan" not in result

    def test_p90_index_correct(self) -> None:
        # sorted[int(0.9 * 10)] = sorted[9] = 10.0 (int-floor indexing)
        dts = [float(i) for i in range(1, 11)]  # 1..10
        eps = [_episode(reflex_mode="rollback", delta_t=d) for d in dts]
        result = compute_per_mode_stats(eps)
        assert result["rollback"].p90 == 10.0

    def test_returns_mode_stats_type(self) -> None:
        eps = [_episode(reflex_mode="human-approve", delta_t=12.0)]
        result = compute_per_mode_stats(eps)
        assert isinstance(result["human-approve"], ModeStats)


# ═══════════════════════════════════════════════════════════════════════════════
# Functional tests — Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class TestPipeline:

    def test_run_returns_episode_and_positive_stable(self) -> None:
        rng = random.Random(1)
        pipeline = Pipeline(rng, force_drift=True)
        ep, stable = pipeline.run(run_id=0, wall_clock=0.0)
        assert isinstance(ep, Episode)
        assert stable > 0.0

    def test_forced_drift_episode_has_nonzero_delta_t(self) -> None:
        rng = random.Random(1)
        ep, _ = Pipeline(rng, force_drift=True).run(0, 0.0)
        assert ep.drift_detected is True
        assert ep.delta_t > 0.0
        assert ep.reflex_mode in REFLEX_PARAMS

    def test_no_drift_episode_has_zero_latencies(self) -> None:
        rng = MagicMock()
        rng.choice.return_value = "LangGraph recovery reflexes"
        rng.gauss.return_value = 0.8   # high confidence, no natural drift
        rng.random.return_value = 0.99  # no stochastic drift
        rng.expovariate.return_value = 5.0  # stable interval
        ep, stable = Pipeline(rng, force_drift=False).run(0, 10.0)
        assert ep.drift_detected is False
        assert ep.delta_t == 0.0
        assert ep.reflex_mode is None
        assert stable == pytest.approx(5.0)

    def test_wall_clock_preserved_in_t_fault(self) -> None:
        rng = random.Random(42)
        ep, _ = Pipeline(rng, force_drift=True).run(0, wall_clock=100.0)
        assert ep.t_fault == pytest.approx(100.0)

    def test_t_recovered_equals_t_fault_plus_delta_t(self) -> None:
        rng = random.Random(7)
        ep, _ = Pipeline(rng, force_drift=True).run(0, 0.0)
        assert ep.t_recovered == pytest.approx(ep.t_fault + ep.delta_t, abs=1e-3)

    def test_run_id_stored_in_episode(self) -> None:
        rng = random.Random(3)
        ep, _ = Pipeline(rng).run(run_id=17, wall_clock=0.0)
        assert ep.run_id == 17

    def test_deterministic_with_same_seed(self) -> None:
        ep_a, s_a = Pipeline(random.Random(99)).run(0, 0.0)
        ep_b, s_b = Pipeline(random.Random(99)).run(0, 0.0)
        assert ep_a == ep_b
        assert s_a == pytest.approx(s_b)

    def test_custom_mtbf_mean_used_for_stable_interval(self) -> None:
        results = []
        for seed in range(50):
            rng = random.Random(seed)
            _, stable = Pipeline(rng, mtbf_mean=1.0).run(0, 0.0)
            results.append(stable)
        assert sum(results) / len(results) == pytest.approx(1.0, rel=0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# Functional tests — Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrchestrator:

    def test_run_returns_correct_episode_count(self) -> None:
        orch = Orchestrator(n_runs=10, seed=0, verbose=False)
        episodes = orch.run()
        assert len(episodes) == 10

    def test_force_drift_all_episodes_have_drift(self) -> None:
        orch = Orchestrator(n_runs=20, seed=0, force_drift=True, verbose=False)
        episodes = orch.run()
        assert all(e.drift_detected for e in episodes)

    def test_no_force_drift_produces_some_non_drift(self) -> None:
        orch = Orchestrator(n_runs=200, seed=42, force_drift=False, verbose=False)
        episodes = orch.run()
        assert any(not e.drift_detected for e in episodes)

    def test_stable_intervals_length_matches_runs(self) -> None:
        orch = Orchestrator(n_runs=15, seed=1, verbose=False)
        orch.run()
        assert len(orch.stable_intervals) == 15

    def test_stable_intervals_all_positive(self) -> None:
        orch = Orchestrator(n_runs=30, seed=2, verbose=False)
        orch.run()
        assert all(s > 0 for s in orch.stable_intervals)

    def test_deterministic_across_identical_seeds(self) -> None:
        eps_a = Orchestrator(n_runs=10, seed=77, verbose=False).run()
        eps_b = Orchestrator(n_runs=10, seed=77, verbose=False).run()
        assert eps_a == eps_b

    def test_different_seeds_produce_different_results(self) -> None:
        eps_a = Orchestrator(n_runs=20, seed=1, verbose=False).run()
        eps_b = Orchestrator(n_runs=20, seed=2, verbose=False).run()
        assert eps_a != eps_b

    def test_verbose_output_emitted(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            Orchestrator(n_runs=5, seed=0, verbose=True).run()
        output = buf.getvalue()
        assert "Run" in output
        assert "Drift" in output

    def test_run_ids_are_sequential(self) -> None:
        orch = Orchestrator(n_runs=5, seed=3, verbose=False)
        episodes = orch.run()
        assert [e.run_id for e in episodes] == list(range(5))


# ═══════════════════════════════════════════════════════════════════════════════
# Functional tests — MetricsComputer
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetricsComputer:

    def _run_small_benchmark(
        self, n: int = 30, force_drift: bool = True
    ) -> tuple[list[Episode], list[float]]:
        orch = Orchestrator(n_runs=n, seed=42, force_drift=force_drift, verbose=False)
        episodes = orch.run()
        return episodes, orch.stable_intervals

    def test_no_drift_events_raises(self) -> None:
        episodes = [_episode(drift_detected=False, reflex_mode=None, delta_t=0.0)]
        with pytest.raises(ValueError, match="No drift events"):
            MetricsComputer().compute(episodes, stable_intervals=[5.0], n_runs=1)

    def test_returns_system_metrics_type(self) -> None:
        eps, stab = self._run_small_benchmark()
        result = MetricsComputer().compute(eps, stab, n_runs=30)
        assert isinstance(result, SystemMetrics)

    def test_drift_events_count_is_correct(self) -> None:
        eps, stab = self._run_small_benchmark(force_drift=True)
        result = MetricsComputer().compute(eps, stab, n_runs=30)
        assert result.drift_events == 30

    def test_drift_rate_force_true_is_one(self) -> None:
        eps, stab = self._run_small_benchmark(force_drift=True)
        result = MetricsComputer().compute(eps, stab, n_runs=30)
        assert result.drift_rate == pytest.approx(1.0)

    def test_theorem1_invariant_pi_up_ge_nrr(self) -> None:
        eps, stab = self._run_small_benchmark(n=50)
        m = MetricsComputer().compute(eps, stab, n_runs=50)
        assert m.pi_up_sys >= m.nrr_sys, "Theorem 1 violated in MetricsComputer output"

    def test_med_ttr_a_within_plausible_range(self) -> None:
        eps, stab = self._run_small_benchmark()
        m = MetricsComputer().compute(eps, stab, n_runs=30)
        assert 1.0 < m.med_ttr_a_sys < 20.0

    def test_per_mode_counts_do_not_exceed_drift_events(self) -> None:
        eps, stab = self._run_small_benchmark(n=100)
        m = MetricsComputer().compute(eps, stab, n_runs=100)
        total_in_modes = sum(s.count for s in m.per_mode.values())
        assert total_in_modes <= m.drift_events

    def test_rolling_median_length(self) -> None:
        eps, stab = self._run_small_benchmark(n=50)
        m = MetricsComputer().compute(eps, stab, n_runs=50)
        # drift_events - ROLLING_WINDOW + 1 entries expected
        expected_len = max(0, 50 - ROLLING_WINDOW + 1)
        assert len(m.rolling_med_ttr_a) == expected_len

    def test_custom_alpha_stored(self) -> None:
        eps, stab = self._run_small_benchmark()
        m = MetricsComputer(alpha=0.95).compute(eps, stab, n_runs=30)
        assert m.alpha == pytest.approx(0.95)

    def test_n_runs_stored_correctly(self) -> None:
        eps, stab = self._run_small_benchmark(n=25)
        m = MetricsComputer().compute(eps, stab, n_runs=25)
        assert m.n_runs == 25


# ═══════════════════════════════════════════════════════════════════════════════
# Functional tests — TelemetryLogger
# ═══════════════════════════════════════════════════════════════════════════════


class TestTelemetryLogger:

    def test_to_records_field_names(self) -> None:
        ep = _episode()
        records = TelemetryLogger.to_records([ep])
        assert len(records) == 1
        rec = records[0]
        expected_keys = {
            "run_id", "query", "confidence", "drift_detected",
            "reflex_mode", "t_detect", "t_decide", "t_execute", "delta_t",
        }
        assert set(rec.keys()) == expected_keys

    def test_to_records_values_match_episode(self) -> None:
        ep = _episode(run_id=7, t_detect=0.5, delta_t=5.94)
        rec = TelemetryLogger.to_records([ep])[0]
        assert rec["run_id"] == 7
        assert rec["t_detect"] == 0.5
        assert rec["delta_t"] == 5.94

    def test_to_records_empty_episodes(self) -> None:
        assert TelemetryLogger.to_records([]) == []

    def test_save_creates_jsonl_file(self, tmp_path: Any) -> None:
        path = str(tmp_path / "tel.jsonl")
        TelemetryLogger().save([_episode()], path)
        with open(path) as fh:
            lines = fh.readlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["run_id"] == 0

    def test_save_each_line_is_valid_json(self, tmp_path: Any) -> None:
        episodes = [_episode(run_id=i) for i in range(5)]
        path = str(tmp_path / "tel.jsonl")
        TelemetryLogger().save(episodes, path)
        with open(path) as fh:
            for line in fh:
                obj = json.loads(line)
                assert "run_id" in obj

    def test_save_prints_confirmation(self, tmp_path: Any, capsys: Any) -> None:
        path = str(tmp_path / "tel.jsonl")
        TelemetryLogger().save([_episode()], path)
        out = capsys.readouterr().out
        assert "Telemetry" in out
        assert "1 records" in out

    def test_none_reflex_mode_serialises_as_null(self, tmp_path: Any) -> None:
        ep = _episode(drift_detected=False, reflex_mode=None, delta_t=0.0)
        path = str(tmp_path / "tel.jsonl")
        TelemetryLogger().save([ep], path)
        with open(path) as fh:
            obj = json.loads(fh.readline())
        assert obj["reflex_mode"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# Functional tests — ResultsSaver
# ═══════════════════════════════════════════════════════════════════════════════


class TestResultsSaver:

    def test_save_creates_valid_json_file(self, tmp_path: Any) -> None:
        path = str(tmp_path / "results.json")
        ResultsSaver().save(_metrics(), path)
        with open(path) as fh:
            data = json.load(fh)
        assert "med_ttr_a_sys" in data

    def test_to_dict_contains_all_top_level_keys(self) -> None:
        d = ResultsSaver.to_dict(_metrics())
        expected = {
            "n_runs", "drift_events", "drift_rate", "mttr_a_sys", "med_ttr_a_sys",
            "std_sys", "p90_sys", "mtbf_sys", "nrr_sys", "pi_up_sys", "nrr_alpha",
            "alpha", "k_alpha", "latency_decomposition", "per_mode", "rolling_med_ttr_a",
        }
        assert set(d.keys()) == expected

    def test_to_dict_latency_decomposition_structure(self) -> None:
        d = ResultsSaver.to_dict(_metrics())
        decomp = d["latency_decomposition"]
        assert isinstance(decomp, dict)
        assert {"t_detect_mean", "t_decide_mean", "t_execute_mean"} == set(decomp.keys())

    def test_to_dict_per_mode_structure(self) -> None:
        d = ResultsSaver.to_dict(_metrics())
        per_mode = d["per_mode"]
        for mode_data in per_mode.values():
            assert set(mode_data.keys()) == {"count", "median", "std", "p90"}

    def test_save_prints_confirmation(self, tmp_path: Any, capsys: Any) -> None:
        path = str(tmp_path / "results.json")
        ResultsSaver().save(_metrics(), path)
        assert "Results" in capsys.readouterr().out

    def test_roundtrip_preserves_scalar_values(self, tmp_path: Any) -> None:
        m = _metrics()
        path = str(tmp_path / "r.json")
        ResultsSaver().save(m, path)
        with open(path) as fh:
            data = json.load(fh)
        assert data["med_ttr_a_sys"] == m.med_ttr_a_sys
        assert data["nrr_sys"] == m.nrr_sys


# ═══════════════════════════════════════════════════════════════════════════════
# Functional tests — Reporter
# ═══════════════════════════════════════════════════════════════════════════════


class TestReporter:

    def _capture(self, metrics: SystemMetrics) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            Reporter().print_report(metrics)
        return buf.getvalue()

    def test_print_report_produces_output(self) -> None:
        output = self._capture(_metrics())
        assert len(output) > 100

    def test_print_report_contains_key_labels(self) -> None:
        output = self._capture(_metrics())
        for label in ("MedTTR-A", "MTTR-A", "MTBF", "NRR", "Paper"):
            assert label in output, f"Missing label: {label}"

    def test_pi_up_ge_nrr_shows_checkmark(self) -> None:
        m = _metrics(pi_up_sys=0.6, nrr_sys=0.5)
        assert "✓" in self._capture(m)

    def test_pi_up_lt_nrr_no_checkmark(self) -> None:
        m = _metrics(pi_up_sys=0.1, nrr_sys=0.5)
        output = self._capture(m)
        assert "✓" not in output

    def test_empty_rolling_median_suppresses_section(self) -> None:
        m = _metrics(rolling_med_ttr_a=[])
        output = self._capture(m)
        assert "Rolling MedTTR-A" not in output

    def test_rolling_median_section_shown_when_present(self) -> None:
        m = _metrics(rolling_med_ttr_a=[5.8, 6.0, 6.2])
        output = self._capture(m)
        assert "Rolling MedTTR-A" in output

    def test_zero_total_latency_skips_percentages(self) -> None:
        m = _metrics(
            latency_decomposition=LatencyDecomposition(
                t_detect_mean=0.0, t_decide_mean=0.0, t_execute_mean=0.0
            )
        )
        # Should not raise ZeroDivisionError
        self._capture(m)

    def test_per_mode_rows_appear_in_output(self) -> None:
        output = self._capture(_metrics())
        assert "auto-replan" in output
        assert "tool-retry" in output


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:

    def _full_run(
        self, n: int = 50, seed: int = 42, force_drift: bool = True
    ) -> tuple[list[Episode], SystemMetrics]:
        orch = Orchestrator(n_runs=n, seed=seed, force_drift=force_drift, verbose=False)
        episodes = orch.run()
        metrics = MetricsComputer().compute(episodes, orch.stable_intervals, n_runs=n)
        return episodes, metrics

    def test_full_pipeline_produces_valid_metrics(self) -> None:
        _, m = self._full_run()
        assert m.drift_events > 0
        assert m.med_ttr_a_sys > 0
        assert m.mtbf_sys > 0

    def test_force_drift_all_episodes_recovered(self) -> None:
        episodes, _ = self._full_run(force_drift=True)
        assert all(e.drift_detected and e.delta_t > 0 for e in episodes)

    def test_theorem1_holds_end_to_end(self) -> None:
        _, m = self._full_run(n=100)
        assert m.pi_up_sys >= m.nrr_sys

    def test_paper_comparison_med_ttr_a_within_15pct(self) -> None:
        _, m = self._full_run(n=200, seed=42)
        paper_value = _PAPER_METRICS["med_ttr_a_s"]
        assert abs(m.med_ttr_a_sys - paper_value) / paper_value < 0.15

    def test_paper_comparison_mtbf_within_20pct(self) -> None:
        _, m = self._full_run(n=200, seed=42)
        paper_value = _PAPER_METRICS["mtbf_s"]
        assert abs(m.mtbf_sys - paper_value) / paper_value < 0.20

    def test_telemetry_jsonl_roundtrip(self, tmp_path: Any) -> None:
        episodes, _ = self._full_run(n=20)
        path = str(tmp_path / "tel.jsonl")
        TelemetryLogger().save(episodes, path)
        with open(path) as fh:
            loaded = [json.loads(line) for line in fh]
        assert len(loaded) == 20
        assert loaded[0]["run_id"] == 0
        assert loaded[-1]["run_id"] == 19

    def test_results_json_roundtrip(self, tmp_path: Any) -> None:
        _, metrics = self._full_run(n=20)
        path = str(tmp_path / "results.json")
        ResultsSaver().save(metrics, path)
        with open(path) as fh:
            data = json.load(fh)
        assert data["med_ttr_a_sys"] == metrics.med_ttr_a_sys
        assert data["n_runs"] == 20

    def test_different_seeds_yield_different_med_ttr_a(self) -> None:
        _, m1 = self._full_run(seed=1)
        _, m2 = self._full_run(seed=999)
        assert m1.med_ttr_a_sys != m2.med_ttr_a_sys

    def test_non_force_drift_fewer_drift_events_than_runs(self) -> None:
        orch = Orchestrator(n_runs=500, seed=42, force_drift=False, verbose=False)
        episodes = orch.run()
        drift_count = sum(1 for e in episodes if e.drift_detected)
        assert drift_count < 500
        assert drift_count > 0

    def test_main_function_runs_and_writes_files(self, tmp_path: Any) -> None:
        tel = str(tmp_path / "tel.jsonl")
        res = str(tmp_path / "res.json")
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(n_runs=10, seed=1, telemetry_path=tel, results_path=res)
        assert "MTTR-A" in buf.getvalue()
        with open(tel) as fh:
            lines = fh.readlines()
        assert len(lines) == 10
        with open(res) as fh:
            data = json.load(fh)
        assert data["n_runs"] == 10

    def test_dunder_main_guard_executes_main(self, tmp_path: Any, monkeypatch: Any) -> None:
        """
        Line 653 — verify if __name__ == '__main__': main() is reachable.
        runpy.run_path executes the guard in a fresh namespace; we verify that
        main() ran by checking its file output.
        """
        monkeypatch.chdir(tmp_path)
        sim_path = str(Path(__file__).parent / "mttr_a_simulation.py")
        buf = io.StringIO()
        with redirect_stdout(buf):
            runpy.run_path(sim_path, run_name="__main__")
        assert (tmp_path / "telemetry.jsonl").exists()
        assert (tmp_path / "results.json").exists()
        assert "MTTR-A" in buf.getvalue()

    def test_reporter_output_stable_section_present_for_large_run(self) -> None:
        _, m = self._full_run(n=100)
        buf = io.StringIO()
        with redirect_stdout(buf):
            Reporter().print_report(m)
        assert "Rolling MedTTR-A" in buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# Data model tests — dataclass contracts
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataModels:

    def test_episode_is_frozen(self) -> None:
        ep = _episode()
        with pytest.raises((AttributeError, TypeError)):
            ep.run_id = 99  # type: ignore[misc]

    def test_reflex_config_is_frozen(self) -> None:
        cfg = ReflexConfig(weight=10, median=5.0, std=1.0)
        with pytest.raises((AttributeError, TypeError)):
            cfg.weight = 99  # type: ignore[misc]

    def test_system_metrics_is_frozen(self) -> None:
        m = _metrics()
        with pytest.raises((AttributeError, TypeError)):
            m.n_runs = 999  # type: ignore[misc]

    def test_latency_decomposition_is_frozen(self) -> None:
        ld = LatencyDecomposition(0.5, 0.15, 5.0)
        with pytest.raises((AttributeError, TypeError)):
            ld.t_detect_mean = 9.9  # type: ignore[misc]

    def test_mode_stats_is_frozen(self) -> None:
        ms = ModeStats(count=10, median=5.0, std=1.0, p90=6.5)
        with pytest.raises((AttributeError, TypeError)):
            ms.count = 0  # type: ignore[misc]

    def test_recovery_result_is_named_tuple(self) -> None:
        r = RecoveryResult("auto-replan", 0.5, 0.1, 5.0)
        assert r.mode == "auto-replan"
        assert r[0] == "auto-replan"

    def test_constants_have_expected_types(self) -> None:
        assert isinstance(QUERY_POOL, list)
        assert all(isinstance(q, str) for q in QUERY_POOL)
        assert isinstance(TAU_DRIFT, float)
        assert isinstance(DEFAULT_ALPHA, float)
        assert isinstance(ROLLING_WINDOW, int)

    def test_reflex_params_has_four_modes(self) -> None:
        assert len(REFLEX_PARAMS) == 4
        assert all(isinstance(v, ReflexConfig) for v in REFLEX_PARAMS.values())


class TestDataLoading:
    """Verify that all simulation data is loaded from files, not hardcoded."""

    def test_query_pool_loads_from_file(self) -> None:
        loaded = _load_query_pool()
        assert isinstance(loaded, list)
        assert len(loaded) > 0
        assert all(isinstance(q, str) and q for q in loaded)

    def test_query_pool_matches_module_constant(self) -> None:
        assert _load_query_pool() == QUERY_POOL

    def test_reflex_params_loads_from_file(self) -> None:
        loaded = _load_reflex_params()
        assert isinstance(loaded, dict)
        assert len(loaded) == 4
        assert all(isinstance(v, ReflexConfig) for v in loaded.values())

    def test_reflex_params_matches_module_constant(self) -> None:
        assert _load_reflex_params() == REFLEX_PARAMS

    def test_reflex_params_all_fields_positive(self) -> None:
        for name, cfg in _load_reflex_params().items():
            assert cfg.weight > 0, f"{name}: weight must be positive"
            assert cfg.median > 0, f"{name}: median must be positive"
            assert cfg.std > 0, f"{name}: std must be positive"

    def test_mock_config_loads_from_file(self) -> None:
        cfg = _load_mock_config()
        assert "confidence_distribution" in cfg
        assert "recovery_timing" in cfg
        assert "mock_provider_latency" in cfg
        assert "ttft_fraction" in cfg

    def test_mock_config_confidence_distribution_valid(self) -> None:
        conf = _load_mock_config()["confidence_distribution"]
        assert 0.0 < conf["base_mu"] < 1.0
        assert conf["base_sigma"] > 0
        assert conf["noise_sigma"] >= 0

    def test_mock_config_recovery_timing_positive(self) -> None:
        timing = _load_mock_config()["recovery_timing"]
        assert timing["t_detect_mean_s"] > 0
        assert timing["t_decide_mean_s"] > 0
        assert timing["t_execute_min_s"] >= 0

    def test_mock_config_latency_ranges_valid(self) -> None:
        for ctx, rng in _load_mock_config()["mock_provider_latency"].items():
            if ctx.startswith("_"):
                continue
            lo, hi = rng
            assert lo >= 0, f"context '{ctx}': min latency must be non-negative"
            assert hi > lo, f"context '{ctx}': max latency must exceed min"

    def test_mock_config_ttft_fraction_in_unit_interval(self) -> None:
        value = _load_mock_config()["ttft_fraction"]["value"]
        assert 0.0 < value < 1.0

    def test_paper_benchmarks_loads_from_file(self) -> None:
        paper = _load_paper_benchmarks()
        assert "metrics" in paper
        assert "med_ttr_a_s" in paper["metrics"]
        assert "mtbf_s" in paper["metrics"]
        assert "nrr" in paper["metrics"]

    def test_paper_benchmarks_values_positive(self) -> None:
        m = _load_paper_benchmarks()["metrics"]
        assert m["med_ttr_a_s"] > 0
        assert m["mtbf_s"] > 0
        assert 0.0 <= m["nrr"] <= 1.0

    def test_paper_benchmarks_match_module_constant(self) -> None:
        assert _load_paper_benchmarks()["metrics"] == _PAPER_METRICS
