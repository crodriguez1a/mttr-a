#!/usr/bin/env python3
"""
MTTR-A: Measuring Cognitive Recovery Latency in Multi-Agent Systems
Mock simulation — Barak Or (2025), arXiv:2511.20663v5

Single-responsibility layers
  Nodes:            reasoning_node, check_drift_node, recovery_node  (pure)
  Metric functions: compute_*                                         (pure)
  Pipeline:         one 3-node run                                    (stateful per run)
  Orchestrator:     repeats Pipeline N times                          (stateful)
  MetricsComputer:  derives SystemMetrics from completed episodes      (stateless)
  TelemetryLogger:  serialises episodes → JSONL                       (I/O)
  ResultsSaver:     serialises SystemMetrics → JSON                   (I/O)
  Reporter:         formats console output                            (I/O)
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

# ── Data directory ────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent / "data"

# ── Typed configuration ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReflexConfig:
    """Calibration parameters for one reflex mode (from Table II)."""

    weight: int    # Sampling weight (proportional to occurrence count)
    median: float  # Target median recovery time (s)
    std: float     # Target standard deviation of recovery time (s)


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_query_pool() -> list[str]:
    path = _DATA_DIR / "query_pool" / "queries.txt"
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]


def _load_reflex_params() -> dict[str, ReflexConfig]:
    data = json.loads((_DATA_DIR / "reflex_params" / "params.json").read_text())
    return {
        name: ReflexConfig(
            weight=r["weight"],
            median=r["median_latency_s"],
            std=r["std_latency_s"],
        )
        for name, r in data["reflexes"].items()
    }


def _load_mock_config() -> dict:
    return json.loads((_DATA_DIR / "mock_config" / "config.json").read_text())


def _load_paper_benchmarks() -> dict:
    return json.loads((_DATA_DIR / "paper_benchmarks" / "benchmarks.json").read_text())


# ── Constants (loaded from data files) ───────────────────────────────────────

QUERY_POOL: list[str] = _load_query_pool()
REFLEX_PARAMS: dict[str, ReflexConfig] = _load_reflex_params()

TAU_DRIFT: float = 0.6       # Confidence threshold for drift detection (Section IV-D)
DEFAULT_MTBF_MEAN: float = 6.73  # Mean stable interval between faults (s)
DEFAULT_ALPHA: float = 0.90  # Confidence level for NRR_α (Theorem 2)
ROLLING_WINDOW: int = 20     # Window size for temporal stability analysis

_MOCK_CFG = _load_mock_config()
_CONF = _MOCK_CFG["confidence_distribution"]
_TIMING = _MOCK_CFG["recovery_timing"]
_CONF_BASE_MU: float     = _CONF["base_mu"]
_CONF_BASE_SIGMA: float  = _CONF["base_sigma"]
_CONF_NOISE_MU: float    = _CONF["noise_mu"]
_CONF_NOISE_SIGMA: float = _CONF["noise_sigma"]
_T_DETECT_MEAN: float    = _TIMING["t_detect_mean_s"]
_T_DECIDE_MEAN: float    = _TIMING["t_decide_mean_s"]
_T_EXECUTE_MIN: float    = _TIMING["t_execute_min_s"]

_PAPER = _load_paper_benchmarks()
_PAPER_METRICS = _PAPER["metrics"]

# ── Data models ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Episode:
    """One reasoning-drift-recovery cycle (one 'run' in the paper, Eq. 1)."""

    run_id: int
    query: str
    confidence: float       # Cosine similarity score c (Eq. 13)
    drift_detected: bool    # True when c < τ_drift or stochastic perturbation
    reflex_mode: str | None  # None when no drift occurred
    t_detect: float         # Drift detection latency (Eq. 5)
    t_decide: float         # Policy-selection delay (Eq. 5, negligible)
    t_execute: float        # Reflex execution time (Eq. 5, dominant)
    delta_t: float          # Total Δt = t_detect + t_decide + t_execute (Eq. 1)
    t_fault: float          # Wall-clock onset of fault t_f (Eq. 1)
    t_recovered: float      # Wall-clock restoration t_r (Eq. 1)
    step_confidences: tuple[float, ...] = ()   # groundedness score at each reasoning step


class RecoveryResult(NamedTuple):
    """Structured return type for recovery_node."""

    mode: str
    t_detect: float
    t_decide: float
    t_execute: float


@dataclass(frozen=True)
class LatencyDecomposition:
    """Mean latency for each additive component of Δt (Eq. 5)."""

    t_detect_mean: float
    t_decide_mean: float
    t_execute_mean: float


@dataclass(frozen=True)
class ModeStats:
    """Per-reflex-mode recovery statistics (Table II)."""

    count: int
    median: float
    std: float
    p90: float


@dataclass(frozen=True)
class SystemMetrics:
    """All system-level reliability metrics (Eqs. 2–12, Theorems 1–2)."""

    n_runs: int
    drift_events: int
    drift_rate: float
    mttr_a_sys: float        # Mean time-to-recovery (Eq. 3)
    med_ttr_a_sys: float     # Robust median estimator (Eq. 4, Algorithm 1 line 9)
    std_sys: float
    p90_sys: float
    mtbf_sys: float          # Mean time between cognitive faults (Eq. 7)
    nrr_sys: float           # Normalized recovery ratio (Eq. 8)
    pi_up_sys: float         # Steady-state cognitive uptime (Theorem 1, Eq. 10)
    nrr_alpha: float         # Confidence-aware NRR (Theorem 2, Eq. 11)
    alpha: float
    k_alpha: float           # Cantelli coefficient
    latency_decomposition: LatencyDecomposition
    per_mode: dict[str, ModeStats]
    rolling_med_ttr_a: list[float]


# ── Pure node functions ───────────────────────────────────────────────────────


def reasoning_node(query: str, rng: random.Random) -> float:
    """
    Mock AG News document retrieval returning confidence c ∈ [0, 1].
    Paper: c = cos(q, d_top) = q·d_top / (||q|| ||d_top||)  (Eq. 13).
    Mock: N(0.65, 0.15) + N(0, 0.05) noise, clamped to [0, 1].
    """
    base: float = rng.gauss(mu=_CONF_BASE_MU, sigma=_CONF_BASE_SIGMA)
    noise: float = rng.gauss(mu=_CONF_NOISE_MU, sigma=_CONF_NOISE_SIGMA)
    return max(0.0, min(1.0, base + noise))


def check_drift_node(
    confidence: float, rng: random.Random, force: bool = False
) -> bool:
    """
    Flag cognitive fault when c < τ_drift or via 5% stochastic perturbation.
    force=True matches the paper's benchmark (all 200 runs are recovery episodes,
    Table II counts sum to exactly 200).
    """
    if force:
        return True
    return confidence < TAU_DRIFT or rng.random() < 0.05


def recovery_node(rng: random.Random) -> RecoveryResult:
    """
    Select reflex mode by weighted policy sampling (Section III-A).
    Simulate additive latency Δt = t_detect + t_decide + t_execute (Eq. 5).
    Execution is dominant; decision is negligible (Section V).
    """
    modes: list[str] = list(REFLEX_PARAMS.keys())
    weights: list[int] = [REFLEX_PARAMS[m].weight for m in modes]
    (mode,) = rng.choices(modes, weights=weights, k=1)

    t_detect: float = rng.expovariate(1.0 / _T_DETECT_MEAN)
    t_decide: float = rng.expovariate(1.0 / _T_DECIDE_MEAN)

    cfg = REFLEX_PARAMS[mode]
    target_exec: float = cfg.median - t_detect - t_decide
    t_execute: float = max(_T_EXECUTE_MIN, rng.gauss(target_exec, cfg.std))

    return RecoveryResult(mode=mode, t_detect=t_detect, t_decide=t_decide, t_execute=t_execute)


# ── Pure metric functions ─────────────────────────────────────────────────────


def compute_mttr_a(recovery_times: list[float]) -> float:
    """Mean time-to-recovery across all fault episodes (Eq. 2–3)."""
    if not recovery_times:
        raise ValueError("Cannot compute MTTR-A: recovery_times is empty")
    return statistics.mean(recovery_times)


def compute_med_ttr_a(recovery_times: list[float]) -> float:
    """Robust median estimator for MTTR-A (Eq. 4, Algorithm 1 line 9)."""
    if not recovery_times:
        raise ValueError("Cannot compute MedTTR-A: recovery_times is empty")
    return statistics.median(recovery_times)


def compute_std(values: list[float]) -> float:
    """Sample standard deviation; returns 0.0 for fewer than two values."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def compute_percentile(values: list[float], pct: float) -> float:
    """Return the value at percentile pct ∈ (0, 1) of a non-empty sequence."""
    if not values:
        raise ValueError("Cannot compute percentile of an empty sequence")
    return sorted(values)[int(pct * len(values))]


def compute_mtbf(stable_intervals: list[float]) -> float:
    """Mean time between cognitive faults (Eq. 6–7); 0.0 when no intervals."""
    if not stable_intervals:
        return 0.0
    return statistics.mean(stable_intervals)


def compute_nrr(med_ttr_a: float, mtbf: float) -> float:
    """Normalized recovery ratio (Eq. 8); nan when MTBF is zero."""
    if mtbf == 0.0:
        return math.nan
    return 1.0 - med_ttr_a / mtbf


def compute_k_alpha(alpha: float) -> float:
    """Cantelli coefficient k_α = sqrt((1-α)/α) for Theorem 2."""
    return math.sqrt((1.0 - alpha) / alpha)


def compute_nrr_alpha(
    med_ttr_a: float, std: float, mtbf: float, alpha: float
) -> float:
    """Confidence-aware NRR_α (Theorem 2, Eq. 11); nan when MTBF is zero."""
    if mtbf == 0.0:
        return math.nan
    k = compute_k_alpha(alpha)
    r_alpha = med_ttr_a + k * std
    return 1.0 - (1.0 / mtbf) * r_alpha


def compute_pi_up(mtbf: float, med_ttr_a: float) -> float:
    """Steady-state cognitive uptime fraction (Theorem 1, Eq. 10); 0.0 when undefined."""
    denom = mtbf + med_ttr_a
    if denom == 0.0:
        return 0.0
    return mtbf / denom


def infer_stable_intervals(episodes: list[Episode]) -> list[float]:
    """
    Derive stable intervals from real episode timestamps (Gap 1).

    Use this instead of synthetic Exp-sampling when you have actual traffic data.
    Each stable interval is the gap between one fault resolving and the next one
    beginning: stable[i] = drift[i+1].t_fault − drift[i].t_recovered

    Returns one fewer value than the number of drift episodes (there is no
    interval after the final fault).  Returns [] when fewer than two drift
    episodes are present.

    In a back-to-back benchmark the values will be near-zero (just Python
    overhead); this function is intended for post-hoc analysis of real traffic
    logs where t_fault and t_recovered carry genuine wall-clock timestamps.
    """
    drift = [ep for ep in episodes if ep.drift_detected]
    if len(drift) < 2:
        return []
    return [
        max(0.0, drift[i + 1].t_fault - drift[i].t_recovered)
        for i in range(len(drift) - 1)
    ]


def compute_rolling_median(values: list[float], window: int) -> list[float]:
    """Rolling median over a sliding window; returns [] when len(values) < window."""
    if len(values) < window:
        return []
    return [
        statistics.median(values[i - window : i])
        for i in range(window, len(values) + 1)
    ]


def compute_latency_decomposition(drift_episodes: list[Episode]) -> LatencyDecomposition:
    """Mean latency per additive component across all drift episodes (Eq. 5)."""
    if not drift_episodes:
        raise ValueError("Cannot decompose latency: drift_episodes is empty")
    return LatencyDecomposition(
        t_detect_mean=round(statistics.mean(e.t_detect for e in drift_episodes), 3),
        t_decide_mean=round(statistics.mean(e.t_decide for e in drift_episodes), 3),
        t_execute_mean=round(statistics.mean(e.t_execute for e in drift_episodes), 3),
    )


def compute_per_mode_stats(drift_episodes: list[Episode]) -> dict[str, ModeStats]:
    """Per-reflex-mode summary statistics for modes that appear in the data."""
    result: dict[str, ModeStats] = {}
    for mode in REFLEX_PARAMS:
        dts = [e.delta_t for e in drift_episodes if e.reflex_mode == mode]
        if not dts:
            continue
        sd = sorted(dts)
        result[mode] = ModeStats(
            count=len(dts),
            median=round(statistics.median(dts), 2),
            std=round(compute_std(dts), 2),
            p90=round(sd[int(0.9 * len(sd))], 2),
        )
    return result


# ── Pipeline ──────────────────────────────────────────────────────────────────


class Pipeline:
    """Executes one reasoning-drift-recovery cycle (Algorithm 1, lines 1–8)."""

    def __init__(
        self,
        rng: random.Random,
        force_drift: bool = True,
        mtbf_mean: float = DEFAULT_MTBF_MEAN,
    ) -> None:
        self._rng = rng
        self._force_drift = force_drift
        self._mtbf_mean = mtbf_mean

    def run(self, run_id: int, wall_clock: float) -> tuple[Episode, float]:
        """
        Execute the three-node pipeline.
        Returns (episode, stable_interval) where stable_interval is the
        cognitive-up time drawn before the next fault.
        """
        query: str = self._rng.choice(QUERY_POOL)
        confidence: float = reasoning_node(query, self._rng)
        drift: bool = check_drift_node(confidence, self._rng, force=self._force_drift)

        t_fault: float = wall_clock

        if drift:
            result = recovery_node(self._rng)
            delta_t = result.t_detect + result.t_decide + result.t_execute
            mode: str | None = result.mode
            t_det, t_dec, t_exe = result.t_detect, result.t_decide, result.t_execute
        else:
            mode, t_det, t_dec, t_exe, delta_t = None, 0.0, 0.0, 0.0, 0.0

        t_recovered: float = t_fault + delta_t
        stable: float = self._rng.expovariate(1.0 / self._mtbf_mean)

        episode = Episode(
            run_id=run_id,
            query=query,
            confidence=round(confidence, 4),
            drift_detected=drift,
            reflex_mode=mode,
            t_detect=round(t_det, 4),
            t_decide=round(t_dec, 4),
            t_execute=round(t_exe, 4),
            delta_t=round(delta_t, 4),
            t_fault=round(t_fault, 4),
            t_recovered=round(t_recovered, 4),
        )
        return episode, stable


# ── Orchestrator ──────────────────────────────────────────────────────────────


class Orchestrator:
    """Runs Pipeline N times and collects episodes and stable intervals."""

    def __init__(
        self,
        n_runs: int = 200,
        seed: int = 42,
        force_drift: bool = True,
        verbose: bool = True,
        mtbf_mean: float = DEFAULT_MTBF_MEAN,
    ) -> None:
        self._n_runs = n_runs
        self._seed = seed
        self._force_drift = force_drift
        self._verbose = verbose
        self._mtbf_mean = mtbf_mean
        self._episodes: list[Episode] = []
        self._stable_intervals: list[float] = []

    def run(self) -> list[Episode]:
        """Execute all pipeline runs and return collected episodes."""
        rng = random.Random(self._seed)
        pipeline = Pipeline(rng, force_drift=self._force_drift, mtbf_mean=self._mtbf_mean)
        wall_clock = 0.0

        if self._verbose:
            print(f"\n{'─'*64}")
            print("  Run    Query excerpt                 Conf   Drift  Reflex")
            print(f"{'─'*64}")

        for run_id in range(self._n_runs):
            episode, stable = pipeline.run(run_id, wall_clock)
            wall_clock = episode.t_recovered + stable
            self._episodes.append(episode)
            self._stable_intervals.append(stable)

            if self._verbose and (run_id < 20 or run_id % 50 == 0):
                drift_marker = "YES" if episode.drift_detected else "  -"
                reflex = episode.reflex_mode or "—"
                print(
                    f"  {run_id:>4}   {episode.query[:32]:<32} "
                    f"{episode.confidence:.3f}  {drift_marker}    {reflex}"
                )

        if self._verbose:
            print(f"{'─'*64}")
            print(f"  ... ({self._n_runs} total runs completed)")

        return list(self._episodes)

    @property
    def stable_intervals(self) -> list[float]:
        """Cognitive-up intervals sampled between consecutive faults (Eq. 6)."""
        return list(self._stable_intervals)


# ── MetricsComputer ───────────────────────────────────────────────────────────


class MetricsComputer:
    """Derives SystemMetrics from completed episodes (Algorithm 1, lines 9–12)."""

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        self._alpha = alpha

    def compute(
        self,
        episodes: list[Episode],
        stable_intervals: list[float],
        n_runs: int,
    ) -> SystemMetrics:
        """
        Compute all system-level metrics.
        Raises ValueError when no drift events are present.
        """
        drift_eps = [e for e in episodes if e.drift_detected]
        all_dts = [e.delta_t for e in drift_eps]

        if not all_dts:
            raise ValueError("No drift events recorded; cannot compute metrics")

        mttr_a = compute_mttr_a(all_dts)
        med_ttr_a = compute_med_ttr_a(all_dts)
        std = compute_std(all_dts)
        p90 = compute_percentile(all_dts, 0.9)
        mtbf = compute_mtbf(stable_intervals)
        nrr = compute_nrr(med_ttr_a, mtbf)
        pi_up = compute_pi_up(mtbf, med_ttr_a)
        nrr_alpha = compute_nrr_alpha(med_ttr_a, std, mtbf, self._alpha)
        k_alpha = compute_k_alpha(self._alpha)
        decomp = compute_latency_decomposition(drift_eps)
        per_mode = compute_per_mode_stats(drift_eps)
        rolling = compute_rolling_median(all_dts, ROLLING_WINDOW)

        return SystemMetrics(
            n_runs=n_runs,
            drift_events=len(all_dts),
            drift_rate=round(len(all_dts) / n_runs, 3) if n_runs > 0 else 0.0,
            mttr_a_sys=round(mttr_a, 3),
            med_ttr_a_sys=round(med_ttr_a, 3),
            std_sys=round(std, 3),
            p90_sys=round(p90, 3),
            mtbf_sys=round(mtbf, 3),
            nrr_sys=round(nrr, 4),
            pi_up_sys=round(pi_up, 4),
            nrr_alpha=round(nrr_alpha, 4),
            alpha=self._alpha,
            k_alpha=round(k_alpha, 4),
            latency_decomposition=decomp,
            per_mode=per_mode,
            rolling_med_ttr_a=rolling,
        )


# ── TelemetryLogger ───────────────────────────────────────────────────────────


class TelemetryLogger:
    """Serialises episodes to JSONL (one JSON object per line)."""

    @staticmethod
    def to_records(episodes: list[Episode]) -> list[dict[str, object]]:
        """Convert episodes to plain dicts suitable for JSON serialisation."""
        return [
            {
                "run_id": ep.run_id,
                "query": ep.query,
                "confidence": ep.confidence,
                "drift_detected": ep.drift_detected,
                "reflex_mode": ep.reflex_mode,
                "t_detect": ep.t_detect,
                "t_decide": ep.t_decide,
                "t_execute": ep.t_execute,
                "delta_t": ep.delta_t,
            }
            for ep in episodes
        ]

    def save(self, episodes: list[Episode], path: str) -> None:
        """Write episodes to a JSONL file and print a confirmation line."""
        records = self.to_records(episodes)
        with open(path, "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        print(f"  Telemetry  → {path}  ({len(records)} records)")


# ── ResultsSaver ──────────────────────────────────────────────────────────────


class ResultsSaver:
    """Serialises SystemMetrics to a pretty-printed JSON file."""

    @staticmethod
    def to_dict(metrics: SystemMetrics) -> dict[str, object]:
        """Convert SystemMetrics to a plain dict for JSON serialisation."""
        return {
            "n_runs": metrics.n_runs,
            "drift_events": metrics.drift_events,
            "drift_rate": metrics.drift_rate,
            "mttr_a_sys": metrics.mttr_a_sys,
            "med_ttr_a_sys": metrics.med_ttr_a_sys,
            "std_sys": metrics.std_sys,
            "p90_sys": metrics.p90_sys,
            "mtbf_sys": metrics.mtbf_sys,
            "nrr_sys": metrics.nrr_sys,
            "pi_up_sys": metrics.pi_up_sys,
            "nrr_alpha": metrics.nrr_alpha,
            "alpha": metrics.alpha,
            "k_alpha": metrics.k_alpha,
            "latency_decomposition": {
                "t_detect_mean": metrics.latency_decomposition.t_detect_mean,
                "t_decide_mean": metrics.latency_decomposition.t_decide_mean,
                "t_execute_mean": metrics.latency_decomposition.t_execute_mean,
            },
            "per_mode": {
                mode: {"count": s.count, "median": s.median, "std": s.std, "p90": s.p90}
                for mode, s in metrics.per_mode.items()
            },
            "rolling_med_ttr_a": metrics.rolling_med_ttr_a,
        }

    def save(self, metrics: SystemMetrics, path: str) -> None:
        """Write metrics to a JSON file and print a confirmation line."""
        with open(path, "w") as fh:
            json.dump(self.to_dict(metrics), fh, indent=2)
        print(f"  Results    → {path}")


# ── Reporter ──────────────────────────────────────────────────────────────────


class Reporter:
    """Formats and prints the benchmark results to stdout."""

    _WIDTH: int = 64

    def print_report(self, metrics: SystemMetrics) -> None:
        """Print a full human-readable report for the given SystemMetrics."""
        W = self._WIDTH
        hr = "─" * W

        print(f"\n{'═'*W}")
        print("  MTTR-A BENCHMARK  ·  Cognitive Recovery Latency in MAS")
        print(f"{'═'*W}")
        print(
            f"  Runs: {metrics.n_runs}  |  Drift events: {metrics.drift_events}  "
            f"|  Drift rate: {metrics.drift_rate:.1%}"
        )

        print("\n  System-Level Reliability Metrics")
        print(f"  {hr}")
        print(
            f"  MedTTR-A (robust):  {metrics.med_ttr_a_sys:>7.3f} s"
            f"  ± {metrics.std_sys:.3f} s  (P90: {metrics.p90_sys:.3f} s)"
        )
        print(f"  MTTR-A (mean):      {metrics.mttr_a_sys:>7.3f} s")
        print(f"  MTBF:               {metrics.mtbf_sys:>7.3f} s")
        print(f"  NRR:                {metrics.nrr_sys:>7.4f}   (Eq. 8)")
        if metrics.pi_up_sys >= metrics.nrr_sys:
            print(f"  π_up (Thm 1):       {metrics.pi_up_sys:>7.4f}   ≥ NRR  ✓")
        else:
            print(f"  π_up (Thm 1):       {metrics.pi_up_sys:>7.4f}")
        print(
            f"  NRR_α (α={metrics.alpha}, Thm 2): {metrics.nrr_alpha:>7.4f}"
            f"  (k_α={metrics.k_alpha:.4f})"
        )

        print("\n  Latency Decomposition  Δt = t_detect + t_decide + t_execute  (Eq. 5)")
        print(f"  {hr}")
        d = metrics.latency_decomposition
        total = d.t_detect_mean + d.t_decide_mean + d.t_execute_mean
        if total > 0:
            print(f"  t_detect:   {d.t_detect_mean:.3f} s  ({d.t_detect_mean/total:.0%})")
            pct_dec = d.t_decide_mean / total
            pct_exe = d.t_execute_mean / total
            print(f"  t_decide:   {d.t_decide_mean:.3f} s  ({pct_dec:.0%})  ← negligible")
            print(f"  t_execute:  {d.t_execute_mean:.3f} s  ({pct_exe:.0%})  ← dominant")

        print("\n  Per-Reflex Mode Results  (Table II)")
        print(f"  {hr}")
        print(f"  {'Reflex Mode':<16}  {'Count':>5}  {'Median':>8}  {'Std':>6}  {'P90':>8}")
        print(f"  {'─'*16}  {'─'*5}  {'─'*8}  {'─'*6}  {'─'*8}")
        for mode, s in metrics.per_mode.items():
            print(
                f"  {mode:<16}  {s.count:>5}  {s.median:>7.2f}s"
                f"  {s.std:>5.2f}s  {s.p90:>7.2f}s"
            )

        rolls = metrics.rolling_med_ttr_a
        if rolls:
            roll_std = compute_std(rolls)
            print(f"\n  Rolling MedTTR-A (window={ROLLING_WINDOW} runs)")
            print(f"  {hr}")
            print(
                f"  Min: {min(rolls):.3f}s  |  Max: {max(rolls):.3f}s  |  "
                f"Std: {roll_std:.3f}s  ← stable"
            )

        print("\n  Paper vs. Simulation Comparison")
        print(f"  {hr}")
        print(f"  {'Metric':<20} {'Paper':>10} {'Simulated':>12}")
        print(f"  {'─'*20}  {'─'*10}  {'─'*12}")
        comparisons: list[tuple[str, float, float]] = [
            ("MedTTR-A (s)", _PAPER_METRICS["med_ttr_a_s"],  metrics.med_ttr_a_sys),
            ("MTBF (s)",     _PAPER_METRICS["mtbf_s"],       metrics.mtbf_sys),
            ("NRR",          _PAPER_METRICS["nrr"],          metrics.nrr_sys),
        ]
        for label, paper_val, sim_val in comparisons:
            print(f"  {label:<20} {paper_val:>10.3f} {sim_val:>12.3f}")

        print(f"{'═'*W}\n")


# ── Entry point ───────────────────────────────────────────────────────────────


def main(
    n_runs: int = 200,
    seed: int = 42,
    telemetry_path: str = "telemetry.jsonl",
    results_path: str = "results.json",
) -> None:
    """Run the full MTTR-A benchmark and persist results."""
    print("MTTR-A Multi-Agent System Benchmark")
    print(f"τ_drift = {TAU_DRIFT}  |  Reflex modes: {list(REFLEX_PARAMS)}")
    print(f"Simulating {n_runs} runs  (seed={seed}, force_drift=True)")
    print("force_drift=True: all runs are recovery episodes, matching Table II\n")

    orchestrator = Orchestrator(n_runs=n_runs, seed=seed, force_drift=True, verbose=True)
    episodes = orchestrator.run()

    computer = MetricsComputer(alpha=DEFAULT_ALPHA)
    metrics = computer.compute(episodes, orchestrator.stable_intervals, n_runs)

    Reporter().print_report(metrics)

    TelemetryLogger().save(episodes, telemetry_path)
    ResultsSaver().save(metrics, results_path)


if __name__ == "__main__":
    main()
