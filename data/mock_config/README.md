# Mock Configuration

## What this is

Tuning parameters for the `MockProvider` and the pure-Python simulation nodes (`reasoning_node`, `recovery_node`). These values control what the simulated confidence scores and latency measurements look like when no real LLM is involved.

None of these values come from the paper. They are calibration choices made to produce realistic-looking benchmark dynamics so that the metric computations are exercised meaningfully.

## Sections

### `confidence_distribution`

Controls how the mock `reasoning_node` generates a confidence score for each episode. The score is drawn from a Gaussian (bell-curve) distribution centred at `base_mu` with spread `base_sigma`, then a small noise term is added, and the result is clamped to [0, 1].

With the default values (mean 0.65, standard deviation 0.15), roughly 30–40% of episodes will produce a confidence score below the drift threshold of 0.6, generating enough drift events to make the metric computation meaningful.

| Parameter | Default | Effect of increasing |
|---|---|---|
| `base_mu` | 0.65 | Fewer drift events (higher average confidence) |
| `base_sigma` | 0.15 | More variable confidence; fatter tails |
| `noise_mu` | 0.0 | Shifts mean confidence up or down |
| `noise_sigma` | 0.05 | More episode-to-episode jitter |

### `recovery_timing`

Controls the `t_detect` and `t_decide` phases in `recovery_node`. Both are drawn from exponential distributions (a statistical model commonly used for the time between events — it produces mostly short values with occasional longer ones).

These are negligible relative to `t_execute` in practice. The paper confirms that execution dominates (~90% of Δt), so these values matter only for the decomposition breakdown.

| Parameter | Default | Meaning |
|---|---|---|
| `t_detect_mean_s` | 0.50 s | Mean time to route the confidence signal through drift detection |
| `t_decide_mean_s` | 0.15 s | Mean time to select the recovery reflex by policy sampling |
| `t_execute_min_s` | 0.10 s | Lower bound on execution time — prevents negative values from Gaussian sampling |

### `mock_provider_latency`

Sleep ranges used by `MockProvider` when `mock_simulate_latency=True`. For each call, a duration is drawn uniformly from `[min, max]` seconds and `time.sleep()` is called to make the benchmark take realistic wall-clock time.

The key `""` (empty string) is the fallback for any unrecognised context hint. Real providers do not use this section.

### `ttft_fraction`

`MockProvider` simulates time-to-first-token by recording the timestamp at `ttft_fraction * elapsed_time` into the call. The default of 0.20 means the first token is "seen" when 20% of the total call duration has elapsed.

## How it is used

`mttr_a_simulation.py` reads this file at import time for `reasoning_node` and `recovery_node` parameters. `mttr_a/providers.py` reads it for `MockProvider` latency ranges, confidence distribution, and TTFT fraction.

## How to modify

Changing these values does not affect paper replication — they control the simulation's behaviour, not the metric definitions. Useful modifications:
- Raise `base_mu` to simulate a well-calibrated, rarely-drifting model
- Lower `base_mu` to simulate a poorly-calibrated model that drifts frequently
- Widen `mock_provider_latency` ranges to simulate slower providers
