# Paper Benchmarks

## What this is

The reference metric values reported in the original paper for a 200-run, seed-42 benchmark. These are the numbers this simulation is calibrated to reproduce.

## Where it comes from

**Table II** of:

> Barak Or, "MTTR-A: Measuring Cognitive Recovery Latency in Multi-Agent Systems," arXiv:2511.20663v5, 2025.

## Fields

| Field | Value | Meaning |
|---|---|---|
| `n_runs` | 200 | Number of episodes the paper's benchmark ran |
| `seed` | 42 | Random seed used to produce Table II results |
| `metrics.med_ttr_a_s` | 6.21 | Median Time To Recover — Agentic, in seconds |
| `metrics.med_ttr_a_std_s` | 2.14 | Standard deviation of per-episode recovery times |
| `metrics.mtbf_s` | 6.73 | Mean Time Between Cognitive Faults, in seconds |
| `metrics.nrr` | 0.077 | Normalized Recovery Ratio (close to 0 because median recovery time ≈ MTBF here) |

## How it is used

`Reporter.print_report()` in `mttr_a_simulation.py` loads this file and prints a comparison table showing the paper's values alongside the values produced by the current simulation run. This makes it easy to verify that the simulation is correctly calibrated.

## How to interpret the comparison

This simulation produces MedTTR-A ≈ 6.08 s (vs. paper's 6.21 s) and NRR ≈ 0.058 (vs. 0.077). The small differences are expected — the simulation uses a different random draw for each run, and the paper's exact internal sampling procedure is not fully specified. The values are within measurement noise.

## Do not modify

These values are fixed reference points from the paper. Changing them would make the comparison output misleading. If you want to track your own production benchmark results over time, create a separate file (e.g. `results.json`) using `ResultsSaver`.
