# Reflex Parameters

## What this is

The calibration data for the four recovery reflex modes. When the drift-detection node flags an episode, the recovery node samples one reflex by weighted random selection and draws an execution time from a Gaussian (bell-curve) distribution parameterised by the values here.

## Where it comes from

All values are taken directly from **Table II** of the paper:

> Barak Or, "MTTR-A: Measuring Cognitive Recovery Latency in Multi-Agent Systems," arXiv:2511.20663v5, 2025.

Do not modify these values if the goal is to replicate the paper's benchmark results.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `weight` | integer | Relative selection probability. A reflex with weight 93 is selected roughly 93/(93+42+44+21) = 46% of the time. |
| `median_latency_s` | float | The target median execution time in seconds. The recovery node draws `t_execute` from a Gaussian centred here. |
| `std_latency_s` | float | Standard deviation of the execution time distribution in seconds. Higher values mean more variable recovery times. |

## The four reflexes

| Reflex | Selection share | Median latency | What it represents |
|---|---|---|---|
| `auto-replan` | ~46% | 5.94 s | Re-runs the reasoning step with a modified prompt — the default, lowest-overhead recovery |
| `tool-retry` | ~21% | 4.46 s | Retries the failing tool call (search, retrieval) with exponential backoff |
| `rollback` | ~22% | 6.99 s | Reverts to a prior known-good checkpoint — slower but more thorough |
| `human-approve` | ~10% | 12.22 s | Escalates to a human reviewer — highest latency, used as a last resort |

## How it is used

`mttr_a_simulation.py` loads this file at import time into the `REFLEX_PARAMS` dict, which is referenced by `recovery_node` for weight-based sampling and latency simulation. The production package's `graph.py` uses the same dict for the LangGraph recovery node.

## How to modify

Changing `weight` values shifts the selection distribution — useful for testing scenarios where one reflex dominates. Changing `median_latency_s` or `std_latency_s` shifts the recovery time distribution — useful for modelling faster or slower reflex implementations.

Adding a new reflex requires a corresponding entry in `graph.py`'s `recovery_node` context-string mapping.
