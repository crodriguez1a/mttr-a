# Query Pool

## What this is

The set of input prompts that cycle through the benchmark. Each run selects one query at random from this list and passes it to the reasoning node, which produces a confidence score. That confidence score is what the drift-detection node evaluates.

## Where it comes from

These 16 queries were chosen to represent the vocabulary of multi-agent reliability topics — the kinds of questions an agent in this domain would actually receive. They are not from the paper; the paper does not specify a fixed query set. They are a representative stand-in for real workload queries.

In production, replace this file with queries sampled from your agent's actual traffic. The distribution of confidence scores — and therefore the drift rate — will shift to reflect your real workload.

## Format

One query per line. Blank lines and lines beginning with `#` are ignored. No minimum or maximum length is enforced, but very long queries may inflate time-to-first-token on real providers.

## How it is used

`mttr_a_simulation.py` loads this file at import time and exposes the contents as the module-level constant `QUERY_POOL`. The `reasoning_node` function samples from it each episode.

The notebook imports `QUERY_POOL` directly and displays the full list in Section 1.

## How to modify

Add, remove, or rewrite queries freely. The only constraint is that at least one query must be present. For reproducible comparisons across runs, keep the file stable — a different query set will produce a different confidence distribution and therefore different MTTR-A numbers.
