# MTTR-A: Cognitive Recovery Latency in Multi-Agent Systems

A Python implementation of the MTTR-A reliability framework from:

> Barak Or, *"MTTR-A: Measuring Cognitive Recovery Latency in Multi-Agent Systems"*, arXiv:2511.20663v5, 2025.

---

## The Problem

When a server crashes, reliability engineers have a clear signal: the service is down. They measure **Mean Time To Recover (MTTR)** — how long until it's back up — and use that to set SLOs (Service Level Objectives — targets for how reliable a system must be, e.g. 'respond within 2 seconds 99% of the time') and improve resilience.

AI agents break differently. An LLM-based agent  doesn't go offline — it keeps running while its reasoning silently deteriorates. It might start looping, lose track of prior context, contradict its own earlier outputs, or drift away from the task without any hard error being thrown. From the infrastructure's perspective, everything looks fine. From the user's perspective, the system is producing nonsense.

This is the gap: **classical reliability metrics assume binary up/down state, but cognitive failures are continuous, semantic, and partially recoverable.** There is no standard way to measure how quickly a distributed reasoning system detects it has gone wrong and pulls itself back to coherent operation.

That question — *how long does it take a multi-agent system to recover its reasoning after it drifts?* — has no industry-standard answer. Until now there has been no metric for it.

---

## The Solution: MTTR-A

MTTR-A adapts classical dependability theory to the cognitive domain. It defines a **fault** as a reasoning drift event (detected when retrieval confidence drops below a threshold) and a **recovery** as the restoration of coherent operation via a reflexive control action.

### Core Metrics

**MTTR-A** — Mean Time-to-Recovery for Agentic Systems

$$\text{MTTR-A}_{\text{sys}} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{M_i} \sum_{j=1}^{M_i} \left( t_{r,i}^{(j)} - t_{f,i}^{(j)} \right)$$

Where $t_f$ is when a fault is detected and $t_r$ is when coherence is restored. When recovery times are skewed (pulled toward higher values by occasional very slow recoveries, like a human-approval escalation), the arithmetic mean gives a misleading picture. The paper instead uses the median — the middle value when all measurements are sorted — as a more reliable estimator. This is called **Median Time To Recover — Agentic (MedTTR-A)**.

**MTBF** — Mean Time Between Cognitive Faults, the average stable operating interval between drift events.

**NRR** — Normalized Recovery Ratio, a dimensionless reliability index:

$$\text{NRR}_{\text{sys}} = 1 - \frac{\text{MTTR-A}_{\text{sys}}}{\text{MTBF}_{\text{sys}}}$$

Values near 1 mean the system recovers fast relative to how often it drifts. Values near 0 (or negative) mean the system spends most of its time recovering.

**NRR_α** — A statistically conservative variant that accounts for variability in recovery times. It uses Cantelli's inequality — a mathematical rule that provides a safe upper bound on how bad the metric could be, even when the underlying distribution is unknown — at a chosen confidence level α (a number between 0 and 1, typically 0.90, meaning the bound holds 90% of the time).

### Theorem 1 — NRR bounds steady-state uptime

The paper proves that in the long run (once the system has been running long enough for its average behaviour to stabilise), the fraction of time it spends in a cognitively healthy state — called π_up (π is the Greek letter 'pi', used here simply as a name for this proportion) — satisfies:

$$\pi_{\text{up}} = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR-A}} \geq 1 - \frac{\text{MTTR-A}}{\text{MTBF}} = \text{NRR}$$

NRR is always a **conservative lower bound** on true cognitive uptime — meaning the real uptime fraction is at least as good as NRR, and often better.

---

## How the Metric is Measured

### Simulation / benchmark pipeline

The simulation and production benchmark measure MTTR-A over N independent episodes. Each episode is a single LLM reasoning call followed by an in-process confidence check — no second LLM call, no self-evaluation prompt:

```
t_queued        ← runner receives the request
    │
    ├── t_reason_start ──[LLM reasoning call]── t_reason_end
    │                                │
    │               cos(output_emb, grounding_doc_emb) ── t_drift_check
    │                                │
    │                          if drift detected:
    │                   t_recovery_start ──[reflex LLM call]── t_recovery_end
    │
    └── (next episode starts here)
```

Confidence is `cos(output_emb, grounding_doc_emb)` computed locally via `all-MiniLM-L6-v2` — no second API call. Recovery is executed by firing a reflex LLM call. The three paper-defined phases:

| Phase | What it captures | Typical share |
|---|---|---|
| `t_detect` | In-process threshold comparison — microseconds | ~8% |
| `t_decide` | Policy selection (weighted draw) — microseconds | ~3% |
| `t_execute` | Reflex LLM call round-trip — seconds | **~90%** |

Execution dominates because detect and decide are in-process operations with no network calls.

### Real agent SDK

The SDK takes a different approach suited to continuous multi-turn tasks. No self-evaluation call, no reflex execution — purely observational:

```
Agent loop (Claude or Gemini):
  [tool call: search_corpus] → doc retrieved
  session.set_evidence(doc)  ← embed the retrieved doc as evidence reference
  session.record_step(reasoning_text) → confidence = cos(output_emb, evidence_emb)
  ... more turns ...
  confidence drops below τ   → t_fault recorded
  ... agent continues, no intervention ...
  confidence returns above τ → t_recovered recorded
  δt = t_recovered − t_fault (wall-clock, real inference calls)
```

The agent's own retrieved document is the evidence reference — groundedness measures how well the agent's reasoning stays anchored to what it found. Before any retrieval, `confidence = None`. Recovery is observed, not forced: the agent already has its evidence in context. `T_detect` and `T_decide` are effectively zero; `δt = T_execute` entirely.

### What is and is not captured (benchmark)

**1. Stable intervals between fault events**
The benchmark samples inter-fault gaps from `Exp(1/MTBF_mean)` rather than measuring them. In real traffic, use `infer_stable_intervals(episodes)` to derive these from actual episode timestamps and pass the result to `MetricsComputer.compute()`.

**2. Time-to-first-token vs. full completion**
Every `LLMResponse` carries an optional `t_first_token` field. When using a streaming provider, set this to the timestamp of the first token received; the telemetry sink writes it as `t_first_token_s` in each JSONL record.

**3. Upstream queue and routing time**
The runner records `t_queued` immediately before `graph.invoke()`. The difference `t_reason_start − t_queued` is the pre-LLM routing overhead and is written as `queue_latency_s` in telemetry. In a back-to-back benchmark this is near-zero; it becomes meaningful under real concurrent traffic.

**4. Tool calls within a reasoning step**
`LLMResponse` carries `tool_calls: tuple[ToolCall, ...]`. The reasoning node aggregates total tool latency into `tool_latency_s` and call count into `n_tool_calls`, both written to telemetry.

### Input and output

`MetricsComputer.compute()` takes:

```python
episodes: list[Episode]         # one per completed pipeline run
stable_intervals: list[float]   # seconds between consecutive fault events
n_runs: int
```

The fields that feed the math:
- `delta_t` — total recovery latency per drift episode → used for MedTTR-A
- `drift_detected` — gates which episodes count as recovery episodes
- `t_fault` / `t_recovered` — wall-clock anchors for stable-interval inference

Output (`SystemMetrics`):
```
MedTTR-A = median(delta_t)              ← primary estimator (paper Algorithm 1)
MTBF     = mean(stable_intervals)
NRR      = 1 − MedTTR-A / MTBF         ← headline reliability number
π_up     = MTBF / (MTBF + MedTTR-A)   ← steady-state cognitive uptime
NRR_α    = 1 − (MedTTR-A + k·σ) / MTBF ← risk-adjusted, Cantelli bound
```

### Cross-model uniformity

The `Episode` schema is entirely provider-agnostic. Timing comes from `time.perf_counter()` at the system level, so Bedrock, Azure OpenAI, and Vertex AI all produce structurally identical records.

The one model-varying input is **confidence**, normalized to `[0, 1]` via retrieval-based scoring — exactly the signal the paper defines. The paper uses `cos(query, retrieved_doc)` from the AG News corpus. This implementation ships a bundled reference corpus in `data/corpus/documents.txt` covering distributed systems, databases, algorithms, ML/AI, software engineering, networking, security, SRE, agent systems, and general reasoning.

For each episode, real providers (`providers.py`):
1. Embed the query with `all-MiniLM-L6-v2` (sentence-transformers, runs locally — no API call).
2. Retrieve the top-matching document from the corpus by dot-product over pre-computed, L2-normalised embeddings.
3. Return `cos(query_emb, top_doc_emb)` as the confidence score — high when the query maps to a well-represented concept, lower when it drifts to unfamiliar territory.

`MockProvider` simulates this distribution with a Gaussian (`N(0.65, 0.15) + N(0.0, 0.05)`), so the full pipeline runs without a corpus, embedding model, or API keys. The Gaussian parameters are tunable in `data/mock_config/config.json`.

To measure MTTR-A against a domain-specific system, replace `data/corpus/documents.txt` with documents representative of your production query space (one document per line, `#` for comments). No code changes needed.

For valid cross-model comparison, hold `drift_threshold`, `seed`, and `QUERY_POOL` constant across runs.

---

### Reflex Taxonomy (simulation / benchmark)

The simulation selects a recovery action from four families defined in the paper. These are the reflexes with empirically measured latencies from Table II:

| Reflex | What it does | Typical latency |
|---|---|---|
| `auto-replan` | Regenerate the reasoning plan | ~5.9 s |
| `tool-retry` | Retry the failed tool call | ~4.5 s |
| `rollback` | Restore a prior checkpoint | ~7.0 s |
| `human-approve` | Escalate to a human | ~12.2 s |

The SDK does not fire any reflex — it observes whether the agent's reasoning naturally returns above τ across subsequent turns (`reflex_mode = "natural"` when recovery is confirmed).

---

## What's In This Repository

The paper defines the metric and validates it with a mock simulation. This repository ships that simulation and extends it with a production-ready implementation designed to run against real LLMs in real infrastructure.

### Paper Replication — the simulation layer

`mttr_a_simulation.py` is a **pure-Python mock** of the paper's LangGraph benchmark — no LLM calls, no external dependencies beyond the standard library. It simulates the three-node reasoning-drift-recovery pipeline and reproduces the paper's reported results within measurement noise.

The code is structured around **single responsibility** — each class does exactly one thing:

```
reasoning_node       → mock confidence score (cosine similarity proxy)
check_drift_node     → flag fault when confidence < τ_drift (0.6)
recovery_node        → select reflex by weighted sampling, simulate latency
        ↓
Pipeline             → one complete reasoning-drift-recovery cycle
Orchestrator         → repeat Pipeline N times, collect episodes
MetricsComputer      → derive MTTR-A, MTBF, NRR from episodes
        ↓
TelemetryLogger      → write episodes → telemetry.jsonl
ResultsSaver         → write metrics  → results.json
Reporter             → print human-readable report
```

**Simulation results (200 runs, seed=42):**

| Metric | Paper | This simulation |
|---|---|---|
| MedTTR-A | 6.21 s ± 2.14 s | **6.08 s ± 2.18 s** |
| MTBF | 6.73 s | **6.45 s** |
| NRR | 0.077 | **0.058** |

Theorem 1 holds: π_up (0.515) ≥ NRR (0.058) ✓

---

## Beyond the Paper

The paper defines the metric. This implementation goes further — it provides everything needed to measure MTTR-A against a real LLM system running in production.

### Real LLM provider integrations

Three hyperscaler providers replace the mock with actual LLM calls. Each handles authentication and health-checking in a uniform way, so the metric computation is identical regardless of which model is underneath. Confidence is always `cos(output_emb, grounding_doc_emb)` — computed locally, never via a second LLM call. A `MockProvider` lets you run the full benchmark without any cloud credentials.

| Provider | Install | What you need |
|---|---|---|
| AWS Bedrock | `pip install -e ".[bedrock]"` | IAM role or AWS credentials |
| Azure OpenAI | `pip install -e ".[azure]"` | Azure endpoint + API key |
| GCP Vertex AI | `pip install -e ".[vertex]"` | GCP project + service account |

### Cloud telemetry sinks

Benchmark results stream to any combination of cloud destinations in real time. A `CompositeSink` fans out to all enabled sinks simultaneously, with failure isolation so one failing destination doesn't interrupt the others.

| Sink | What it writes to |
|---|---|
| `JsonlSink` | Local JSONL file (always on — one record per episode) |
| `CloudWatchSink` | AWS CloudWatch Metrics (latency, drift rate, NRR as custom dimensions) |
| `AzureMonitorSink` | Azure Monitor / Application Insights (structured log events) |
| `GCPLoggingSink` | GCP Cloud Logging (structured JSON entries) |

### Extended latency instrumentation

The paper's Episode model captures the three phases of recovery. This implementation adds four additional fields that surface latency sources the paper's equations don't see:

| Field | What it captures | When it matters |
|---|---|---|
| `queue_latency_s` | Pre-LLM routing time — load balancer, API gateway, rate-limit wait | Under concurrent production traffic |
| `ttft_s` | Time-to-first-token (how long before any part of the response appears) | Streaming providers; perceived latency |
| `tool_latency_s` | Total time spent inside external tool calls within a reasoning step | Agents that call search, SQL, or code execution |
| `n_tool_calls` | Number of tool invocations per episode | Proxy for reasoning complexity |

All four fields flow through every telemetry sink automatically.

### Portable configuration

Every setting — provider, model ID, thresholds, telemetry paths, sink flags — loads from a `.env` file without any `export` commands. Platform-injected secrets (Kubernetes environment variables, ECS task definitions, Cloud Run) always take precedence. Copy `.env.example` to `.env` and fill in what you need.

### Real agent SDK — `mttr_a.sdk`

The paper validates MTTR-A with a simulated pipeline using `time.sleep()` as a proxy for recovery latency. The SDK applies the same metric to real agents — no simulation, no proxy.

**How it works:**

```
User submits task
      ↓
Agent loop (Claude or Gemini):
  Turn 1: Thought → [tool call: search_corpus] → retrieved doc
                    session.set_evidence(doc)   ← embed the retrieved doc
          session.record_step(reasoning_text)   → confidence = cos(output, evidence_emb)
  Turn 2: Thought → [tool call: calculate]     → result
          session.record_step(reasoning_text)   → confidence = 0.42  ← FAULT (< τ)
          t_fault recorded. Observer only — no intervention.
  Turn 3: Thought (agent reasons with evidence already in context)
          session.record_step(reasoning_text)   → confidence = 0.81  ← RECOVERED
          t_recovered recorded. δt = t_recovered − t_fault.
      ↓
session.episode() → Episode (feeds MetricsComputer)
```

When the agent calls `search_corpus`, the retrieved document is embedded and stored as the evidence reference via `session.set_evidence()`. Every subsequent `record_step()` call scores the agent's reasoning against that embedding: `cos(output_emb, evidence_emb)`. Before any retrieval, `confidence = None` — there is nothing to score against.

Drift is detected when confidence drops below `τ = 0.6`. The agent already has the retrieved document in its context window — no external intervention is applied. Recovery is confirmed when confidence returns above τ on its own. `δt` is the wall-clock time between those two events, spanning real inference calls.

**SDK usage:**

```python
from mttr_a.sdk import ClaudeAdapter
from mttr_a.sdk.tools import DEMO_TOOLS

session = ClaudeAdapter().run(
    task="Calculate memory requirements for a 7B parameter transformer in float16, "
         "then search for memory optimization techniques.",
    tools=DEMO_TOOLS,          # search_corpus, calculate, conclude
    api_key="sk-ant-...",
    model="claude-sonnet-4-6",
    max_turns=12,
)
episode = session.episode()    # Episode feeds MetricsComputer
print(f"drift={episode.drift_detected}, delta_t={episode.delta_t:.2f}s")
# None values appear for turns before the first search_corpus call
print(f"per-step groundedness: {episode.step_confidences}")
```

The same `Episode` object feeds directly into `MetricsComputer` — SDK episodes and benchmark episodes are interchangeable in aggregate metrics.

| Adapter | Provider | Install |
|---|---|---|
| `ClaudeAdapter` | Anthropic Claude (tool use) | `pip install anthropic` |
| `GeminiAdapter` | Google Gemini (function calling) | `pip install google-genai` |

### Web dashboard

`web/` is a FastAPI + SSE dashboard that drives a real Claude or Gemini agent through a multi-step task using the SDK. It streams per-turn groundedness scores, fault events, and aggregate metrics live as the agent runs.

```bash
pip install -e ".[web]"
uvicorn web.app:app --reload --port 8000
# open http://127.0.0.1:8000
```

### Diagnostic notebook

`mttr_a_diagnostic.ipynb` is a guided, interactive walkthrough of the full measurement pipeline — from raw episode data to a multi-panel operational dashboard. See the [Diagnostic Notebook](#diagnostic-notebook) section below.

---

## Quickstart

**Run the simulation (no dependencies beyond the standard library):**

```bash
python mttr_a_simulation.py
```

Output: console report, `telemetry.jsonl` (one JSON record per episode), `results.json` (all computed metrics).

**Run the production benchmark with MockProvider:**

```bash
pip install -e ".[notebook]"   # or just -e . for the core package
python -m mttr_a               # reads all settings from .env
```

**Open the diagnostic notebook:**

```bash
pip install -e ".[notebook]"
jupyter notebook mttr_a_diagnostic.ipynb
```

**Launch the web dashboard:**

```bash
pip install -e ".[web]"
uvicorn web.app:app --reload --port 8000
# open http://127.0.0.1:8000
```

The dashboard runs a real Claude or Gemini agent on a user-supplied task and streams per-turn groundedness scores, fault events, and recovery confirmation live.

**Run a real agent task via the SDK:**

```python
from mttr_a.sdk import ClaudeAdapter
from mttr_a.sdk.tools import DEMO_TOOLS

session = ClaudeAdapter().run(
    task="Explain Raft consensus and calculate quorum size for a 5-node cluster.",
    tools=DEMO_TOOLS,
    api_key="sk-ant-...",
    model="claude-sonnet-4-6",
)
print(session.episode())
```

**Run the test suite:**

```bash
pip install pytest
pytest test_mttr_a.py test_production.py test_sdk.py -q
```

---

## Diagnostic Notebook

`mttr_a_diagnostic.ipynb` provides a visual, interactive environment for understanding and diagnosing MTTR-A results. It is written for a first-time reader — every section explains what is being measured and why, without assuming prior familiarity with reliability engineering or statistical terminology.

**What the notebook covers:**

| Section | What it shows |
|---|---|
| What cognitive drift looks like | Real prompts, confidence scores, the Episode data model with annotated field descriptions |
| Running the benchmark | Importing `mttr_a` as a Python library and collecting episodes in a single function call |
| Telemetry exploration | The raw JSONL schema and a sample of stable vs. drift episodes side by side |
| Cognitive drift timeline | Confidence per episode with drift markers, τ_drift threshold, amber stems showing recovery cost, and shaded bands for consecutive drift clusters |
| Latency decomposition | Why t_execute accounts for ~90% of recovery time and what that means for optimisation |
| Rolling MedTTR-A | Whether recovery speed is improving, degrading, or stable across the run |
| Diagnostic dashboard | A single exportable figure combining all metrics — KPI tiles, drift timeline, rolling trend, decomposition, and per-reflex breakdown |
| Per-reflex analysis | Which recovery paths are selected most often and which are disproportionately slow |
| Extended instrumentation | Queue latency, TTFT, and tool-call overhead under real traffic |
| Diagnostic summary | A plain-language narrative and a reference threshold table for use as a CI gate |

**To launch:**

```bash
pip install -e ".[notebook]"
jupyter notebook mttr_a_diagnostic.ipynb
```

`FAST_MODE = True` (default) runs 30 episodes in ~7 seconds — enough to see the full shape of the data. Set `FAST_MODE = False` for a full 200-episode run (~45 seconds) with more stable metric estimates. Swap `MTTR_PROVIDER` in `.env` to route through a real LLM provider — no code changes needed.

---

## Key Design Decisions

**`force_drift=True` (default)** — All 200 runs produce a recovery episode, matching the paper's Table II where reflex-mode counts sum to exactly 200. Set `force_drift=False` for a probabilistic simulation where only ~42% of runs trigger drift.

**MedTTR-A as the system estimator** — Algorithm 1 in the paper uses `median(ΔTi)` rather than the mean, because recovery distributions are right-skewed — most recoveries are fast, but occasional slow ones (especially human-approval escalations) pull the average upward, making the mean a misleading summary. The `mttr_a_sys` field (mean) is also computed for reference.

**Stable intervals from Exp(1/MTBF)** — Inter-fault stable time is drawn from an exponential distribution (a statistical model commonly used for the time between random events; it produces mostly short gaps with occasional long ones) with mean 6.73 s, matching the paper's reported MTBF and enabling correct NRR computation.

---

## File Reference

**Simulation layer (paper replication)**

| File | Purpose |
|---|---|
| `mttr_a_simulation.py` | Data models, pure metric functions, Pipeline, Orchestrator, MetricsComputer, I/O, Reporter |
| `test_mttr_a.py` | 149-test suite: unit, functional, and integration coverage — 100% coverage |

**Production package (`mttr_a/`)**

| File | Purpose |
|---|---|
| `mttr_a/config.py` | `BenchmarkConfig`, `ProviderConfig`, `ProviderKind`, `load_from_env()` — auto-loads `.env` |
| `mttr_a/providers.py` | `BaseLLMProvider` ABC, `MockProvider`, `BedrockProvider`, `AzureOpenAIProvider`, `VertexAIProvider`, `ToolCall`, `LLMResponse` |
| `mttr_a/graph.py` | LangGraph `StateGraph` — three-node pipeline with full instrumentation |
| `mttr_a/sinks.py` | `JsonlSink`, `CloudWatchSink`, `AzureMonitorSink`, `GCPLoggingSink`, `CompositeSink` |
| `mttr_a/runner.py` | `ProductionRunner` — drives N episodes, collects instrumentation, returns `SystemMetrics` |
| `mttr_a/__main__.py` | `python -m mttr_a` CLI entrypoint |
| `test_production.py` | 76-test suite for the production package |
| `test_live.py` | Live integration tests (skipped without real credentials) |
| `example_production.py` | Runnable demo — MockProvider works immediately, swap comments to go live |
| `.env.example` | All environment variables documented |
| `pyproject.toml` | Package definition with per-provider extras (`pip install -e ".[bedrock]"`) |

**Real agent SDK (`mttr_a/sdk/`)**

| File | Purpose |
|---|---|
| `mttr_a/sdk/session.py` | `MTTRSession` — per-task measurement object; `record_step()`, `set_evidence()`, `episode()` |
| `mttr_a/sdk/tools.py` | `CorpusTool`, `CalculatorTool`, `ConcludeTool` — demo tools with Claude and Gemini schemas |
| `mttr_a/sdk/adapters/claude.py` | `ClaudeAdapter` — drives Claude tool-use loop, measures drift per turn |
| `mttr_a/sdk/adapters/gemini.py` | `GeminiAdapter` — drives Gemini function-calling loop, same measurement contract |
| `mttr_a/sdk/__init__.py` | Public re-exports: `MTTRSession`, `StepRecord`, `ClaudeAdapter`, `GeminiAdapter` |
| `test_sdk.py` | 72-test suite for the SDK: session, tools, schemas, public exports |

**Web dashboard (`web/`)**

| File | Purpose |
|---|---|
| `web/app.py` | FastAPI backend — `/api/run` (benchmark), `/api/task` (SDK agent loop), `/api/stream/{id}` (SSE) |
| `web/static/index.html` | Single-page dashboard — agent task with live groundedness streaming |

---

## Reference

Barak Or. "MTTR-A: Measuring Cognitive Recovery Latency in Multi-Agent Systems." *arXiv preprint arXiv:2511.20663v5*, December 2025.
