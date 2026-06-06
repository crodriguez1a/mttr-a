"""
MTTR-A Production Example
=========================

Runs end-to-end with MockProvider (no API keys required) and prints a full
MTTR-A system-health report.  The comments marked "SWAP →" show the exact
one-line change to route traffic through each hyperscaler.

Usage
-----
  # Quick run (no API keys)
  python example_production.py

  # Real Bedrock (Claude 3.5 Sonnet)
  export MTTR_PROVIDER=bedrock
  export MTTR_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
  export MTTR_AWS_REGION=us-east-1
  python example_production.py

  # Real Azure OpenAI (GPT-4o)
  export MTTR_PROVIDER=azure_openai
  export MTTR_AZURE_ENDPOINT=https://my-resource.openai.azure.com
  export MTTR_AZURE_DEPLOYMENT=gpt-4o
  export AZURE_OPENAI_API_KEY=<key>
  python example_production.py

  # Real Vertex AI (Gemini 1.5 Pro)
  export MTTR_PROVIDER=vertex_ai
  export MTTR_GCP_PROJECT=my-gcp-project
  export MTTR_MODEL_ID=gemini-1.5-pro-002
  gcloud auth application-default login
  python example_production.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from mttr_a import (
    BenchmarkConfig,
    CompositeSink,
    JsonlSink,
    ProviderConfig,
    ProviderKind,
    build_provider,
    load_from_env,
    ProductionRunner,
)
from mttr_a_simulation import Reporter


# ── Configuration ─────────────────────────────────────────────────────────────

def build_config() -> BenchmarkConfig:
    """
    Returns a config from environment variables when MTTR_PROVIDER is set,
    otherwise falls back to a hard-coded MockProvider config so the example
    runs immediately without any credentials.
    """
    if os.getenv("MTTR_PROVIDER"):
        return load_from_env()

    return BenchmarkConfig(
        provider=ProviderConfig(
            kind=ProviderKind.MOCK,
            # SWAP → ProviderKind.BEDROCK, model_id="anthropic.claude-3-5-sonnet-20241022-v2:0"
            # SWAP → ProviderKind.AZURE_OPENAI, azure_endpoint="...", azure_deployment="gpt-4o"
            # SWAP → ProviderKind.VERTEX_AI, gcp_project="my-project", model_id="gemini-1.5-pro-002"
            mock_simulate_latency=False,   # flip to True for realistic timing
        ),
        n_runs=200,
        seed=42,
        drift_threshold=0.6,
        stochastic_fault_rate=0.05,
        mtbf_mean=6.73,
        alpha=0.90,
        telemetry_path="telemetry_example.jsonl",
        results_path="results_example.json",
        verbose=True,
    )


# ── Sink wiring ────────────────────────────────────────────────────────────────

def build_sink(config: BenchmarkConfig, extra_path: str | None = None) -> CompositeSink:
    """
    Fan-out sink: always writes local JSONL.
    Add cloud sinks here for production deployments:

      from mttr_a import CloudWatchSink, AzureMonitorSink, GCPLoggingSink

      # AWS
      cloud = CloudWatchSink(region="us-east-1", namespace="MTTR-A")

      # Azure
      cloud = AzureMonitorSink(connection_string=os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"])

      # GCP
      cloud = GCPLoggingSink(project="my-gcp-project")

      return CompositeSink(local, cloud)
    """
    local = JsonlSink(extra_path or config.telemetry_path)
    return CompositeSink(local)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    config = build_config()

    provider_label = config.provider.kind.value
    print(f"\nMTTR-A Production Example  —  provider: {provider_label}")
    print(f"n_runs={config.n_runs}  seed={config.seed}  τ_drift={config.drift_threshold}\n")

    provider = build_provider(config.provider, seed=config.seed)

    # Optional: confirm the provider is reachable before committing to a full run
    if provider_label != "mock":
        print("Health-checking provider...", end=" ", flush=True)
        ok = provider.health_check()
        print("OK" if ok else "FAILED — aborting")
        if not ok:
            return

    sink = build_sink(config)
    runner = ProductionRunner(config=config, provider=provider, sink=sink)
    metrics = runner.run()

    # Re-use the simulation's console reporter (works with any SystemMetrics)
    Reporter().print_report(metrics)

    telemetry_file = Path(config.telemetry_path)
    n_lines = sum(1 for _ in telemetry_file.open()) if telemetry_file.exists() else 0
    print(f"\nTelemetry written → {telemetry_file}  ({n_lines} episodes)")

    # ── Swap-in checklist ───────────────────────────────────────────────────
    print("\n── Swap to a real provider ─────────────────────────────────────────")
    print("  Bedrock:   export MTTR_PROVIDER=bedrock MTTR_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0")
    print("  Azure:     export MTTR_PROVIDER=azure_openai MTTR_AZURE_ENDPOINT=... MTTR_AZURE_DEPLOYMENT=gpt-4o")
    print("  Vertex AI: export MTTR_PROVIDER=vertex_ai MTTR_GCP_PROJECT=my-project MTTR_MODEL_ID=gemini-1.5-pro-002")
    print("  Then re-run:  python example_production.py")
    print("────────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
