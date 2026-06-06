"""
Live integration tests — require real cloud credentials.

These tests are skipped automatically unless the matching environment variables
are set.  They are intentionally excluded from the CI test suite; run them
manually before a production deployment to verify provider wiring end-to-end.

Running specific suites
-----------------------
  # AWS Bedrock
  export MTTR_PROVIDER=bedrock
  export MTTR_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
  export MTTR_AWS_REGION=us-east-1
  pytest test_live.py -v -m bedrock

  # Azure OpenAI
  export MTTR_PROVIDER=azure_openai
  export MTTR_AZURE_ENDPOINT=https://my-resource.openai.azure.com
  export MTTR_AZURE_DEPLOYMENT=gpt-4o
  export AZURE_OPENAI_API_KEY=...
  pytest test_live.py -v -m azure

  # GCP Vertex AI
  export MTTR_PROVIDER=vertex_ai
  export MTTR_GCP_PROJECT=my-gcp-project
  export MTTR_MODEL_ID=gemini-1.5-pro-002
  gcloud auth application-default login
  pytest test_live.py -v -m vertex

  # All live providers at once
  pytest test_live.py -v -m live

  # Telemetry sinks
  export MTTR_SINK_CLOUDWATCH=true
  pytest test_live.py -v -m cloudwatch
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

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
from mttr_a.providers import LLMResponse
from mttr_a.sinks import CloudWatchSink, AzureMonitorSink, GCPLoggingSink


# ── Skip guards ───────────────────────────────────────────────────────────────

def _env(*keys: str) -> bool:
    """True if every key is set and non-empty."""
    return all(os.getenv(k) for k in keys)


skip_bedrock = pytest.mark.skipif(
    not _env("MTTR_PROVIDER", "MTTR_MODEL_ID") or os.getenv("MTTR_PROVIDER") != "bedrock",
    reason="Set MTTR_PROVIDER=bedrock + MTTR_MODEL_ID + AWS credentials to run",
)

skip_azure = pytest.mark.skipif(
    not _env("MTTR_PROVIDER", "MTTR_AZURE_ENDPOINT", "MTTR_AZURE_DEPLOYMENT", "AZURE_OPENAI_API_KEY")
    or os.getenv("MTTR_PROVIDER") != "azure_openai",
    reason="Set MTTR_PROVIDER=azure_openai + endpoint + deployment + AZURE_OPENAI_API_KEY to run",
)

skip_vertex = pytest.mark.skipif(
    not _env("MTTR_PROVIDER", "MTTR_GCP_PROJECT")
    or os.getenv("MTTR_PROVIDER") != "vertex_ai",
    reason="Set MTTR_PROVIDER=vertex_ai + MTTR_GCP_PROJECT + ADC to run",
)

skip_cloudwatch = pytest.mark.skipif(
    os.getenv("MTTR_SINK_CLOUDWATCH", "").lower() != "true",
    reason="Set MTTR_SINK_CLOUDWATCH=true + AWS credentials to run",
)

skip_azure_monitor = pytest.mark.skipif(
    os.getenv("MTTR_SINK_AZURE_MONITOR", "").lower() != "true"
    or not _env("APPLICATIONINSIGHTS_CONNECTION_STRING"),
    reason="Set MTTR_SINK_AZURE_MONITOR=true + APPLICATIONINSIGHTS_CONNECTION_STRING to run",
)

skip_gcp_logging = pytest.mark.skipif(
    os.getenv("MTTR_SINK_GCP_LOGGING", "").lower() != "true"
    or not _env("MTTR_GCP_PROJECT"),
    reason="Set MTTR_SINK_GCP_LOGGING=true + MTTR_GCP_PROJECT + ADC to run",
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _live_config(tmp_path: Path, n_runs: int = 5) -> BenchmarkConfig:
    """Load env-based config with a short run count for live tests."""
    cfg = load_from_env()
    return BenchmarkConfig(
        provider=cfg.provider,
        n_runs=n_runs,
        seed=cfg.seed,
        drift_threshold=cfg.drift_threshold,
        stochastic_fault_rate=cfg.stochastic_fault_rate,
        mtbf_mean=cfg.mtbf_mean,
        alpha=cfg.alpha,
        telemetry_path=str(tmp_path / "live_telemetry.jsonl"),
        results_path=str(tmp_path / "live_results.json"),
        verbose=True,
    )


# ── AWS Bedrock ───────────────────────────────────────────────────────────────

@pytest.mark.live
@pytest.mark.bedrock
class TestBedrockProvider:
    @skip_bedrock
    def test_health_check(self):
        cfg = load_from_env()
        provider = build_provider(cfg.provider, seed=0)
        assert provider.health_check() is True

    @skip_bedrock
    def test_invoke_returns_response(self):
        cfg = load_from_env()
        provider = build_provider(cfg.provider, seed=0)
        result = provider.invoke("What is MTTR?")
        assert isinstance(result, LLMResponse)
        assert len(result.content) > 0
        assert 0.0 <= result.confidence <= 1.0
        assert result.latency_s > 0.0

    @skip_bedrock
    def test_confidence_in_range(self):
        cfg = load_from_env()
        provider = build_provider(cfg.provider, seed=0)
        result = provider.invoke("Explain cognitive drift in agentic systems.")
        assert 0.0 <= result.confidence <= 1.0

    @skip_bedrock
    def test_full_benchmark_run(self, tmp_path):
        cfg = _live_config(tmp_path, n_runs=5)
        provider = build_provider(cfg.provider, seed=cfg.seed)
        sink = JsonlSink(cfg.telemetry_path)
        metrics = ProductionRunner(cfg, provider, sink).run()
        assert metrics.n_runs == 5
        assert metrics.mtbf_sys > 0.0
        assert Path(cfg.telemetry_path).exists()
        assert sum(1 for _ in Path(cfg.telemetry_path).open()) == 5


# ── Azure OpenAI ──────────────────────────────────────────────────────────────

@pytest.mark.live
@pytest.mark.azure
class TestAzureOpenAIProvider:
    @skip_azure
    def test_health_check(self):
        cfg = load_from_env()
        provider = build_provider(cfg.provider, seed=0)
        assert provider.health_check() is True

    @skip_azure
    def test_invoke_returns_response(self):
        cfg = load_from_env()
        provider = build_provider(cfg.provider, seed=0)
        result = provider.invoke("What is MTTR?")
        assert isinstance(result, LLMResponse)
        assert len(result.content) > 0
        assert 0.0 <= result.confidence <= 1.0

    @skip_azure
    def test_recovery_contexts(self):
        cfg = load_from_env()
        provider = build_provider(cfg.provider, seed=0)
        for context in ("recovery", "retry", "rollback"):
            result = provider.invoke("Explain cognitive drift.", context=context)
            assert 0.0 <= result.confidence <= 1.0

    @skip_azure
    def test_full_benchmark_run(self, tmp_path):
        cfg = _live_config(tmp_path, n_runs=5)
        provider = build_provider(cfg.provider, seed=cfg.seed)
        sink = JsonlSink(cfg.telemetry_path)
        metrics = ProductionRunner(cfg, provider, sink).run()
        assert metrics.n_runs == 5
        assert metrics.mtbf_sys > 0.0


# ── GCP Vertex AI ─────────────────────────────────────────────────────────────

@pytest.mark.live
@pytest.mark.vertex
class TestVertexAIProvider:
    @skip_vertex
    def test_health_check(self):
        cfg = load_from_env()
        provider = build_provider(cfg.provider, seed=0)
        assert provider.health_check() is True

    @skip_vertex
    def test_invoke_returns_response(self):
        cfg = load_from_env()
        provider = build_provider(cfg.provider, seed=0)
        result = provider.invoke("What is MTTR?")
        assert isinstance(result, LLMResponse)
        assert len(result.content) > 0
        assert 0.0 <= result.confidence <= 1.0

    @skip_vertex
    def test_full_benchmark_run(self, tmp_path):
        cfg = _live_config(tmp_path, n_runs=5)
        provider = build_provider(cfg.provider, seed=cfg.seed)
        sink = JsonlSink(cfg.telemetry_path)
        metrics = ProductionRunner(cfg, provider, sink).run()
        assert metrics.n_runs == 5
        assert metrics.mtbf_sys > 0.0


# ── Telemetry sinks ───────────────────────────────────────────────────────────

@pytest.mark.live
class TestCloudWatchSink:
    @skip_cloudwatch
    def test_emit_single_episode(self):
        from test_production import _ep
        region = os.getenv("MTTR_AWS_REGION", "us-east-1")
        sink = CloudWatchSink(region=region)
        sink.emit(_ep(drift_detected=True, reflex_mode="auto-replan", delta_t=1.2,
                      t_detect=0.1, t_decide=0.05, t_execute=1.05))

    @skip_cloudwatch
    def test_emit_no_drift_episode(self):
        from test_production import _ep
        region = os.getenv("MTTR_AWS_REGION", "us-east-1")
        sink = CloudWatchSink(region=region)
        sink.emit(_ep(drift_detected=False))


@pytest.mark.live
class TestAzureMonitorSink:
    @skip_azure_monitor
    def test_emit_episode(self):
        from test_production import _ep
        conn = os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"]
        sink = AzureMonitorSink(connection_string=conn)
        sink.emit(_ep(drift_detected=True, reflex_mode="rollback", delta_t=0.8))
        sink.flush()


@pytest.mark.live
class TestGCPLoggingSink:
    @skip_gcp_logging
    def test_emit_episode(self):
        from test_production import _ep
        project = os.environ["MTTR_GCP_PROJECT"]
        sink = GCPLoggingSink(project=project)
        sink.emit(_ep(drift_detected=True, reflex_mode="tool-retry", delta_t=0.5))
        sink.flush()


# ── End-to-end with composite sink ───────────────────────────────────────────

@pytest.mark.live
class TestLiveEndToEnd:
    """Full pipeline with whatever provider + sinks are configured."""

    @pytest.mark.skipif(
        os.getenv("MTTR_PROVIDER", "mock") == "mock",
        reason="Set MTTR_PROVIDER to a real provider to run end-to-end live test",
    )
    def test_full_pipeline(self, tmp_path):
        cfg = _live_config(tmp_path, n_runs=10)
        provider = build_provider(cfg.provider, seed=cfg.seed)

        assert provider.health_check(), "Provider health check failed"

        sinks = [JsonlSink(cfg.telemetry_path)]
        if os.getenv("MTTR_SINK_CLOUDWATCH", "").lower() == "true":
            sinks.append(CloudWatchSink(region=cfg.provider.aws_region))
        if os.getenv("MTTR_SINK_AZURE_MONITOR", "").lower() == "true":
            sinks.append(AzureMonitorSink(os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"]))
        if os.getenv("MTTR_SINK_GCP_LOGGING", "").lower() == "true":
            sinks.append(GCPLoggingSink(project=cfg.provider.gcp_project))

        sink = CompositeSink(*sinks)
        metrics = ProductionRunner(cfg, provider, sink).run()

        assert metrics.n_runs == 10
        assert metrics.n_runs == sum(1 for _ in Path(cfg.telemetry_path).open())
        assert 0.0 <= metrics.pi_up_sys <= 1.0
        assert metrics.mtbf_sys > 0.0
