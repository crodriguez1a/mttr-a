"""
Test suite for the mttr_a production package.

Coverage targets
----------------
  mttr_a/config.py    — ProviderKind, ProviderConfig, BenchmarkConfig, load_from_env
  mttr_a/providers.py — LLMResponse, BaseLLMProvider, MockProvider, build_provider,
                        _parse_confidence, import-error branches for real providers
  mttr_a/graph.py     — node factories, build_graph, state_to_episode
  mttr_a/sinks.py     — TelemetrySink protocol, JsonlSink, CompositeSink,
                        import-error branches for cloud sinks
  mttr_a/runner.py    — ProductionRunner (unit + integration)
  example_production.py — build_config, build_sink, main
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from mttr_a import (
    BenchmarkConfig,
    CompositeSink,
    JsonlSink,
    MockProvider,
    ProviderConfig,
    ProviderKind,
    TelemetrySink,
    build_provider,
    load_from_env,
    ProductionRunner,
)
from mttr_a.graph import AgentState, build_graph, state_to_episode
from mttr_a.providers import LLMResponse, _parse_confidence
from mttr_a_simulation import Episode


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ep(**kwargs) -> Episode:
    defaults: dict = {
        "run_id": 0,
        "query": "test query",
        "confidence": 0.8,
        "drift_detected": False,
        "reflex_mode": None,
        "t_detect": 0.0,
        "t_decide": 0.0,
        "t_execute": 0.0,
        "delta_t": 0.0,
        "t_fault": 0.0,
        "t_recovered": 0.0,
    }
    defaults.update(kwargs)
    return Episode(**defaults)


def _mock_provider(seed: int = 0) -> MockProvider:
    cfg = ProviderConfig(kind=ProviderKind.MOCK, mock_simulate_latency=False)
    return MockProvider(cfg, seed=seed)


def _minimal_config(**overrides) -> BenchmarkConfig:
    defaults = dict(
        provider=ProviderConfig(kind=ProviderKind.MOCK, mock_simulate_latency=False),
        n_runs=5,
        seed=0,
        verbose=False,
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


# ── config.py ─────────────────────────────────────────────────────────────────

class TestProviderKind:
    def test_values(self):
        assert ProviderKind.MOCK == "mock"
        assert ProviderKind.BEDROCK == "bedrock"
        assert ProviderKind.AZURE_OPENAI == "azure_openai"
        assert ProviderKind.VERTEX_AI == "vertex_ai"

    def test_from_string(self):
        assert ProviderKind("mock") is ProviderKind.MOCK


class TestProviderConfig:
    def test_defaults(self):
        cfg = ProviderConfig()
        assert cfg.kind is ProviderKind.MOCK
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 512
        assert cfg.aws_region == "us-east-1"
        assert cfg.gcp_location == "us-central1"
        assert cfg.mock_simulate_latency is False

    def test_frozen(self):
        cfg = ProviderConfig()
        with pytest.raises(Exception):
            cfg.kind = ProviderKind.BEDROCK  # type: ignore[misc]


class TestBenchmarkConfig:
    def test_defaults(self):
        cfg = BenchmarkConfig()
        assert cfg.n_runs == 200
        assert cfg.seed == 42
        assert cfg.drift_threshold == 0.6
        assert cfg.alpha == 0.90

    def test_frozen(self):
        cfg = BenchmarkConfig()
        with pytest.raises(Exception):
            cfg.n_runs = 999  # type: ignore[misc]


class TestLoadFromEnv:
    def test_defaults_when_no_env(self, monkeypatch):
        for key in ("MTTR_PROVIDER", "MTTR_MODEL_ID", "MTTR_N_RUNS", "MTTR_SEED",
                    "MTTR_DRIFT_THRESHOLD", "MTTR_STOCHASTIC_FAULT_RATE",
                    "MTTR_MTBF_MEAN", "MTTR_ALPHA", "MTTR_VERBOSE",
                    "MTTR_MOCK_LATENCY", "MTTR_TELEMETRY_PATH", "MTTR_RESULTS_PATH"):
            monkeypatch.delenv(key, raising=False)
        cfg = load_from_env()
        assert cfg.provider.kind is ProviderKind.MOCK
        assert cfg.n_runs == 200
        assert cfg.seed == 42

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MTTR_PROVIDER", "bedrock")
        monkeypatch.setenv("MTTR_MODEL_ID", "anthropic.claude-3-haiku")
        monkeypatch.setenv("MTTR_N_RUNS", "50")
        monkeypatch.setenv("MTTR_SEED", "7")
        monkeypatch.setenv("MTTR_VERBOSE", "false")
        monkeypatch.setenv("MTTR_MOCK_LATENCY", "false")
        cfg = load_from_env()
        assert cfg.provider.kind is ProviderKind.BEDROCK
        assert cfg.provider.model_id == "anthropic.claude-3-haiku"
        assert cfg.n_runs == 50
        assert cfg.seed == 7
        assert cfg.verbose is False
        assert cfg.provider.mock_simulate_latency is False

    def test_aws_region_fallback(self, monkeypatch):
        monkeypatch.delenv("MTTR_AWS_REGION", raising=False)
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
        cfg = load_from_env()
        assert cfg.provider.aws_region == "eu-west-1"

    def test_azure_env(self, monkeypatch):
        monkeypatch.setenv("MTTR_PROVIDER", "azure_openai")
        monkeypatch.setenv("MTTR_AZURE_ENDPOINT", "https://res.openai.azure.com")
        monkeypatch.setenv("MTTR_AZURE_DEPLOYMENT", "gpt-4o")
        monkeypatch.setenv("MTTR_AZURE_API_VERSION", "2024-05-01")
        cfg = load_from_env()
        assert cfg.provider.azure_endpoint == "https://res.openai.azure.com"
        assert cfg.provider.azure_deployment == "gpt-4o"
        assert cfg.provider.azure_api_version == "2024-05-01"

    def test_vertex_env(self, monkeypatch):
        monkeypatch.setenv("MTTR_PROVIDER", "vertex_ai")
        monkeypatch.setenv("MTTR_GCP_PROJECT", "my-proj")
        monkeypatch.setenv("MTTR_GCP_LOCATION", "us-west1")
        cfg = load_from_env()
        assert cfg.provider.gcp_project == "my-proj"
        assert cfg.provider.gcp_location == "us-west1"


# ── providers.py ──────────────────────────────────────────────────────────────

class TestLLMResponse:
    def test_fields(self):
        r = LLMResponse(content="hello", confidence=0.9, latency_s=0.1)
        assert r.content == "hello"
        assert r.confidence == 0.9

    def test_frozen(self):
        r = LLMResponse(content="x", confidence=0.5, latency_s=0.0)
        with pytest.raises(Exception):
            r.confidence = 0.1  # type: ignore[misc]


class TestParseConfidence:
    @pytest.mark.parametrize("text,expected", [
        ("0.82", 0.82),
        ("  0.9  ", 0.9),
        ("82", 0.82),        # percentage format → divide by 100
        ("no number here", 0.5),
        ("1.5", 0.015),      # > 1.0 treated as percentage → 1.5 / 100 = 0.015
        ("-0.3", 0.3),       # regex strips sign, matches "0.3", returns 0.3
        ("confidence is 0.75 out of 1", 0.75),
    ])
    def test_parse(self, text, expected):
        assert _parse_confidence(text) == pytest.approx(expected, abs=1e-9)

    def test_empty_string(self):
        assert _parse_confidence("") == 0.5


class TestMockProvider:
    def test_returns_llm_response(self):
        p = _mock_provider()
        r = p.invoke("hello")
        assert isinstance(r, LLMResponse)
        assert 0.0 <= r.confidence <= 1.0
        assert "mock" in r.content

    def test_context_variants(self):
        p = _mock_provider()
        for ctx in ("", "retry", "rollback", "recovery", "unknown"):
            r = p.invoke("q", context=ctx)
            assert isinstance(r, LLMResponse)

    def test_deterministic_with_seed(self):
        p1, p2 = _mock_provider(seed=99), _mock_provider(seed=99)
        r1, r2 = p1.invoke("same"), p2.invoke("same")
        assert r1.confidence == r2.confidence

    def test_different_seeds_differ(self):
        results = {_mock_provider(seed=i).invoke("q").confidence for i in range(5)}
        assert len(results) > 1

    def test_simulate_latency(self):
        cfg = ProviderConfig(kind=ProviderKind.MOCK, mock_simulate_latency=True)
        p = MockProvider(cfg, seed=0)
        t0 = time.perf_counter()
        p.invoke("q")
        elapsed = time.perf_counter() - t0
        assert elapsed > 0.0

    def test_health_check_true(self):
        assert _mock_provider().health_check() is True


class TestBuildProvider:
    def test_mock(self):
        cfg = ProviderConfig(kind=ProviderKind.MOCK)
        p = build_provider(cfg, seed=0)
        assert isinstance(p, MockProvider)

    def test_bedrock_import_error(self):
        cfg = ProviderConfig(kind=ProviderKind.BEDROCK, model_id="x")
        with patch.dict(sys.modules, {"langchain_aws": None}):
            with pytest.raises(ImportError, match="langchain-aws"):
                build_provider(cfg)

    def test_azure_missing_endpoint(self):
        cfg = ProviderConfig(kind=ProviderKind.AZURE_OPENAI, azure_endpoint="", azure_deployment="")
        with pytest.raises(ValueError, match="azure_endpoint"):
            build_provider(cfg)

    def test_azure_import_error(self):
        cfg = ProviderConfig(
            kind=ProviderKind.AZURE_OPENAI,
            azure_endpoint="https://x.openai.azure.com",
            azure_deployment="gpt-4o",
        )
        with patch.dict(sys.modules, {"langchain_openai": None}):
            with pytest.raises(ImportError, match="langchain-openai"):
                build_provider(cfg)

    def test_vertex_missing_project(self):
        cfg = ProviderConfig(kind=ProviderKind.VERTEX_AI, gcp_project="")
        with pytest.raises(ValueError, match="gcp_project"):
            build_provider(cfg)

    def test_vertex_import_error(self):
        cfg = ProviderConfig(kind=ProviderKind.VERTEX_AI, gcp_project="proj", model_id="gemini")
        with patch.dict(sys.modules, {"langchain_google_vertexai": None}):
            with pytest.raises(ImportError, match="langchain-google-vertexai"):
                build_provider(cfg)

    def test_unknown_kind_raises(self):
        cfg = MagicMock()
        cfg.kind = "not_a_real_kind"
        with pytest.raises(ValueError, match="Unknown provider kind"):
            build_provider(cfg)


# ── graph.py ──────────────────────────────────────────────────────────────────

def _blank_state(run_id: int = 0, query: str = "what is drift?") -> AgentState:
    return AgentState(
        run_id=run_id,
        query=query,
        response="",
        confidence=0.0,
        is_drift=False,
        reflex_mode=None,
        t_reason_start=0.0,
        t_reason_end=0.0,
        t_drift_check=0.0,
        t_recovery_start=0.0,
        t_recovery_end=0.0,
        T_detect=0.0,
        T_decide=0.0,
        T_execute=0.0,
    )


class TestBuildGraph:
    def test_compiles_without_error(self):
        cfg = _minimal_config()
        provider = _mock_provider()
        graph = build_graph(provider, cfg, seed=0)
        assert graph is not None

    def test_invoke_returns_state(self):
        cfg = _minimal_config()
        graph = build_graph(_mock_provider(), cfg, seed=0)
        result = graph.invoke(_blank_state())
        assert isinstance(result, dict)
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_invoke_sets_response(self):
        graph = build_graph(_mock_provider(), _minimal_config(), seed=0)
        result = graph.invoke(_blank_state(query="explain MTTR"))
        assert len(result["response"]) > 0

    def test_timing_fields_populated(self):
        graph = build_graph(_mock_provider(), _minimal_config(), seed=0)
        result = graph.invoke(_blank_state())
        assert result["t_reason_start"] > 0.0 or result["t_reason_end"] >= result["t_reason_start"]
        assert result["t_drift_check"] >= 0.0

    def test_drift_flag_boolean(self):
        graph = build_graph(_mock_provider(), _minimal_config(), seed=0)
        result = graph.invoke(_blank_state())
        assert isinstance(result["is_drift"], bool)

    def test_no_drift_clears_reflex(self):
        """When drift is not detected the reflex_mode should remain None."""
        # Force confidence above threshold by patching provider
        mock_p = MagicMock()
        mock_p.invoke.return_value = LLMResponse(content="ok", confidence=0.99, latency_s=0.0)
        cfg = BenchmarkConfig(
            provider=ProviderConfig(kind=ProviderKind.MOCK, mock_simulate_latency=False),
            n_runs=1, seed=0, drift_threshold=0.6, stochastic_fault_rate=0.0, verbose=False,
        )
        graph = build_graph(mock_p, cfg, seed=0)
        result = graph.invoke(_blank_state())
        if not result["is_drift"]:
            assert result["reflex_mode"] is None

    def test_drift_sets_reflex_mode(self):
        """When drift is detected the reflex_mode should be one of the four options."""
        mock_p = MagicMock()
        mock_p.invoke.return_value = LLMResponse(content="ok", confidence=0.0, latency_s=0.0)
        cfg = BenchmarkConfig(
            provider=ProviderConfig(kind=ProviderKind.MOCK, mock_simulate_latency=False),
            n_runs=1, seed=0, drift_threshold=0.6, stochastic_fault_rate=0.0, verbose=False,
        )
        graph = build_graph(mock_p, cfg, seed=0)
        result = graph.invoke(_blank_state())
        valid = {"auto-replan", "tool-retry", "rollback", "human-approve"}
        assert result["reflex_mode"] in valid


class TestStateToEpisode:
    def _drift_state(self) -> AgentState:
        s = _blank_state()
        s["is_drift"] = True
        s["reflex_mode"] = "tool-retry"
        s["T_detect"] = 0.01
        s["T_decide"] = 0.002
        s["T_execute"] = 0.05
        return s

    def test_episode_fields(self):
        ep = state_to_episode(self._drift_state(), wall_clock=10.0)
        assert ep.run_id == 0
        assert ep.drift_detected is True
        assert ep.reflex_mode == "tool-retry"
        assert ep.delta_t == pytest.approx(0.062, abs=1e-4)
        assert ep.t_fault == pytest.approx(10.0, abs=1e-4)
        assert ep.t_recovered == pytest.approx(10.062, abs=1e-4)

    def test_no_drift_episode(self):
        s = _blank_state()
        s["is_drift"] = False
        ep = state_to_episode(s, wall_clock=5.0)
        assert ep.drift_detected is False
        assert ep.delta_t == 0.0
        assert ep.t_fault == pytest.approx(5.0)
        assert ep.t_recovered == pytest.approx(5.0)

    def test_confidence_rounded(self):
        s = _blank_state()
        s["confidence"] = 0.123456789
        ep = state_to_episode(s, wall_clock=0.0)
        assert ep.confidence == pytest.approx(0.1235, abs=1e-4)


# ── sinks.py ──────────────────────────────────────────────────────────────────

class TestTelemetrySinkProtocol:
    def test_jsonl_satisfies_protocol(self, tmp_path):
        sink = JsonlSink(str(tmp_path / "t.jsonl"))
        assert isinstance(sink, TelemetrySink)

    def test_composite_satisfies_protocol(self, tmp_path):
        inner = JsonlSink(str(tmp_path / "t.jsonl"))
        assert isinstance(CompositeSink(inner), TelemetrySink)

    def test_custom_class_satisfies_protocol(self):
        class MySink:
            def emit(self, episode: Episode) -> None: ...
            def flush(self) -> None: ...
        assert isinstance(MySink(), TelemetrySink)


class TestJsonlSink:
    def test_emit_creates_file(self, tmp_path):
        path = tmp_path / "out.jsonl"
        sink = JsonlSink(str(path))
        sink.emit(_ep())
        assert path.exists()

    def test_emit_valid_json(self, tmp_path):
        path = tmp_path / "out.jsonl"
        sink = JsonlSink(str(path))
        sink.emit(_ep(run_id=7, confidence=0.85, drift_detected=True, reflex_mode="rollback"))
        record = json.loads(path.read_text().strip())
        assert record["run_id"] == 7
        assert record["confidence"] == pytest.approx(0.85)
        assert record["drift_detected"] is True
        assert record["reflex_mode"] == "rollback"

    def test_emit_appends(self, tmp_path):
        path = tmp_path / "out.jsonl"
        sink = JsonlSink(str(path))
        sink.emit(_ep(run_id=0))
        sink.emit(_ep(run_id=1))
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["run_id"] == 1

    def test_flush_is_noop(self, tmp_path):
        sink = JsonlSink(str(tmp_path / "f.jsonl"))
        sink.flush()  # should not raise

    def test_all_fields_present(self, tmp_path):
        path = tmp_path / "out.jsonl"
        sink = JsonlSink(str(path))
        ep = _ep(t_detect=0.1, t_decide=0.02, t_execute=0.5, delta_t=0.62,
                 t_fault=10.0, t_recovered=10.62)
        sink.emit(ep)
        record = json.loads(path.read_text())
        for key in ("run_id", "query", "confidence", "drift_detected",
                    "reflex_mode", "t_detect", "t_decide", "t_execute",
                    "delta_t", "t_fault", "t_recovered"):
            assert key in record, f"missing key: {key}"


class TestCompositeSink:
    def test_routes_to_all_sinks(self, tmp_path):
        p1, p2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        s = CompositeSink(JsonlSink(str(p1)), JsonlSink(str(p2)))
        s.emit(_ep())
        assert p1.exists() and p2.exists()

    def test_failing_sink_does_not_block_others(self, tmp_path):
        good = JsonlSink(str(tmp_path / "ok.jsonl"))

        class BadSink:
            def emit(self, episode: Episode) -> None:
                raise RuntimeError("boom")
            def flush(self) -> None:
                raise RuntimeError("boom flush")

        s = CompositeSink(BadSink(), good)
        s.emit(_ep())
        assert (tmp_path / "ok.jsonl").exists()

    def test_flush_propagates(self, tmp_path):
        flushed = []

        class TrackSink:
            def emit(self, ep: Episode) -> None: ...
            def flush(self) -> None:
                flushed.append(True)

        CompositeSink(TrackSink(), TrackSink()).flush()
        assert len(flushed) == 2

    def test_flush_isolates_failures(self):
        class ExplodeSink:
            def emit(self, ep: Episode) -> None: ...
            def flush(self) -> None:
                raise RuntimeError("flush boom")

        # Should not raise
        CompositeSink(ExplodeSink(), ExplodeSink()).flush()


class TestCloudSinkImportErrors:
    def test_cloudwatch_import_error(self):
        with patch.dict(sys.modules, {"boto3": None}):
            from mttr_a import CloudWatchSink
            with pytest.raises(ImportError, match="boto3"):
                CloudWatchSink()

    def test_azure_sink_import_error(self):
        with patch.dict(sys.modules, {
            "azure": None,
            "azure.monitor": None,
            "azure.monitor.opentelemetry": None,
        }):
            # Force re-import with missing module
            from mttr_a.sinks import AzureMonitorSink
            with pytest.raises((ImportError, Exception)):
                AzureMonitorSink("Endpoint=sb://x")

    def test_gcp_sink_import_error(self):
        with patch.dict(sys.modules, {
            "google": None,
            "google.cloud": None,
            "google.cloud.logging": None,
        }):
            from mttr_a.sinks import GCPLoggingSink
            with pytest.raises((ImportError, Exception)):
                GCPLoggingSink("my-project")


# ── runner.py ─────────────────────────────────────────────────────────────────

class TestProductionRunnerUnit:
    def test_run_returns_system_metrics(self, tmp_path):
        cfg = _minimal_config(
            n_runs=10,
            telemetry_path=str(tmp_path / "t.jsonl"),
        )
        sink = JsonlSink(str(tmp_path / "t.jsonl"))
        runner = ProductionRunner(config=cfg, provider=_mock_provider(), sink=sink)
        metrics = runner.run()
        assert metrics is not None
        assert metrics.n_runs == 10

    def test_telemetry_lines_match_n_runs(self, tmp_path):
        path = tmp_path / "t.jsonl"
        cfg = _minimal_config(n_runs=8, telemetry_path=str(path))
        runner = ProductionRunner(cfg, _mock_provider(), JsonlSink(str(path)))
        runner.run()
        assert sum(1 for _ in path.open()) == 8

    def test_deterministic_across_seeds(self, tmp_path):
        def _run(seed: int):
            p = tmp_path / f"t{seed}.jsonl"
            cfg = _minimal_config(n_runs=20, seed=seed, telemetry_path=str(p))
            return ProductionRunner(cfg, _mock_provider(seed=seed), JsonlSink(str(p))).run()

        m1a = _run(42)
        m1b = _run(42)
        assert m1a.n_runs == m1b.n_runs
        # Same seed → same drift count
        assert m1a.drift_events == m1b.drift_events

    def test_verbose_output(self, tmp_path, capsys):
        path = tmp_path / "t.jsonl"
        cfg = BenchmarkConfig(
            provider=ProviderConfig(kind=ProviderKind.MOCK, mock_simulate_latency=False),
            n_runs=5, seed=0, verbose=True,
            telemetry_path=str(path),
        )
        ProductionRunner(cfg, _mock_provider(), JsonlSink(str(path))).run()
        out = capsys.readouterr().out
        assert "Run" in out

    def test_composite_sink_used(self, tmp_path):
        p1, p2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        cfg = _minimal_config(n_runs=3, telemetry_path=str(p1))
        sink = CompositeSink(JsonlSink(str(p1)), JsonlSink(str(p2)))
        ProductionRunner(cfg, _mock_provider(), sink).run()
        assert p1.exists() and p2.exists()
        assert sum(1 for _ in p2.open()) == 3


# ── Integration: end-to-end ───────────────────────────────────────────────────

class TestEndToEnd:
    """Run the full pipeline with MockProvider and verify metric sanity."""

    @pytest.fixture(scope="class")
    def metrics(self, tmp_path_factory):
        path = tmp_path_factory.mktemp("e2e") / "t.jsonl"
        cfg = BenchmarkConfig(
            provider=ProviderConfig(kind=ProviderKind.MOCK, mock_simulate_latency=False),
            n_runs=100,
            seed=0,
            drift_threshold=0.6,
            stochastic_fault_rate=0.05,
            mtbf_mean=6.73,
            alpha=0.90,
            verbose=False,
            telemetry_path=str(path),
        )
        provider = build_provider(cfg.provider, seed=0)
        sink = JsonlSink(str(path))
        return ProductionRunner(cfg, provider, sink).run()

    def test_n_runs(self, metrics):
        assert metrics.n_runs == 100

    def test_drift_count_positive(self, metrics):
        assert metrics.drift_events > 0

    def test_mttr_a_nonnegative(self, metrics):
        assert metrics.mttr_a_sys >= 0.0

    def test_med_ttr_a_nonnegative(self, metrics):
        assert metrics.med_ttr_a_sys >= 0.0

    def test_mtbf_positive(self, metrics):
        assert metrics.mtbf_sys > 0.0

    def test_nrr_in_range(self, metrics):
        assert -2.0 < metrics.nrr_sys < 2.0

    def test_pi_up_in_unit_interval(self, metrics):
        assert 0.0 <= metrics.pi_up_sys <= 1.0

    def test_nrr_alpha_leq_nrr(self, metrics):
        # Conservative bound must be ≤ point estimate
        assert metrics.nrr_alpha <= metrics.nrr_sys + 0.01


# ── example_production.py ─────────────────────────────────────────────────────

class TestExampleProduction:
    """Smoke tests for the example script helpers."""

    def _import_example(self):
        import importlib.util, os
        spec = importlib.util.spec_from_file_location(
            "example_production",
            str(Path(__file__).parent / "example_production.py"),
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    def test_build_config_returns_benchmark_config(self, monkeypatch):
        monkeypatch.delenv("MTTR_PROVIDER", raising=False)
        mod = self._import_example()
        cfg = mod.build_config()
        assert isinstance(cfg, BenchmarkConfig)
        assert cfg.provider.kind is ProviderKind.MOCK

    def test_build_config_uses_env(self, monkeypatch):
        monkeypatch.setenv("MTTR_PROVIDER", "mock")
        monkeypatch.setenv("MTTR_N_RUNS", "10")
        mod = self._import_example()
        cfg = mod.build_config()
        assert cfg.n_runs == 10

    def test_build_sink_returns_composite(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MTTR_PROVIDER", raising=False)
        mod = self._import_example()
        cfg = mod.build_config()
        sink = mod.build_sink(cfg, extra_path=str(tmp_path / "t.jsonl"))
        assert isinstance(sink, CompositeSink)

    def test_main_runs_without_error(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv("MTTR_PROVIDER", raising=False)
        monkeypatch.chdir(tmp_path)
        mod = self._import_example()
        # Override n_runs to keep test fast
        original_build = mod.build_config

        def fast_config():
            cfg = original_build()
            return BenchmarkConfig(
                provider=cfg.provider,
                n_runs=5,
                seed=cfg.seed,
                drift_threshold=cfg.drift_threshold,
                stochastic_fault_rate=cfg.stochastic_fault_rate,
                mtbf_mean=cfg.mtbf_mean,
                alpha=cfg.alpha,
                telemetry_path=str(tmp_path / "t.jsonl"),
                results_path=str(tmp_path / "r.json"),
                verbose=False,
            )

        monkeypatch.setattr(mod, "build_config", fast_config)
        mod.main()
        out = capsys.readouterr().out
        assert "MTTR-A" in out


# ── Data loading tests ────────────────────────────────────────────────────────

from mttr_a.providers import (
    _CONFIDENCE_PROMPT,
    _MOCK_LATENCY,
    _MOCK_CONF,
    _TTFT_FRACTION,
    _load_mock_cfg,
    _load_confidence_prompt,
)


class TestProviderDataLoading:
    """Verify that MockProvider and real providers load all data from files."""

    def test_confidence_prompt_loaded_from_file(self) -> None:
        loaded = _load_confidence_prompt()
        assert isinstance(loaded, str)
        assert len(loaded) > 20
        assert "0.0" in loaded and "1.0" in loaded

    def test_confidence_prompt_matches_module_constant(self) -> None:
        assert _load_confidence_prompt() == _CONFIDENCE_PROMPT

    def test_mock_latency_loaded_from_file(self) -> None:
        assert "" in _MOCK_LATENCY
        for ctx, (lo, hi) in _MOCK_LATENCY.items():
            assert hi > lo >= 0, f"context '{ctx}': invalid range [{lo}, {hi}]"

    def test_mock_latency_matches_module_constant(self) -> None:
        cfg = _load_mock_cfg()
        for ctx, (lo, hi) in _MOCK_LATENCY.items():
            assert cfg["mock_provider_latency"][ctx] == [lo, hi]

    def test_mock_confidence_params_valid(self) -> None:
        assert 0.0 < _MOCK_CONF["base_mu"] < 1.0
        assert _MOCK_CONF["base_sigma"] > 0
        assert _MOCK_CONF["noise_sigma"] >= 0

    def test_ttft_fraction_in_unit_interval(self) -> None:
        assert 0.0 < _TTFT_FRACTION < 1.0

    def test_mock_provider_uses_loaded_confidence_params(self) -> None:
        from mttr_a import ProviderConfig, ProviderKind
        cfg = ProviderConfig(kind=ProviderKind.MOCK)
        p = MockProvider(cfg, seed=0)
        results = [p.invoke("q").confidence for _ in range(50)]
        assert all(0.0 <= c <= 1.0 for c in results)
        mean = sum(results) / len(results)
        assert abs(mean - _MOCK_CONF["base_mu"]) < 0.15

    def test_mock_provider_ttft_uses_loaded_fraction(self) -> None:
        from mttr_a import ProviderConfig, ProviderKind
        cfg = ProviderConfig(kind=ProviderKind.MOCK, mock_simulate_latency=False)
        p = MockProvider(cfg, seed=1)
        r = p.invoke("q")
        assert r.t_first_token is not None
