"""
Environment-driven configuration for the MTTR-A production package.

All settings have sensible defaults and can be overridden via environment
variables, making the package 12-factor compliant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


def _load_dotenv(env_path: str | Path | None = None) -> None:
    """
    Load variables from a .env file into os.environ.

    Search order when env_path is not given:
      1. .env in the current working directory
      2. .env one directory up (useful when running from a sub-directory)

    Already-set environment variables are never overwritten, so platform-injected
    secrets (Kubernetes, ECS, Cloud Run) always take precedence over the file.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return  # python-dotenv not installed; env vars must come from the environment

    if env_path is not None:
        load_dotenv(env_path, override=False)
        return

    for candidate in (Path.cwd() / ".env", Path.cwd().parent / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return


class ProviderKind(str, Enum):
    MOCK = "mock"
    BEDROCK = "bedrock"
    AZURE_OPENAI = "azure_openai"
    VERTEX_AI = "vertex_ai"


@dataclass(frozen=True)
class ProviderConfig:
    """LLM provider identity and credentials."""

    kind: ProviderKind = ProviderKind.MOCK

    # Model identity — meaning varies by provider
    #   Bedrock:    "anthropic.claude-3-5-sonnet-20241022-v2:0"
    #   Azure:      deployment name, e.g. "gpt-4o"
    #   Vertex AI:  "gemini-1.5-pro-002"
    model_id: str = "mock-model"

    temperature: float = 0.0
    max_tokens: int = 512

    # AWS Bedrock
    aws_region: str = "us-east-1"

    # Azure OpenAI
    azure_endpoint: str = ""       # https://<resource>.openai.azure.com
    azure_deployment: str = ""
    azure_api_version: str = "2024-02-01"

    # GCP Vertex AI
    gcp_project: str = ""
    gcp_location: str = "us-central1"

    # MockProvider: add realistic latency for demos without real API keys
    mock_simulate_latency: bool = False


@dataclass(frozen=True)
class SinkConfig:
    """Telemetry sink fan-out configuration. JSONL is always written."""

    # AWS CloudWatch Metrics
    cloudwatch: bool = False
    cloudwatch_namespace: str = "MTTR-A"

    # Azure Monitor / Application Insights
    azure_monitor: bool = False
    applicationinsights_connection_string: str = ""

    # GCP Cloud Logging
    gcp_logging: bool = False


@dataclass(frozen=True)
class BenchmarkConfig:
    """Full runtime configuration for a benchmark run."""

    provider: ProviderConfig = field(default_factory=ProviderConfig)
    sinks: SinkConfig = field(default_factory=SinkConfig)
    n_runs: int = 200
    seed: int = 42
    drift_threshold: float = 0.6        # τ_drift (Section IV-D)
    stochastic_fault_rate: float = 0.05 # 5% random perturbation
    mtbf_mean: float = 6.73             # Exp mean for stable intervals (s)
    alpha: float = 0.90                 # Confidence level for NRR_α
    telemetry_path: str = "telemetry.jsonl"
    results_path: str = "results.json"
    verbose: bool = True
    steps_per_episode: int = 3


def load_from_env(env_file: str | Path | None = None) -> BenchmarkConfig:
    """
    Build a BenchmarkConfig from environment variables.

    Automatically loads a .env file (current directory, then one level up) before
    reading variables, so no manual `export` is required.  Pass env_file to target
    a specific path.  Variables already in the environment always win over .env values,
    so platform-injected secrets (k8s, ECS, Cloud Run) are never overridden.

    Provider env vars
    -----------------
    MTTR_PROVIDER=bedrock
      MTTR_MODEL_ID       e.g. anthropic.claude-3-5-sonnet-20241022-v2:0
      MTTR_AWS_REGION     default: us-east-1 (falls back to AWS_DEFAULT_REGION)

    MTTR_PROVIDER=azure_openai
      MTTR_AZURE_ENDPOINT      e.g. https://my-resource.openai.azure.com
      MTTR_AZURE_DEPLOYMENT    e.g. gpt-4o
      AZURE_OPENAI_API_KEY     set in environment

    MTTR_PROVIDER=vertex_ai
      MTTR_GCP_PROJECT    GCP project ID
      MTTR_GCP_LOCATION   default: us-central1
      MTTR_MODEL_ID       e.g. gemini-1.5-pro-002

    Sink env vars
    -------------
    MTTR_SINK_CLOUDWATCH=true
      MTTR_CLOUDWATCH_NAMESPACE    default: MTTR-A

    MTTR_SINK_AZURE_MONITOR=true
      APPLICATIONINSIGHTS_CONNECTION_STRING   required

    MTTR_SINK_GCP_LOGGING=true
      (uses MTTR_GCP_PROJECT defined above)
    """
    _load_dotenv(env_file)

    kind = ProviderKind(os.getenv("MTTR_PROVIDER", "mock"))

    provider = ProviderConfig(
        kind=kind,
        model_id=os.getenv("MTTR_MODEL_ID", "mock-model"),
        temperature=float(os.getenv("MTTR_TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MTTR_MAX_TOKENS", "512")),
        aws_region=os.getenv("MTTR_AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")),
        azure_endpoint=os.getenv("MTTR_AZURE_ENDPOINT", ""),
        azure_deployment=os.getenv("MTTR_AZURE_DEPLOYMENT", ""),
        azure_api_version=os.getenv("MTTR_AZURE_API_VERSION", "2024-02-01"),
        gcp_project=os.getenv("MTTR_GCP_PROJECT", ""),
        gcp_location=os.getenv("MTTR_GCP_LOCATION", "us-central1"),
        mock_simulate_latency=os.getenv("MTTR_MOCK_LATENCY", "false").lower() == "true",
    )

    sinks = SinkConfig(
        cloudwatch=os.getenv("MTTR_SINK_CLOUDWATCH", "").lower() == "true",
        cloudwatch_namespace=os.getenv("MTTR_CLOUDWATCH_NAMESPACE", "MTTR-A"),
        azure_monitor=os.getenv("MTTR_SINK_AZURE_MONITOR", "").lower() == "true",
        applicationinsights_connection_string=os.getenv(
            "APPLICATIONINSIGHTS_CONNECTION_STRING", ""
        ),
        gcp_logging=os.getenv("MTTR_SINK_GCP_LOGGING", "").lower() == "true",
    )

    return BenchmarkConfig(
        provider=provider,
        sinks=sinks,
        n_runs=int(os.getenv("MTTR_N_RUNS", "200")),
        seed=int(os.getenv("MTTR_SEED", "42")),
        drift_threshold=float(os.getenv("MTTR_DRIFT_THRESHOLD", "0.6")),
        stochastic_fault_rate=float(os.getenv("MTTR_STOCHASTIC_FAULT_RATE", "0.05")),
        mtbf_mean=float(os.getenv("MTTR_MTBF_MEAN", "6.73")),
        alpha=float(os.getenv("MTTR_ALPHA", "0.90")),
        telemetry_path=os.getenv("MTTR_TELEMETRY_PATH", "telemetry.jsonl"),
        results_path=os.getenv("MTTR_RESULTS_PATH", "results.json"),
        verbose=os.getenv("MTTR_VERBOSE", "true").lower() == "true",
    )
