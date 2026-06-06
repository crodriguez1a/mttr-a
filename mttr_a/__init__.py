"""
mttr_a — Production package for MTTR-A cognitive reliability measurement.

Public API
----------
  BenchmarkConfig   full runtime configuration
  ProviderConfig    LLM provider settings
  ProviderKind      enum: mock | bedrock | azure_openai | vertex_ai
  build_provider    instantiate the right provider from config
  ProductionRunner  drive N LangGraph episodes → SystemMetrics
  JsonlSink         local JSONL telemetry
  CloudWatchSink    AWS CloudWatch telemetry
  AzureMonitorSink  Azure Monitor telemetry
  GCPLoggingSink    GCP Cloud Logging telemetry
  CompositeSink     fan-out to multiple sinks
  load_from_env     build BenchmarkConfig from environment variables

The metrics / data-model layer lives in mttr_a_simulation and is unchanged.
"""

from .config import BenchmarkConfig, ProviderConfig, SinkConfig, ProviderKind, load_from_env
from .providers import (
    BaseLLMProvider,
    LLMResponse,
    ToolCall,
    MockProvider,
    BedrockProvider,
    AzureOpenAIProvider,
    VertexAIProvider,
    build_provider,
)
from .runner import ProductionRunner
from .sinks import (
    TelemetrySink,
    JsonlSink,
    CloudWatchSink,
    AzureMonitorSink,
    GCPLoggingSink,
    CompositeSink,
)

__all__ = [
    "BenchmarkConfig",
    "ProviderConfig",
    "SinkConfig",
    "ProviderKind",
    "load_from_env",
    "BaseLLMProvider",
    "LLMResponse",
    "ToolCall",
    "MockProvider",
    "BedrockProvider",
    "AzureOpenAIProvider",
    "VertexAIProvider",
    "build_provider",
    "ProductionRunner",
    "TelemetrySink",
    "JsonlSink",
    "CloudWatchSink",
    "AzureMonitorSink",
    "GCPLoggingSink",
    "CompositeSink",
]
