"""
CLI entrypoint for the MTTR-A production package.

Usage
-----
  python -m mttr_a            # reads all config from environment
  mttr-a                      # same, via installed console script

All settings are controlled by environment variables — see .env.example.

Sink fan-out
------------
JSONL is always written.  Add cloud sinks by setting:
  MTTR_SINK_CLOUDWATCH=true          (AWS CloudWatch)
  MTTR_SINK_AZURE_MONITOR=true       (Azure Monitor)
  MTTR_SINK_GCP_LOGGING=true         (GCP Cloud Logging)
"""

from __future__ import annotations

import sys

from mttr_a_simulation import Reporter, ResultsSaver

from . import (
    CompositeSink,
    JsonlSink,
    ProductionRunner,
    build_provider,
    load_from_env,
)
from .sinks import AzureMonitorSink, CloudWatchSink, GCPLoggingSink


def _build_sink(cfg) -> CompositeSink:
    """Assemble the telemetry fan-out from cfg.sinks (all vars from load_from_env)."""
    s = cfg.sinks
    sinks = [JsonlSink(cfg.telemetry_path)]

    if s.cloudwatch:
        sinks.append(
            CloudWatchSink(region=cfg.provider.aws_region, namespace=s.cloudwatch_namespace)
        )

    if s.azure_monitor:
        if not s.applicationinsights_connection_string:
            print(
                "ERROR: MTTR_SINK_AZURE_MONITOR=true but "
                "APPLICATIONINSIGHTS_CONNECTION_STRING is not set.",
                file=sys.stderr,
            )
            sys.exit(1)
        sinks.append(AzureMonitorSink(connection_string=s.applicationinsights_connection_string))

    if s.gcp_logging:
        if not cfg.provider.gcp_project:
            print(
                "ERROR: MTTR_SINK_GCP_LOGGING=true but MTTR_GCP_PROJECT is not set.",
                file=sys.stderr,
            )
            sys.exit(1)
        sinks.append(GCPLoggingSink(project=cfg.provider.gcp_project))

    return CompositeSink(*sinks)


def main() -> None:
    cfg = load_from_env()
    provider_label = cfg.provider.kind.value

    print(f"MTTR-A  provider={provider_label}  n_runs={cfg.n_runs}  seed={cfg.seed}")

    provider = build_provider(cfg.provider, seed=cfg.seed)

    if provider_label != "mock":
        print("Health-checking provider...", end=" ", flush=True)
        ok = provider.health_check()
        print("OK" if ok else "FAILED — check credentials and network access")
        if not ok:
            sys.exit(1)

    sink = _build_sink(cfg)
    metrics = ProductionRunner(cfg, provider, sink).run()

    Reporter().print_report(metrics)

    ResultsSaver().save(metrics, cfg.results_path)
    print(f"\nResults  → {cfg.results_path}")
    print(f"Telemetry → {cfg.telemetry_path}")


if __name__ == "__main__":
    main()
