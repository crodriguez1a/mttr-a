"""
Enterprise telemetry sinks.

TelemetrySink is a Protocol — any object with .emit() and .flush() qualifies.
Use CompositeSink to fan out to multiple destinations simultaneously.

Available sinks
---------------
  JsonlSink        local JSONL file (default, no dependencies)
  CloudWatchSink   AWS CloudWatch Metrics (requires boto3)
  AzureMonitorSink Azure Monitor Custom Metrics (requires azure-monitor-opentelemetry)
  GCPLoggingSink   GCP Cloud Logging structured logs (requires google-cloud-logging)
  CompositeSink    fan-out to any number of sinks
"""

from __future__ import annotations

import json
import logging
from typing import Protocol, runtime_checkable

from mttr_a_simulation import Episode

logger = logging.getLogger(__name__)


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class TelemetrySink(Protocol):
    def emit(self, episode: Episode, extended: dict | None = None) -> None: ...
    def flush(self) -> None: ...


# ── JSONL (local) ─────────────────────────────────────────────────────────────

class JsonlSink:
    """
    Appends one JSON record per episode to a local file.
    Safe for single-process use; for multi-process write a rotating file handler.
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def emit(self, episode: Episode, extended: dict | None = None) -> None:
        record = {
            "run_id": episode.run_id,
            "query": episode.query,
            "confidence": episode.confidence,
            "drift_detected": episode.drift_detected,
            "reflex_mode": episode.reflex_mode,
            "t_detect": episode.t_detect,
            "t_decide": episode.t_decide,
            "t_execute": episode.t_execute,
            "delta_t": episode.delta_t,
            "t_fault": episode.t_fault,
            "t_recovered": episode.t_recovered,
        }
        if extended:
            record.update(extended)
        with open(self._path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def flush(self) -> None:
        pass  # writes are immediate


# ── AWS CloudWatch ────────────────────────────────────────────────────────────

class CloudWatchSink:
    """
    Emits MTTR-A episode metrics to AWS CloudWatch Metrics.

    Authentication: standard boto3 credential chain
      (IAM role → IRSA in EKS → env vars → ~/.aws/credentials)

    Metrics emitted per episode (namespace: MTTR-A):
      RecoveryLatency   Seconds   delta_t for drift episodes
      DriftDetected     Count     1/0 per episode
      ReflexMode        Count     1 per mode (dimensions used as labels)

    Install: pip install boto3
    """

    _NAMESPACE = "MTTR-A"

    def __init__(self, region: str = "us-east-1", namespace: str = "MTTR-A") -> None:
        try:
            import boto3
            self._client = boto3.client("cloudwatch", region_name=region)
        except ImportError as exc:
            raise ImportError("Install boto3: pip install boto3") from exc
        self._namespace = namespace

    def emit(self, episode: Episode, extended: dict | None = None) -> None:
        metric_data = [
            {
                "MetricName": "DriftDetected",
                "Value": float(episode.drift_detected),
                "Unit": "Count",
            },
        ]
        if episode.drift_detected and episode.delta_t > 0:
            metric_data += [
                {
                    "MetricName": "RecoveryLatency",
                    "Value": episode.delta_t,
                    "Unit": "Seconds",
                    "Dimensions": [
                        {"Name": "ReflexMode", "Value": episode.reflex_mode or "none"},
                    ],
                },
                {
                    "MetricName": "DetectLatency",
                    "Value": episode.t_detect,
                    "Unit": "Seconds",
                },
                {
                    "MetricName": "ExecuteLatency",
                    "Value": episode.t_execute,
                    "Unit": "Seconds",
                    "Dimensions": [
                        {"Name": "ReflexMode", "Value": episode.reflex_mode or "none"},
                    ],
                },
            ]
        if extended:
            if extended.get("queue_latency_s"):
                metric_data.append({
                    "MetricName": "QueueLatency",
                    "Value": extended["queue_latency_s"],
                    "Unit": "Seconds",
                })
            if extended.get("tool_latency_s"):
                metric_data.append({
                    "MetricName": "ToolLatency",
                    "Value": extended["tool_latency_s"],
                    "Unit": "Seconds",
                })
        try:
            self._client.put_metric_data(
                Namespace=self._namespace,
                MetricData=metric_data,
            )
        except Exception as exc:
            logger.warning("CloudWatch emit failed: %s", exc)

    def flush(self) -> None:
        pass


# ── Azure Monitor ─────────────────────────────────────────────────────────────

class AzureMonitorSink:
    """
    Emits structured logs to Azure Monitor via the Python logging handler.

    Authentication: APPLICATIONINSIGHTS_CONNECTION_STRING environment variable
      or DefaultAzureCredential for managed identity.

    Install: pip install azure-monitor-opentelemetry
    """

    def __init__(self, connection_string: str) -> None:
        try:
            from azure.monitor.opentelemetry import configure_azure_monitor
            configure_azure_monitor(connection_string=connection_string)
        except ImportError as exc:
            raise ImportError(
                "Install azure-monitor-opentelemetry: "
                "pip install azure-monitor-opentelemetry"
            ) from exc
        self._log = logging.getLogger("mttr_a.azure")

    def emit(self, episode: Episode, extended: dict | None = None) -> None:
        dimensions = {
            "run_id": episode.run_id,
            "drift_detected": episode.drift_detected,
            "reflex_mode": episode.reflex_mode,
            "delta_t": episode.delta_t,
            "t_detect": episode.t_detect,
            "t_decide": episode.t_decide,
            "t_execute": episode.t_execute,
            "confidence": episode.confidence,
        }
        if extended:
            dimensions.update(extended)
        self._log.info(
            "mttr_a_episode",
            extra={"custom_dimensions": dimensions},
        )

    def flush(self) -> None:
        pass


# ── GCP Cloud Logging ─────────────────────────────────────────────────────────

class GCPLoggingSink:
    """
    Emits structured logs to GCP Cloud Logging.

    Authentication: Application Default Credentials
      gcloud auth application-default login

    Logs appear under the log name "mttr_a_episodes" in your GCP project.

    Install: pip install google-cloud-logging
    """

    def __init__(self, project: str, log_name: str = "mttr_a_episodes") -> None:
        try:
            import google.cloud.logging
            client = google.cloud.logging.Client(project=project)
            self._logger = client.logger(log_name)
        except ImportError as exc:
            raise ImportError(
                "Install google-cloud-logging: pip install google-cloud-logging"
            ) from exc

    def emit(self, episode: Episode, extended: dict | None = None) -> None:
        record = {
            "run_id": episode.run_id,
            "query": episode.query,
            "confidence": episode.confidence,
            "drift_detected": episode.drift_detected,
            "reflex_mode": episode.reflex_mode,
            "delta_t_s": episode.delta_t,
            "t_detect_s": episode.t_detect,
            "t_decide_s": episode.t_decide,
            "t_execute_s": episode.t_execute,
        }
        if extended:
            record.update(extended)
        self._logger.log_struct(record)

    def flush(self) -> None:
        pass


# ── Composite (fan-out) ───────────────────────────────────────────────────────

class CompositeSink:
    """
    Fan-out sink that emits to multiple destinations in sequence.
    A failure in one sink is logged but does not block the others.
    """

    def __init__(self, *sinks: TelemetrySink) -> None:
        self._sinks = sinks

    def emit(self, episode: Episode, extended: dict | None = None) -> None:
        for sink in self._sinks:
            try:
                sink.emit(episode, extended)
            except Exception as exc:
                logger.warning(
                    "Sink %s failed on run_id=%d: %s",
                    type(sink).__name__, episode.run_id, exc,
                )

    def flush(self) -> None:
        for sink in self._sinks:
            try:
                sink.flush()
            except Exception as exc:
                logger.warning("Sink %s flush failed: %s", type(sink).__name__, exc)
