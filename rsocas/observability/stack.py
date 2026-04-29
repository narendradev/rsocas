"""ObservabilityStack — wires OTel + MLflow + Prometheus exporters.

Composition-based design: accepts any list of ObservabilityExporter instances.
Adding a new backend requires zero changes to this module.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rsocas.contracts.traces import TreeTrace
    from rsocas.development.orchestrator import ContinualLearningSystem, RunResult

logger = logging.getLogger(__name__)


class ObservabilityStack:
    """Wires all observability exporters together via the uniform list pattern."""

    def __init__(self, exporters: list | None = None) -> None:
        self._exporters = exporters or []

    @classmethod
    def from_env(cls) -> ObservabilityStack:
        """Build stack from environment variables. Missing vars = backend disabled."""
        exporters: list = []

        otel_endpoint = os.environ.get("RSOCAS_OTEL_ENDPOINT")
        if otel_endpoint:
            from rsocas.observability.otel import OtelExporter

            exporters.append(OtelExporter(
                endpoint=otel_endpoint,
                exporter_type=os.environ.get("RSOCAS_OTEL_EXPORTER", "otlp_grpc"),
            ))

        mlflow_uri = os.environ.get("RSOCAS_MLFLOW_URI")
        if mlflow_uri:
            from rsocas.observability.mlflow_logger import MlflowLogger

            exporters.append(MlflowLogger(tracking_uri=mlflow_uri))

        metrics_port = os.environ.get("RSOCAS_METRICS_PORT")
        if metrics_port:
            from rsocas.observability.metrics import PrometheusMetrics

            exporters.append(PrometheusMetrics(port=int(metrics_port)))

        return cls(exporters=exporters)

    def observe(
        self,
        trace: TreeTrace,
        result: RunResult,
        system: ContinualLearningSystem,
    ) -> None:
        """Emit telemetry for a completed run. Never crashes."""
        status = system.status()
        for exporter in self._exporters:
            try:
                exporter.on_run(trace, result, status)
            except Exception:
                logger.warning(
                    "Exporter %s failed", type(exporter).__name__, exc_info=True,
                )

    def wrap(self, system: ContinualLearningSystem) -> ObservedSystem:
        """Return a wrapper that auto-emits telemetry on each run()."""
        return ObservedSystem(system, self)

    def shutdown(self) -> None:
        """Flush and close all backends. Tolerates failures."""
        for exporter in self._exporters:
            try:
                exporter.shutdown()
            except Exception:
                logger.warning(
                    "Shutdown failed for %s",
                    type(exporter).__name__,
                    exc_info=True,
                )


class ObservedSystem:
    """Proxy around ContinualLearningSystem that emits telemetry on each run()."""

    def __init__(
        self,
        inner: ContinualLearningSystem,
        stack: ObservabilityStack,
    ) -> None:
        self._inner = inner
        self._stack = stack

    def run(self, trace: TreeTrace) -> RunResult:
        result = self._inner.run(trace)
        self._stack.observe(trace, result, self._inner)
        return result

    def receive_human_feedback(
        self, timestamp: float, feedback_type: str = "general",
    ) -> None:
        self._inner.receive_human_feedback(timestamp, feedback_type)

    def status(self):
        return self._inner.status()
