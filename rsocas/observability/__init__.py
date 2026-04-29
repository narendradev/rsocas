"""Observability stack for RSOCAS — OTel, MLflow, Prometheus.

All backends are optional. Missing libraries degrade gracefully.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rsocas.contracts.traces import TreeTrace
    from rsocas.development.orchestrator import RunResult, SystemStatus


class ObservabilityExporter(Protocol):
    def on_run(self, trace: "TreeTrace", result: "RunResult", status: "SystemStatus") -> None: ...
    def shutdown(self) -> None: ...


from rsocas.observability.stack import ObservabilityStack, ObservedSystem

__all__ = ["ObservabilityExporter", "ObservabilityStack", "ObservedSystem"]
