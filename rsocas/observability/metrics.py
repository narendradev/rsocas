"""Prometheus metrics exporter for RSOCAS.

Gracefully degrades when prometheus_client is not installed.
Uses a custom CollectorRegistry to avoid global state pollution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rsocas.contracts.traces import TreeTrace
    from rsocas.development.orchestrator import RunResult, SystemStatus

_PROM_AVAILABLE = False
try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )

    _PROM_AVAILABLE = True
except ImportError:
    pass


class PrometheusMetrics:
    """Prometheus metrics exporter implementing ObservabilityExporter protocol."""

    def __init__(
        self,
        namespace: str = "rsocas",
        registry: object | None = None,
        port: int | None = None,
    ) -> None:
        self._enabled = False
        if not _PROM_AVAILABLE:
            return

        self._registry = registry or CollectorRegistry()
        self._enabled = True

        # Counters
        self.requests_total = Counter(
            f"{namespace}_requests_total",
            "Total requests",
            ["task_type"],
            registry=self._registry,
        )
        self.tokens_total = Counter(
            f"{namespace}_tokens_total",
            "Total tokens",
            registry=self._registry,
        )
        self.surfacing_total = Counter(
            f"{namespace}_surfacing_events_total",
            "Surfacing events",
            registry=self._registry,
        )

        # Histograms
        self.disagreement = Histogram(
            f"{namespace}_disagreement_magnitude",
            "Disagreement magnitude",
            buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
            registry=self._registry,
        )
        self.latency = Histogram(
            f"{namespace}_latency_seconds",
            "Latency in seconds",
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
            registry=self._registry,
        )
        self.tokens_per_req = Histogram(
            f"{namespace}_tokens_per_request",
            "Tokens per request",
            buckets=[100, 500, 1000, 5000, 10000, 50000],
            registry=self._registry,
        )

        # Gauges
        self.stage = Gauge(
            f"{namespace}_developmental_stage",
            "Developmental stage value",
            registry=self._registry,
        )
        self.temperature = Gauge(
            f"{namespace}_annealing_temperature",
            "Annealing temperature",
            registry=self._registry,
        )
        self.breathing_rate_gauge = Gauge(
            f"{namespace}_breathing_rate",
            "Breathing rate",
            registry=self._registry,
        )
        self.archive_size_gauge = Gauge(
            f"{namespace}_archive_size",
            "Archive size",
            registry=self._registry,
        )
        self.active_combinators_gauge = Gauge(
            f"{namespace}_active_combinators",
            "Active combinators count",
            registry=self._registry,
        )

        if port is not None:
            start_http_server(port, registry=self._registry)

    def on_run(
        self,
        trace: TreeTrace,
        result: RunResult,
        status: SystemStatus,
    ) -> None:
        """Single entry point for recording metrics from a run."""
        if not self._enabled:
            return
        try:
            self._record(trace, result, status)
        except Exception:
            pass

    def _record(
        self,
        trace: TreeTrace,
        result: RunResult,
        status: SystemStatus,
    ) -> None:
        self.requests_total.labels(task_type=trace.task_type).inc()
        self.tokens_total.inc(trace.total_tokens)
        self.latency.observe(trace.execution_time_ms / 1000.0)
        self.tokens_per_req.observe(trace.total_tokens)

        if result.disagreement:
            self.disagreement.observe(result.disagreement.magnitude)
        if result.surfaced_for_human:
            self.surfacing_total.inc()

        self.stage.set(result.stage.value)
        self.temperature.set(
            status.temperature if status.temperature is not None else 0,
        )
        self.breathing_rate_gauge.set(
            status.breathing_rate if status.breathing_rate is not None else 0,
        )
        self.archive_size_gauge.set(status.archive_size)
        self.active_combinators_gauge.set(status.active_combinators)

    def shutdown(self) -> None:
        """No-op shutdown for protocol compliance."""
