"""OpenTelemetry exporter for RSOCAS traces.

Converts TreeTrace / RunResult / SystemStatus into OpenTelemetry spans.
Gracefully degrades when opentelemetry is not installed.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider as _TracerProvider
    from opentelemetry.sdk.trace.export import SpanExporter as _SpanExporter

    from rsocas.contracts.evaluation import DisagreementSignal, EvalResult
    from rsocas.contracts.traces import LeafTrace, NodeTrace, TreeTrace
    from rsocas.development.orchestrator import RunResult, SystemStatus

_log = logging.getLogger(__name__)

_OTEL_AVAILABLE = False
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    _OTEL_AVAILABLE = True
except ImportError:
    pass


def _safe_attrs(mapping: dict) -> dict[str, str | int | float | bool]:
    """Filter dict to OTel-safe attribute values, drop None."""
    return {k: v for k, v in mapping.items() if v is not None}


class OtelExporter:
    """Exports RSOCAS data as OpenTelemetry spans."""

    def __init__(
        self,
        service_name: str = "rsocas",
        exporter_type: str = "console",
        endpoint: str | None = None,
        _test_exporter: _SpanExporter | None = None,
    ) -> None:
        self._enabled = False
        if not _OTEL_AVAILABLE:
            return

        provider = TracerProvider()

        if _test_exporter is not None:
            provider.add_span_processor(SimpleSpanProcessor(_test_exporter))
        elif exporter_type == "console":
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        elif exporter_type in ("otlp_grpc", "otlp_http"):
            span_exporter = _build_otlp_exporter(exporter_type, endpoint)
            if span_exporter is not None:
                provider.add_span_processor(BatchSpanProcessor(span_exporter))
            else:
                _log.warning("OTLP exporter unavailable; OTel disabled")
                return

        self._provider: _TracerProvider = provider
        self._tracer = provider.get_tracer("rsocas", "0.1.0")
        self._enabled = True

    def on_run(
        self,
        trace: TreeTrace,
        result: RunResult,
        status: SystemStatus,
    ) -> None:
        if not self._enabled:
            return
        try:
            self._export(trace, result)
        except Exception:
            _log.debug("OTel export failed", exc_info=True)

    def shutdown(self) -> None:
        if not self._enabled:
            return
        self._provider.force_flush()
        self._provider.shutdown()

    def _export(self, trace: TreeTrace, result: RunResult) -> None:
        node_map: dict[str, NodeTrace] = {n.id: n for n in trace.nodes}
        child_ids: set[str] = set()
        for node in trace.nodes:
            child_ids.update(node.children)
        root_ids = [n.id for n in trace.nodes if n.id not in child_ids]

        leaf_by_node: dict[str, list[LeafTrace]] = {}
        for lt in trace.leaf_traces:
            leaf_by_node.setdefault(lt.node_id, []).append(lt)

        with self._tracer.start_as_current_span("rsocas.run") as root_span:
            root_span.set_attributes(_safe_attrs({
                "rsocas.trace_id": trace.trace_id,
                "rsocas.task_type": trace.task_type,
                "rsocas.k": trace.k,
                "rsocas.depth": trace.depth,
                "rsocas.tau": trace.tau,
                "rsocas.total_tokens": trace.total_tokens,
                "rsocas.stage": result.stage.name,
                "rsocas.execution_time_ms": trace.execution_time_ms,
            }))

            for rid in root_ids:
                self._walk(rid, node_map, leaf_by_node)

            self._attach_evals(root_span, result)
            self._attach_disagreement(root_span, result)

    def _walk(
        self,
        node_id: str,
        node_map: dict[str, NodeTrace],
        leaf_by_node: dict[str, list[LeafTrace]],
    ) -> None:
        node = node_map.get(node_id)
        if node is None:
            return

        with self._tracer.start_as_current_span(f"node.{node.combinator}") as span:
            span.set_attributes(_safe_attrs({
                "rsocas.node_id": node.id,
                "rsocas.combinator": node.combinator,
                "rsocas.input_size": node.input_size,
                "rsocas.llm_calls": node.llm_calls,
                "rsocas.position": node.position,
                "rsocas.depth": node.depth,
                "rsocas.latency_ms": node.latency_ms,
                "rsocas.score": node.score,
            }))

            for lt in leaf_by_node.get(node.id, ()):
                span.add_event("leaf_trace", attributes=_safe_attrs({
                    "rsocas.leaf.model": lt.model,
                    "rsocas.leaf.tokens_in": lt.tokens_in,
                    "rsocas.leaf.tokens_out": lt.tokens_out,
                    "rsocas.leaf.confidence": lt.confidence,
                }))

            for child_id in node.children:
                self._walk(child_id, node_map, leaf_by_node)

    @staticmethod
    def _attach_evals(span: otel_trace.Span, result: RunResult) -> None:
        if not result.evaluations:
            return
        for ev in result.evaluations:
            span.add_event("eval_result", attributes=_safe_attrs({
                "rsocas.eval.score": ev.score,
                "rsocas.eval.confidence": ev.confidence,
                "rsocas.eval.signal_type": ev.signal_type,
            }))

    @staticmethod
    def _attach_disagreement(span: otel_trace.Span, result: RunResult) -> None:
        if not result.disagreement:
            return
        d = result.disagreement
        span.set_attributes(_safe_attrs({
            "rsocas.disagreement.magnitude": d.magnitude,
            "rsocas.disagreement.should_surface": d.should_surface,
            "rsocas.disagreement.outlier_voice": d.outlier_voice,
        }))


def _build_otlp_exporter(
    exporter_type: str,
    endpoint: str | None,
) -> _SpanExporter | None:
    """Attempt to build an OTLP exporter; return None on import failure."""
    try:
        if exporter_type == "otlp_grpc":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        kwargs = {}
        if endpoint is not None:
            kwargs["endpoint"] = endpoint
        return OTLPSpanExporter(**kwargs)
    except ImportError:
        return None
