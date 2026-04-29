"""Tests for Prometheus metrics exporter.

Every test uses a fresh CollectorRegistry -- no global state pollution.
No HTTP server is started.
"""

from __future__ import annotations

import time
import uuid

import pytest
from prometheus_client import CollectorRegistry

from rsocas.contracts.evaluation import DisagreementSignal
from rsocas.contracts.traces import LeafTrace, NodeTrace, TreeTrace
from rsocas.development.orchestrator import RunResult, SystemStatus
from rsocas.development.stages import DevelopmentalStage
from rsocas.observability.metrics import PrometheusMetrics, _PROM_AVAILABLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace(
    task_type: str = "QA",
    total_tokens: int = 1000,
    execution_time_ms: float = 2000.0,
) -> TreeTrace:
    """Build a minimal synthetic TreeTrace."""
    leaf = NodeTrace(
        id="leaf_0",
        depth=1,
        position=0,
        combinator="LEAF",
        input_size=5000,
        output="answer",
        llm_calls=1,
        latency_ms=500.0,
    )
    root = NodeTrace(
        id="root",
        depth=0,
        position=0,
        combinator="REDUCE",
        input_size=5000,
        output="answer",
        children=("leaf_0",),
        llm_calls=1,
        latency_ms=200.0,
    )
    return TreeTrace(
        trace_id=uuid.uuid4().hex,
        task_type=task_type,
        k=1,
        depth=1,
        tau=5000,
        cost_estimate=1.0,
        nodes=(root, leaf),
        leaf_traces=(
            LeafTrace(
                node_id="leaf_0",
                prompt="question",
                response="answer",
                tokens_in=500,
                tokens_out=500,
            ),
        ),
        final_output="answer",
        timestamp=time.time(),
        execution_time_ms=execution_time_ms,
        total_llm_calls=2,
        total_tokens=total_tokens,
    )


def _make_result(
    stage: DevelopmentalStage = DevelopmentalStage.FETAL,
    disagreement_magnitude: float | None = 0.5,
    surfaced: bool = False,
) -> RunResult:
    """Build a minimal RunResult."""
    disagreement = None
    if disagreement_magnitude is not None:
        disagreement = DisagreementSignal(
            magnitude=disagreement_magnitude,
            timestamp=time.time(),
        )
    return RunResult(
        output="answer",
        trace=None,
        evaluations=None,
        disagreement=disagreement,
        stage=stage,
        surfaced_for_human=surfaced,
    )


def _make_status(
    stage: DevelopmentalStage = DevelopmentalStage.FETAL,
    temperature: float | None = 0.8,
    breathing_rate: float | None = 1.5,
    archive_size: int = 42,
    active_combinators: int = 7,
) -> SystemStatus:
    """Build a minimal SystemStatus."""
    return SystemStatus(
        stage=stage,
        enabled_features=frozenset({"execution", "evaluation"}),
        total_runs=1,
        archive_size=archive_size,
        active_combinators=active_combinators,
        breathing_rate=breathing_rate,
        temperature=temperature,
    )


def _fresh_metrics() -> PrometheusMetrics:
    """Create a PrometheusMetrics with a fresh registry (no port)."""
    return PrometheusMetrics(registry=CollectorRegistry())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPrometheusMetrics:
    def test_disabled_when_not_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import rsocas.observability.metrics as mod

        monkeypatch.setattr(mod, "_PROM_AVAILABLE", False)
        m = PrometheusMetrics()
        assert not m._enabled
        # Should not raise
        m.on_run(_make_trace(), _make_result(), _make_status())

    def test_requests_counter_increments(self) -> None:
        m = _fresh_metrics()
        m.on_run(_make_trace(), _make_result(), _make_status())
        assert m.requests_total.labels(task_type="QA")._value.get() == 1.0

    def test_requests_counter_by_task_type(self) -> None:
        m = _fresh_metrics()
        m.on_run(_make_trace(task_type="QA"), _make_result(), _make_status())
        m.on_run(_make_trace(task_type="QA"), _make_result(), _make_status())
        m.on_run(_make_trace(task_type="SUMMARY"), _make_result(), _make_status())
        assert m.requests_total.labels(task_type="QA")._value.get() == 2.0
        assert m.requests_total.labels(task_type="SUMMARY")._value.get() == 1.0

    def test_tokens_counter(self) -> None:
        m = _fresh_metrics()
        m.on_run(_make_trace(total_tokens=500), _make_result(), _make_status())
        m.on_run(_make_trace(total_tokens=300), _make_result(), _make_status())
        assert m.tokens_total._value.get() == 800.0

    def test_surfacing_counter(self) -> None:
        m = _fresh_metrics()
        m.on_run(_make_trace(), _make_result(surfaced=True), _make_status())
        assert m.surfacing_total._value.get() == 1.0

    def test_no_surfacing_counter(self) -> None:
        m = _fresh_metrics()
        m.on_run(_make_trace(), _make_result(surfaced=False), _make_status())
        assert m.surfacing_total._value.get() == 0.0

    def test_disagreement_histogram(self) -> None:
        m = _fresh_metrics()
        m.on_run(
            _make_trace(),
            _make_result(disagreement_magnitude=0.75),
            _make_status(),
        )
        # Histogram _sum tracks the sum of observed values
        assert m.disagreement._sum.get() == 0.75

    def test_latency_converts_ms_to_seconds(self) -> None:
        m = _fresh_metrics()
        m.on_run(
            _make_trace(execution_time_ms=3000.0),
            _make_result(),
            _make_status(),
        )
        assert m.latency._sum.get() == 3.0

    def test_stage_gauge(self) -> None:
        m = _fresh_metrics()
        m.on_run(
            _make_trace(),
            _make_result(stage=DevelopmentalStage.CHILDHOOD),
            _make_status(),
        )
        assert m.stage._value.get() == DevelopmentalStage.CHILDHOOD.value

    def test_temperature_gauge(self) -> None:
        m = _fresh_metrics()
        m.on_run(_make_trace(), _make_result(), _make_status(temperature=0.42))
        assert m.temperature._value.get() == 0.42

    def test_temperature_null_sets_zero(self) -> None:
        m = _fresh_metrics()
        m.on_run(_make_trace(), _make_result(), _make_status(temperature=None))
        assert m.temperature._value.get() == 0.0

    def test_breathing_rate_gauge(self) -> None:
        m = _fresh_metrics()
        m.on_run(_make_trace(), _make_result(), _make_status(breathing_rate=2.5))
        assert m.breathing_rate_gauge._value.get() == 2.5

    def test_archive_size_gauge(self) -> None:
        m = _fresh_metrics()
        m.on_run(_make_trace(), _make_result(), _make_status(archive_size=99))
        assert m.archive_size_gauge._value.get() == 99.0

    def test_multiple_runs_accumulate(self) -> None:
        m = _fresh_metrics()
        for _ in range(5):
            m.on_run(_make_trace(), _make_result(), _make_status())
        assert m.requests_total.labels(task_type="QA")._value.get() == 5.0

    def test_on_run_never_crashes(self) -> None:
        m = _fresh_metrics()
        # Pass completely wrong types -- should not raise
        m.on_run(object(), object(), object())  # type: ignore[arg-type]
