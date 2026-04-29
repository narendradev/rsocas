"""Tests for the OpenTelemetry exporter."""

from __future__ import annotations

import time

import pytest

from rsocas.contracts.evaluation import DisagreementSignal, EvalResult
from rsocas.contracts.traces import LeafTrace, NodeTrace, TreeTrace
from rsocas.development.orchestrator import RunResult, SystemStatus
from rsocas.development.stages import DevelopmentalStage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(
    id: str,
    depth: int = 0,
    position: int = 0,
    combinator: str = "MAP",
    children: tuple[str, ...] = (),
    llm_calls: int = 1,
    score: float | None = None,
) -> NodeTrace:
    return NodeTrace(
        id=id,
        depth=depth,
        position=position,
        combinator=combinator,
        input_size=10,
        output="out",
        children=children,
        llm_calls=llm_calls,
        score=score,
    )


def _leaf(node_id: str, model: str = "test-model") -> LeafTrace:
    return LeafTrace(
        node_id=node_id,
        prompt="p",
        response="r",
        tokens_in=5,
        tokens_out=10,
        model=model,
        confidence=0.9,
    )


def _trace(
    nodes: tuple[NodeTrace, ...] = (),
    leaf_traces: tuple[LeafTrace, ...] = (),
) -> TreeTrace:
    return TreeTrace(
        trace_id="t-1",
        task_type="qa",
        k=3,
        depth=2,
        tau=5,
        cost_estimate=0.01,
        nodes=nodes,
        leaf_traces=leaf_traces,
        final_output="answer",
        timestamp=time.time(),
        execution_time_ms=100.0,
        total_llm_calls=len(nodes),
        total_tokens=100,
    )


def _result(
    trace: TreeTrace,
    evaluations: tuple[EvalResult, ...] | None = None,
    disagreement: DisagreementSignal | None = None,
) -> RunResult:
    return RunResult(
        output=trace.final_output,
        trace=trace,
        evaluations=evaluations,
        disagreement=disagreement,
        stage=DevelopmentalStage.FETAL,
        surfaced_for_human=False,
    )


def _status() -> SystemStatus:
    return SystemStatus(
        stage=DevelopmentalStage.FETAL,
        enabled_features=frozenset({"execution", "evaluation"}),
        total_runs=1,
        archive_size=0,
        active_combinators=0,
        breathing_rate=None,
        temperature=None,
    )


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

def test_disabled_when_otel_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import rsocas.observability.otel as otel_mod

    monkeypatch.setattr(otel_mod, "_OTEL_AVAILABLE", False)
    exporter = otel_mod.OtelExporter()
    assert not exporter._enabled

    trace = _trace(nodes=(_node("a"),))
    result = _result(trace)
    exporter.on_run(trace, result, _status())  # no crash
    exporter.shutdown()  # no crash


def test_on_run_never_crashes() -> None:
    """Pass garbage data; verify no exception bubbles up."""
    otel_mod = pytest.importorskip("rsocas.observability.otel")
    otel_sdk = pytest.importorskip("opentelemetry.sdk.trace.export")

    mem = otel_sdk.export.InMemorySpanExporter() if hasattr(otel_sdk, "export") else None
    # InMemorySpanExporter lives in opentelemetry.sdk.trace.export
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    mem = InMemorySpanExporter()
    exporter = otel_mod.OtelExporter(_test_exporter=mem)

    # completely wrong types
    exporter.on_run("not-a-trace", "not-a-result", "not-a-status")  # type: ignore[arg-type]
    exporter.shutdown()


# ---------------------------------------------------------------------------
# Span creation tests (require OTel)
# ---------------------------------------------------------------------------

@pytest.fixture()
def otel_exporter():
    """Yield an OtelExporter backed by InMemorySpanExporter."""
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from rsocas.observability.otel import OtelExporter

    mem = InMemorySpanExporter()
    exp = OtelExporter(_test_exporter=mem)
    yield exp, mem
    exp.shutdown()


def test_single_leaf_trace(otel_exporter) -> None:
    exp, mem = otel_exporter
    n = _node("a")
    trace = _trace(nodes=(n,))
    result = _result(trace)

    exp.on_run(trace, result, _status())
    exp.shutdown()

    spans = mem.get_finished_spans()
    # root span + 1 node span
    assert len(spans) == 2
    names = {s.name for s in spans}
    assert "rsocas.run" in names
    assert "node.MAP" in names


def test_tree_3_leaves(otel_exporter) -> None:
    exp, mem = otel_exporter
    root = _node("root", children=("c1", "c2", "c3"))
    c1 = _node("c1", depth=1, position=0, combinator="SPLIT")
    c2 = _node("c2", depth=1, position=1, combinator="FILTER")
    c3 = _node("c3", depth=1, position=2, combinator="REDUCE")
    trace = _trace(nodes=(root, c1, c2, c3))
    result = _result(trace)

    exp.on_run(trace, result, _status())
    exp.shutdown()

    spans = mem.get_finished_spans()
    # root span + 4 node spans
    assert len(spans) == 5

    # Verify parent-child: c1/c2/c3 parent should be node.MAP (the root node span)
    root_node_span = next(s for s in spans if s.name == "node.MAP")
    child_spans = [s for s in spans if s.name in ("node.SPLIT", "node.FILTER", "node.REDUCE")]
    for cs in child_spans:
        assert cs.parent is not None
        assert cs.parent.span_id == root_node_span.context.span_id


def test_leaf_traces_as_events(otel_exporter) -> None:
    exp, mem = otel_exporter
    n = _node("leaf1", combinator="CALL")
    lt = _leaf("leaf1", model="gpt-4")
    trace = _trace(nodes=(n,), leaf_traces=(lt,))
    result = _result(trace)

    exp.on_run(trace, result, _status())
    exp.shutdown()

    spans = mem.get_finished_spans()
    node_span = next(s for s in spans if s.name == "node.CALL")
    events = node_span.events
    assert len(events) == 1
    assert events[0].name == "leaf_trace"
    assert events[0].attributes["rsocas.leaf.model"] == "gpt-4"
    assert events[0].attributes["rsocas.leaf.tokens_in"] == 5


def test_eval_results_as_events(otel_exporter) -> None:
    exp, mem = otel_exporter
    n = _node("a")
    trace = _trace(nodes=(n,))
    ev = EvalResult(score=0.85, confidence=0.9, signal_type="semantic")
    result = _result(trace, evaluations=(ev,))

    exp.on_run(trace, result, _status())
    exp.shutdown()

    spans = mem.get_finished_spans()
    root_span = next(s for s in spans if s.name == "rsocas.run")
    eval_events = [e for e in root_span.events if e.name == "eval_result"]
    assert len(eval_events) == 1
    assert eval_events[0].attributes["rsocas.eval.score"] == 0.85
    assert eval_events[0].attributes["rsocas.eval.signal_type"] == "semantic"


def test_disagreement_as_attributes(otel_exporter) -> None:
    exp, mem = otel_exporter
    n = _node("a")
    trace = _trace(nodes=(n,))
    dis = DisagreementSignal(
        magnitude=0.42,
        should_surface=True,
        outlier_voice="semantic",
    )
    result = _result(trace, disagreement=dis)

    exp.on_run(trace, result, _status())
    exp.shutdown()

    spans = mem.get_finished_spans()
    root_span = next(s for s in spans if s.name == "rsocas.run")
    assert root_span.attributes["rsocas.disagreement.magnitude"] == 0.42
    assert root_span.attributes["rsocas.disagreement.should_surface"] is True
    assert root_span.attributes["rsocas.disagreement.outlier_voice"] == "semantic"


def test_empty_trace(otel_exporter) -> None:
    exp, mem = otel_exporter
    trace = _trace(nodes=(), leaf_traces=())
    result = _result(trace)

    exp.on_run(trace, result, _status())  # no crash
    exp.shutdown()

    spans = mem.get_finished_spans()
    assert len(spans) == 1  # just root span
    assert spans[0].name == "rsocas.run"


def test_shutdown_flushes(otel_exporter) -> None:
    exp, mem = otel_exporter
    trace = _trace(nodes=(_node("x"),))
    result = _result(trace)

    exp.on_run(trace, result, _status())
    exp.shutdown()

    # After shutdown, spans should be flushed
    assert len(mem.get_finished_spans()) > 0
