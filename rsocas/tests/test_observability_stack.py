"""Tests for ObservabilityStack — unit, wrap, hardening, and integration."""

from __future__ import annotations

import os
import time
import uuid

import pytest

from rsocas.contracts.traces import LeafTrace, NodeTrace, TreeTrace
from rsocas.development.orchestrator import ContinualLearningSystem, RunResult
from rsocas.development.stages import DevelopmentalController, DevelopmentalStage
from rsocas.observability.stack import ObservabilityStack, ObservedSystem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockExporter:
    """Records on_run / shutdown calls; optionally raises."""

    def __init__(self, should_fail: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.should_fail = should_fail
        self.shutdown_called = False

    def on_run(self, trace, result, status) -> None:
        if self.should_fail:
            raise RuntimeError("mock failure")
        self.calls.append((trace.trace_id, result.stage.name))

    def shutdown(self) -> None:
        if self.should_fail:
            raise RuntimeError("shutdown failure")
        self.shutdown_called = True


def _make_trace(
    task_type: str = "QA",
    final_output: str = "Paris",
    final_score: float | None = None,
    n_leaves: int = 3,
    depth: int = 1,
) -> TreeTrace:
    """Build a synthetic TreeTrace with configurable structure."""
    leaves = []
    leaf_traces = []
    for i in range(n_leaves):
        nid = f"leaf_{i}"
        leaves.append(NodeTrace(
            id=nid, depth=depth, position=i, combinator="LEAF",
            input_size=5000 + i * 1000,
            output=f"partial answer {i} with some details about the topic",
            children=(), llm_calls=1, latency_ms=500.0 + i * 100,
        ))
        leaf_traces.append(LeafTrace(
            node_id=nid,
            prompt=f"Given the following text about geography and history, answer: What is the capital? " * 20,
            response=f"partial answer {i} with some details about the topic",
            tokens_in=200 + i * 50, tokens_out=30 + i * 5, model="nemotron-3-super",
        ))

    root = NodeTrace(
        id="root", depth=0, position=0, combinator="REDUCE",
        input_size=sum(leaf.input_size for leaf in leaves),
        output=final_output,
        children=tuple(leaf.id for leaf in leaves),
        llm_calls=1, latency_ms=200.0,
    )
    all_nodes = tuple([root] + leaves)

    return TreeTrace(
        trace_id=uuid.uuid4().hex,
        task_type=task_type,
        k=n_leaves, depth=depth, tau=5000,
        cost_estimate=1.5,
        nodes=all_nodes,
        leaf_traces=tuple(leaf_traces),
        final_output=final_output,
        final_score=final_score,
        timestamp=time.time(),
        execution_time_ms=2000.0,
        total_llm_calls=n_leaves + 1,
        total_tokens=sum(lt.tokens_in + lt.tokens_out for lt in leaf_traces),
    )


def _build_system(
    stage: DevelopmentalStage = DevelopmentalStage.CHILDHOOD,
) -> ContinualLearningSystem:
    """Build a ContinualLearningSystem at the given stage with standard wiring."""
    from rsocas.archive.trace_archive import TraceArchive
    from rsocas.breathing.annealing import AnnealingSchedule
    from rsocas.breathing.tempo import PIDTempoController
    from rsocas.evaluation.boundary_detection import BoundaryDetectionEval
    from rsocas.evaluation.goodhart_resistant import GoodhartResistantEval
    from rsocas.evaluation.info_theoretic import InformationTheoreticEval

    evaluators = (
        InformationTheoreticEval(),
        BoundaryDetectionEval(),
        GoodhartResistantEval(),
    )
    dev = DevelopmentalController(stage)
    archive = TraceArchive(":memory:")
    tempo = PIDTempoController()
    annealing = AnnealingSchedule()

    return ContinualLearningSystem(
        evaluators=evaluators,
        development=dev,
        archive=archive,
        tempo=tempo,
        annealing=annealing,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestObservabilityStackUnit:
    """Unit tests for ObservabilityStack core behavior."""

    def test_empty_exporters_no_crash(self) -> None:
        stack = ObservabilityStack(exporters=[])
        system = _build_system()
        trace = _make_trace()
        result = system.run(trace)
        stack.observe(trace, result, system)  # should not raise

    def test_observe_calls_all_exporters(self) -> None:
        exporters = [MockExporter(), MockExporter(), MockExporter()]
        stack = ObservabilityStack(exporters=exporters)
        system = _build_system()
        trace = _make_trace()
        result = system.run(trace)
        stack.observe(trace, result, system)

        for exp in exporters:
            assert len(exp.calls) == 1
            assert exp.calls[0][0] == trace.trace_id

    def test_from_env_no_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RSOCAS_OTEL_ENDPOINT", raising=False)
        monkeypatch.delenv("RSOCAS_MLFLOW_URI", raising=False)
        monkeypatch.delenv("RSOCAS_METRICS_PORT", raising=False)
        stack = ObservabilityStack.from_env()
        assert len(stack._exporters) == 0

    def test_from_env_otel_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RSOCAS_OTEL_ENDPOINT", "http://localhost:4317")
        monkeypatch.delenv("RSOCAS_MLFLOW_URI", raising=False)
        monkeypatch.delenv("RSOCAS_METRICS_PORT", raising=False)
        stack = ObservabilityStack.from_env()
        assert len(stack._exporters) == 1

    def test_from_env_all_three(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RSOCAS_OTEL_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("RSOCAS_MLFLOW_URI", "http://localhost:5000")
        monkeypatch.setenv("RSOCAS_METRICS_PORT", "9090")
        stack = ObservabilityStack.from_env()
        assert len(stack._exporters) == 3

    def test_shutdown_calls_all(self) -> None:
        exporters = [MockExporter(), MockExporter(), MockExporter()]
        stack = ObservabilityStack(exporters=exporters)
        stack.shutdown()
        for exp in exporters:
            assert exp.shutdown_called


# ---------------------------------------------------------------------------
# Wrap tests
# ---------------------------------------------------------------------------


class TestObservedSystemWrap:
    """Tests for the ObservedSystem wrapper produced by stack.wrap()."""

    def test_wrap_produces_same_results(self) -> None:
        system = _build_system()
        stack = ObservabilityStack(exporters=[MockExporter()])
        wrapped = stack.wrap(system)
        trace = _make_trace()
        result = wrapped.run(trace)
        assert isinstance(result, RunResult)
        assert result.output == trace.final_output

    def test_wrap_emits_telemetry(self) -> None:
        exporter = MockExporter()
        system = _build_system()
        wrapped = ObservabilityStack(exporters=[exporter]).wrap(system)
        trace = _make_trace()
        wrapped.run(trace)
        assert len(exporter.calls) == 1
        assert exporter.calls[0][0] == trace.trace_id

    def test_wrap_preserves_human_feedback(self) -> None:
        from rsocas.breathing.annealing import AnnealingSchedule
        from rsocas.breathing.feedback_anchor import FeedbackAnchor
        from rsocas.breathing.tempo import PIDTempoController
        from rsocas.evaluation.boundary_detection import BoundaryDetectionEval
        from rsocas.evaluation.goodhart_resistant import GoodhartResistantEval
        from rsocas.evaluation.info_theoretic import InformationTheoreticEval

        dev = DevelopmentalController(DevelopmentalStage.BORN)
        tempo = PIDTempoController()
        annealing = AnnealingSchedule()
        feedback = FeedbackAnchor()
        system = ContinualLearningSystem(
            evaluators=(
                InformationTheoreticEval(),
                BoundaryDetectionEval(),
                GoodhartResistantEval(),
            ),
            development=dev,
            tempo=tempo,
            annealing=annealing,
            feedback_anchor=feedback,
        )
        wrapped = ObservabilityStack(exporters=[]).wrap(system)

        for _ in range(5):
            annealing.cool()
        temp_before = annealing.temperature
        wrapped.receive_human_feedback(time.time(), "correction")
        assert annealing.temperature > temp_before

    def test_wrap_preserves_status(self) -> None:
        system = _build_system()
        wrapped = ObservabilityStack(exporters=[]).wrap(system)
        wrapped.run(_make_trace())
        status = wrapped.status()
        assert status.total_runs == 1
        assert status.stage == DevelopmentalStage.CHILDHOOD


# ---------------------------------------------------------------------------
# Hardening tests
# ---------------------------------------------------------------------------


class TestObservabilityStackHardening:
    """Verify resilience when exporters fail."""

    def test_one_exporter_fails_others_still_called(self) -> None:
        good_1 = MockExporter()
        bad = MockExporter(should_fail=True)
        good_2 = MockExporter()
        stack = ObservabilityStack(exporters=[good_1, bad, good_2])
        system = _build_system()
        trace = _make_trace()
        result = system.run(trace)
        stack.observe(trace, result, system)

        assert len(good_1.calls) == 1
        assert len(bad.calls) == 0
        assert len(good_2.calls) == 1

    def test_all_exporters_fail_run_still_succeeds(self) -> None:
        exporters = [MockExporter(should_fail=True) for _ in range(3)]
        system = _build_system()
        wrapped = ObservabilityStack(exporters=exporters).wrap(system)
        trace = _make_trace()
        result = wrapped.run(trace)
        assert result.output == trace.final_output

    def test_shutdown_tolerates_failures(self) -> None:
        good = MockExporter()
        bad = MockExporter(should_fail=True)
        good_2 = MockExporter()
        stack = ObservabilityStack(exporters=[good, bad, good_2])
        stack.shutdown()  # should not raise
        assert good.shutdown_called
        assert good_2.shutdown_called

    def test_observe_with_none_disagreement(self) -> None:
        exporter = MockExporter()
        system = _build_system(stage=DevelopmentalStage.EMBRYONIC)
        wrapped = ObservabilityStack(exporters=[exporter]).wrap(system)
        trace = _make_trace()
        result = wrapped.run(trace)
        # embryonic stage: no evaluation, so disagreement is None
        assert result.disagreement is None
        assert len(exporter.calls) == 1


# ---------------------------------------------------------------------------
# Integration test — full pipeline with observability
# ---------------------------------------------------------------------------


class TestFullPipelineWithObservability:
    """Mirror TestFullPipeline but with ObservabilityStack wrapping the system."""

    def test_full_pipeline_10_runs_with_observability(self) -> None:
        from rsocas.archive.trace_archive import TraceArchive
        from rsocas.breathing.annealing import AnnealingSchedule
        from rsocas.breathing.feedback_anchor import FeedbackAnchor
        from rsocas.breathing.interference import InterferencePattern
        from rsocas.breathing.tempo import PIDTempoController
        from rsocas.evaluation.boundary_detection import BoundaryDetectionEval
        from rsocas.evaluation.goodhart_resistant import GoodhartResistantEval
        from rsocas.evaluation.info_theoretic import InformationTheoreticEval

        archive = TraceArchive(":memory:")
        tempo = PIDTempoController()
        annealing = AnnealingSchedule(t_init=1.0, t_min=0.01)
        feedback = FeedbackAnchor()
        interference = InterferencePattern()
        evaluators = (
            InformationTheoreticEval(),
            BoundaryDetectionEval(),
            GoodhartResistantEval(),
        )
        dev = DevelopmentalController(DevelopmentalStage.CHILDHOOD)
        system = ContinualLearningSystem(
            evaluators=evaluators,
            development=dev,
            tempo=tempo,
            annealing=annealing,
            archive=archive,
            interference=interference,
            feedback_anchor=feedback,
        )

        # Three mock exporters
        exporters = [MockExporter(), MockExporter(), MockExporter()]
        stack = ObservabilityStack(exporters=exporters)
        wrapped = stack.wrap(system)

        all_results = []
        for i in range(10):
            score = 0.3 + (i % 3) * 0.25
            trace = _make_trace(
                final_output=f"answer_{i}",
                final_score=score,
                n_leaves=2 + (i % 3),
            )
            result = wrapped.run(trace)
            all_results.append(result)

        # Each exporter received exactly 10 calls
        for exp in exporters:
            assert len(exp.calls) == 10

        # Archive has all 10 traces
        assert archive.count() == 10

        # All results have evaluations and disagreement (CHILDHOOD stage)
        assert all(r.evaluations is not None for r in all_results)
        assert all(r.disagreement is not None for r in all_results)

        # System progressed developmentally
        status = wrapped.status()
        assert status.total_runs == 10
        assert status.stage.value >= DevelopmentalStage.CHILDHOOD.value

        # Shutdown works
        stack.shutdown()
        for exp in exporters:
            assert exp.shutdown_called
