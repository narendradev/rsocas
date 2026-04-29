"""Tests for developmental stages and the ContinualLearningSystem orchestrator."""

from __future__ import annotations

import time

import pytest

from rsocas.archive.trace_archive import TraceArchive
from rsocas.breathing.annealing import AnnealingSchedule
from rsocas.breathing.feedback_anchor import FeedbackAnchor
from rsocas.breathing.interference import InterferencePattern
from rsocas.breathing.tempo import PIDTempoController
from rsocas.contracts.evaluation import DisagreementSignal, EvalResult, Evaluator
from rsocas.contracts.traces import LeafTrace, NodeTrace, TreeTrace
from rsocas.development.orchestrator import (
    ContinualLearningSystem,
    RunResult,
    SystemStatus,
)
from rsocas.development.stages import (
    DevelopmentalController,
    DevelopmentalMetrics,
    DevelopmentalStage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_test_trace(
    trace_id: str = "trace-001",
    final_output: str = "test output",
    timestamp: float | None = None,
) -> TreeTrace:
    """Build a minimal valid TreeTrace for testing."""
    ts = timestamp if timestamp is not None else time.time()
    node = NodeTrace(
        id="node-0",
        depth=0,
        position=0,
        combinator="SPLIT",
        input_size=10,
        output="node output",
    )
    leaf = LeafTrace(
        node_id="node-0",
        prompt="test prompt",
        response="test response",
    )
    return TreeTrace(
        trace_id=trace_id,
        task_type="qa",
        k=3,
        depth=1,
        tau=1,
        cost_estimate=0.01,
        nodes=(node,),
        leaf_traces=(leaf,),
        final_output=final_output,
        timestamp=ts,
        execution_time_ms=100.0,
    )


class _StubEvaluator:
    """A simple evaluator that returns a fixed score."""

    def __init__(self, score: float, signal_type: str) -> None:
        self._score = score
        self._signal_type = signal_type

    @property
    def signal_type(self) -> str:
        return self._signal_type

    def evaluate(
        self,
        trace: TreeTrace,
        ground_truth: str | None = None,
    ) -> EvalResult:
        return EvalResult(
            score=self._score,
            confidence=0.9,
            signal_type=self._signal_type,
            per_node_scores={},
            explanation=f"Stub eval: {self._signal_type}",
        )


# ---------------------------------------------------------------------------
# Developmental stages tests
# ---------------------------------------------------------------------------


class TestDevelopmentalStages:
    """Tests for DevelopmentalStage enum and DevelopmentalController."""

    def test_stage_ordering(self) -> None:
        """Stages are ordered from 0 (EMBRYONIC) to 5 (ADULT)."""
        assert DevelopmentalStage.EMBRYONIC < DevelopmentalStage.FETAL
        assert DevelopmentalStage.FETAL < DevelopmentalStage.BORN
        assert DevelopmentalStage.BORN < DevelopmentalStage.CHILDHOOD
        assert DevelopmentalStage.CHILDHOOD < DevelopmentalStage.ADOLESCENCE
        assert DevelopmentalStage.ADOLESCENCE < DevelopmentalStage.ADULT

    def test_enabled_features_per_stage(self) -> None:
        """Verify get_enabled_features for each stage."""
        ctrl = DevelopmentalController()

        ctrl.force_transition(DevelopmentalStage.EMBRYONIC, 0.0)
        assert ctrl.get_enabled_features() == {"execution"}

        ctrl.force_transition(DevelopmentalStage.FETAL, 1.0)
        assert ctrl.get_enabled_features() == {"execution", "evaluation"}

        ctrl.force_transition(DevelopmentalStage.BORN, 2.0)
        assert ctrl.get_enabled_features() == {
            "execution", "evaluation", "breathing",
        }

        ctrl.force_transition(DevelopmentalStage.CHILDHOOD, 3.0)
        assert ctrl.get_enabled_features() == {
            "execution", "evaluation", "breathing", "archive", "penumbra",
        }

        ctrl.force_transition(DevelopmentalStage.ADOLESCENCE, 4.0)
        assert ctrl.get_enabled_features() == {
            "execution", "evaluation", "breathing", "archive", "penumbra",
            "optimization",
        }

        ctrl.force_transition(DevelopmentalStage.ADULT, 5.0)
        assert ctrl.get_enabled_features() == {
            "execution", "evaluation", "breathing", "archive", "penumbra",
            "optimization", "autonomy",
        }

    def test_developmental_transition_embryonic_to_fetal(self) -> None:
        """EMBRYONIC -> FETAL transition is always allowed."""
        ctrl = DevelopmentalController()
        metrics = DevelopmentalMetrics(
            total_traces=1,
            consecutive_traces_with_eval=0,
            successful_dissolutions=0,
            archive_size=0,
            disagreement_correlation=None,
        )
        result = ctrl.check_transition(metrics, 1.0)
        assert result == DevelopmentalStage.FETAL
        assert ctrl.current_stage == DevelopmentalStage.FETAL

    def test_developmental_transition_is_irreversible(self) -> None:
        """After transitioning to FETAL, cannot go back to EMBRYONIC via check_transition."""
        ctrl = DevelopmentalController()
        metrics = DevelopmentalMetrics(
            total_traces=1,
            consecutive_traces_with_eval=0,
            successful_dissolutions=0,
            archive_size=0,
            disagreement_correlation=None,
        )
        ctrl.check_transition(metrics, 1.0)
        assert ctrl.current_stage == DevelopmentalStage.FETAL

        # check_transition only moves forward -- cannot go back
        # Running check_transition again should try FETAL -> BORN,
        # which requires consecutive_traces_with_eval >= 100
        result = ctrl.check_transition(metrics, 2.0)
        assert result is None
        assert ctrl.current_stage == DevelopmentalStage.FETAL

    def test_fetal_to_born_threshold(self) -> None:
        """FETAL -> BORN requires consecutive_traces_with_eval >= 100."""
        ctrl = DevelopmentalController(DevelopmentalStage.FETAL)

        # Not enough traces
        metrics_low = DevelopmentalMetrics(
            total_traces=50,
            consecutive_traces_with_eval=99,
            successful_dissolutions=0,
            archive_size=0,
            disagreement_correlation=None,
        )
        assert ctrl.check_transition(metrics_low, 1.0) is None

        # Enough traces
        metrics_high = DevelopmentalMetrics(
            total_traces=100,
            consecutive_traces_with_eval=100,
            successful_dissolutions=0,
            archive_size=0,
            disagreement_correlation=None,
        )
        assert ctrl.check_transition(metrics_high, 2.0) == DevelopmentalStage.BORN

    def test_transition_history_recorded(self) -> None:
        """Transition history records all transitions."""
        ctrl = DevelopmentalController()
        metrics = DevelopmentalMetrics(
            total_traces=1,
            consecutive_traces_with_eval=0,
            successful_dissolutions=0,
            archive_size=0,
            disagreement_correlation=None,
        )
        ctrl.check_transition(metrics, 10.0)
        history = ctrl.transition_history
        assert len(history) == 1
        assert history[0] == (10.0, DevelopmentalStage.EMBRYONIC, DevelopmentalStage.FETAL)

    def test_no_transition_past_adult(self) -> None:
        """No transition possible from ADULT."""
        ctrl = DevelopmentalController(DevelopmentalStage.ADULT)
        metrics = DevelopmentalMetrics(
            total_traces=10000,
            consecutive_traces_with_eval=10000,
            successful_dissolutions=10000,
            archive_size=10000,
            disagreement_correlation=0.99,
        )
        assert ctrl.check_transition(metrics, 1.0) is None


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


class TestContinualLearningSystem:
    """Tests for the ContinualLearningSystem orchestrator."""

    def test_embryonic_stage_no_evaluation(self) -> None:
        """System at EMBRYONIC, run() returns RunResult with no evaluations."""
        evaluators = (
            _StubEvaluator(0.8, "info"),
            _StubEvaluator(0.7, "boundary"),
        )
        system = ContinualLearningSystem(
            evaluators=evaluators,
            development=DevelopmentalController(DevelopmentalStage.EMBRYONIC),
        )
        trace = make_test_trace()
        result = system.run(trace)

        assert result.evaluations is None
        assert result.disagreement is None
        assert result.output == "test output"
        assert result.trace is trace

    def test_fetal_stage_runs_evaluation(self) -> None:
        """Force to FETAL, provide evaluators, verify evaluations and disagreement."""
        evaluators = (
            _StubEvaluator(0.9, "info"),
            _StubEvaluator(0.3, "boundary"),
            _StubEvaluator(0.7, "goodhart"),
        )
        system = ContinualLearningSystem(
            evaluators=evaluators,
            development=DevelopmentalController(DevelopmentalStage.FETAL),
        )
        trace = make_test_trace()
        result = system.run(trace)

        assert result.evaluations is not None
        assert len(result.evaluations) == 3
        assert result.disagreement is not None
        # Max pairwise diff is |0.9 - 0.3| = 0.6
        assert result.disagreement.magnitude == pytest.approx(0.6, abs=0.01)

    def test_born_stage_ticks_breathing(self) -> None:
        """Force to BORN, provide tempo + annealing, verify breathing state changes."""
        tempo = PIDTempoController()
        annealing = AnnealingSchedule(t_init=1.0, cooling_rate=0.95)
        initial_temp = annealing.temperature

        system = ContinualLearningSystem(
            evaluators=(_StubEvaluator(0.5, "info"),),
            tempo=tempo,
            annealing=annealing,
            development=DevelopmentalController(DevelopmentalStage.BORN),
        )
        trace = make_test_trace()
        system.run(trace)

        # Annealing should have cooled
        assert annealing.temperature < initial_temp

    def test_childhood_stage_stores_in_archive(self) -> None:
        """Force to CHILDHOOD, provide archive, verify trace stored."""
        archive = TraceArchive(":memory:")
        system = ContinualLearningSystem(
            evaluators=(_StubEvaluator(0.5, "info"),),
            archive=archive,
            development=DevelopmentalController(DevelopmentalStage.CHILDHOOD),
        )
        trace = make_test_trace()
        system.run(trace)

        assert archive.count() == 1
        loaded = archive.load("trace-001")
        assert loaded is not None
        assert loaded.final_output == "test output"

    def test_developmental_transition_embryonic_to_fetal_via_run(self) -> None:
        """Start at EMBRYONIC, run once, verify transition to FETAL."""
        system = ContinualLearningSystem(
            evaluators=(_StubEvaluator(0.5, "info"),),
            development=DevelopmentalController(DevelopmentalStage.EMBRYONIC),
        )
        trace = make_test_trace()
        result = system.run(trace)

        # EMBRYONIC -> FETAL is always allowed, should happen after first run
        assert result.stage == DevelopmentalStage.FETAL

    def test_surfacing_logic(self) -> None:
        """At BORN with high disagreement + constructive interference -> surfaced_for_human=True."""
        tempo = PIDTempoController()
        interference = InterferencePattern()

        # Set up frequencies so interference is constructive (close freqs)
        # and enough time since last surface
        now = time.time()
        for i in range(5):
            tempo.record_human_feedback(now - i * 60)
            tempo.record_system_event(now - i * 60, "run")

        # High disagreement evaluators: 0.9 vs 0.1 -> magnitude 0.8 -> should_surface=True
        evaluators = (
            _StubEvaluator(0.9, "info"),
            _StubEvaluator(0.1, "boundary"),
        )
        system = ContinualLearningSystem(
            evaluators=evaluators,
            tempo=tempo,
            interference=interference,
            development=DevelopmentalController(DevelopmentalStage.BORN),
            disagreement_threshold=0.3,
        )
        trace = make_test_trace(timestamp=now + 1000)
        result = system.run(trace)

        assert result.surfaced_for_human is True

    def test_no_surfacing_low_disagreement(self) -> None:
        """At BORN with low disagreement -> surfaced_for_human=False."""
        tempo = PIDTempoController()
        interference = InterferencePattern()

        now = time.time()
        for i in range(5):
            tempo.record_human_feedback(now - i * 60)
            tempo.record_system_event(now - i * 60, "run")

        # Low disagreement: both at 0.5 -> magnitude 0.0 -> should_surface=False
        evaluators = (
            _StubEvaluator(0.5, "info"),
            _StubEvaluator(0.5, "boundary"),
        )
        system = ContinualLearningSystem(
            evaluators=evaluators,
            tempo=tempo,
            interference=interference,
            development=DevelopmentalController(DevelopmentalStage.BORN),
            disagreement_threshold=0.3,
        )
        trace = make_test_trace(timestamp=now + 1000)
        result = system.run(trace)

        assert result.surfaced_for_human is False

    def test_receive_human_feedback_updates_anchor(self) -> None:
        """Call receive_human_feedback, verify feedback_anchor recorded."""
        anchor = FeedbackAnchor()
        system = ContinualLearningSystem(
            feedback_anchor=anchor,
        )
        ts = time.time()
        system.receive_human_feedback(ts, "correction")

        # Anchor should have recorded the event
        assert anchor.time_since_last(now=ts) == pytest.approx(0.0, abs=0.01)

    def test_receive_human_feedback_reheats_annealing(self) -> None:
        """Call receive_human_feedback, verify annealing reheat."""
        annealing = AnnealingSchedule(t_init=1.0, cooling_rate=0.95)

        # Cool several times first
        for _ in range(10):
            annealing.cool()
        cooled_temp = annealing.temperature
        assert cooled_temp < 1.0

        system = ContinualLearningSystem(annealing=annealing)
        system.receive_human_feedback(time.time())

        # Temperature should have increased (reheat)
        assert annealing.temperature > cooled_temp

    def test_status_returns_correct_snapshot(self) -> None:
        """Verify all fields populated correctly in status."""
        tempo = PIDTempoController()
        annealing = AnnealingSchedule(t_init=1.0)
        archive = TraceArchive(":memory:")

        system = ContinualLearningSystem(
            tempo=tempo,
            annealing=annealing,
            archive=archive,
            development=DevelopmentalController(DevelopmentalStage.FETAL),
        )

        status = system.status()
        assert status.stage == DevelopmentalStage.FETAL
        assert "evaluation" in status.enabled_features
        assert "execution" in status.enabled_features
        assert status.total_runs == 0
        assert status.archive_size == 0
        assert status.active_combinators == 0
        assert status.breathing_rate is not None
        assert status.temperature == 1.0

    def test_run_counter_increments(self) -> None:
        """Run multiple times, verify total_runs in status."""
        system = ContinualLearningSystem(
            development=DevelopmentalController(DevelopmentalStage.FETAL),
        )

        for i in range(5):
            trace = make_test_trace(trace_id=f"trace-{i}")
            system.run(trace)

        assert system.status().total_runs == 5

    def test_archive_stores_evaluations(self) -> None:
        """Archive stores both trace and evaluation data."""
        archive = TraceArchive(":memory:")
        evaluators = (
            _StubEvaluator(0.9, "info"),
            _StubEvaluator(0.3, "boundary"),
        )
        system = ContinualLearningSystem(
            evaluators=evaluators,
            archive=archive,
            development=DevelopmentalController(DevelopmentalStage.CHILDHOOD),
        )
        trace = make_test_trace()
        system.run(trace)

        # Archive should contain the trace
        assert archive.count() == 1

        # Disagreement should also be stored (magnitude 0.6 >= 0.3 threshold)
        failures = archive.query_by_failure(min_disagreement=0.3)
        assert len(failures) == 1

    def test_embryonic_skips_breathing_and_archive(self) -> None:
        """At EMBRYONIC, breathing and archive are not used."""
        annealing = AnnealingSchedule(t_init=1.0, cooling_rate=0.95)
        archive = TraceArchive(":memory:")
        initial_temp = annealing.temperature

        system = ContinualLearningSystem(
            annealing=annealing,
            archive=archive,
            development=DevelopmentalController(DevelopmentalStage.EMBRYONIC),
        )
        trace = make_test_trace()
        system.run(trace)

        # Annealing should NOT have cooled (breathing not enabled at EMBRYONIC)
        assert annealing.temperature == initial_temp
        # Archive should be empty (not enabled at EMBRYONIC)
        assert archive.count() == 0
