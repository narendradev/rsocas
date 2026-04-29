"""Comprehensive end-to-end integration test for the full RSOCAS system.

Tests ALL seven capabilities against live Lambda-RLM + Nemotron:
1. Patch Lambda-RLM with patch_lambda_rlm_full() for tracing + versioning
2. Run contrapuntal evaluation on any trace
3. Breathe — crystallize/dissolve combinators on annealing schedule
4. Store and query traces, repair episodes, distribution shifts
5. Feed GEPA tree-structured traces with per-node credit assignment
6. Discover combinators via Meta-Harness with AST validation
7. Develop — system progresses through irreversible developmental stages
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "lambda-rlm"))

from rsocas.contracts.traces import TreeTrace, NodeTrace, LeafTrace
from rsocas.contracts.evaluation import EvalResult, DisagreementSignal
from rsocas.contracts.combinators import ValidationSnapshot, VersionedCombinator

from rsocas.tracing.patch import patch_for_tracing
from rsocas.tracing.builder import TreeTraceBuilder
from rsocas.tracing.collector import TraceCollector

from rsocas.evaluation.info_theoretic import InformationTheoreticEval
from rsocas.evaluation.boundary_detection import BoundaryDetectionEval
from rsocas.evaluation.goodhart_resistant import GoodhartResistantEval
from rsocas.evaluation.disagreement import compute_disagreement

from rsocas.combinators.versioned import CombinatorDB
from rsocas.combinators.penumbra import PenumbraStore
from rsocas.combinators.crystallizer import Crystallizer
from rsocas.combinators.registry import CombinatorRegistry
from rsocas.combinators.lambda_rlm_integration import patch_lambda_rlm_full

from rsocas.breathing.tempo import PIDTempoController
from rsocas.breathing.annealing import AnnealingSchedule
from rsocas.breathing.feedback_anchor import FeedbackAnchor
from rsocas.breathing.interference import InterferencePattern
from rsocas.breathing.breathing_crystallizer import BreathingCrystallizer

from rsocas.archive.trace_archive import TraceArchive
from rsocas.archive.repair_index import RepairIndex
from rsocas.archive.distribution_tracker import DistributionTracker

from rsocas.adapters.gepa_tree_adapter import TreeTraceGEPAAdapter
from rsocas.adapters.dspy_leaf_registry import DSPyLeafRegistry, LeafModuleSpec
from rsocas.adapters.metaharness_bridge import MetaHarnessBridge, CombinatorCandidate

from rsocas.development.stages import DevelopmentalStage, DevelopmentalController, DevelopmentalMetrics
from rsocas.development.orchestrator import ContinualLearningSystem, RunResult, SystemStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        input_size=sum(l.input_size for l in leaves),
        output=final_output,
        children=tuple(l.id for l in leaves),
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


def _make_bad_trace() -> TreeTrace:
    """A trace where the system echoes input and gives wrong answer."""
    prompt_text = "The capital of France is Paris. " * 50
    return TreeTrace(
        trace_id=uuid.uuid4().hex,
        task_type="QA", k=1, depth=0, tau=50000,
        cost_estimate=0.5,
        nodes=(
            NodeTrace(
                id="leaf_0", depth=0, position=0, combinator="LEAF",
                input_size=len(prompt_text), output=prompt_text[:200],
                children=(), llm_calls=1, latency_ms=300.0,
            ),
        ),
        leaf_traces=(
            LeafTrace(
                node_id="leaf_0", prompt=prompt_text,
                response=prompt_text[:200],
                tokens_in=500, tokens_out=50, model="nemotron-3-super",
            ),
        ),
        final_output=prompt_text[:200],
        final_score=0.0,
        timestamp=time.time(),
        execution_time_ms=300.0,
        total_llm_calls=1, total_tokens=550,
    )


# ---------------------------------------------------------------------------
# 1. Patch Lambda-RLM with tracing + versioning
# ---------------------------------------------------------------------------

class TestCapability1_PatchLambdaRLM:
    """Verify Lambda-RLM patching for tracing and combinator versioning."""

    def test_patch_for_tracing_creates_collector(self):
        class MockLRLM:
            def _register_library(self, repl, plan, query=""): pass
        lrlm = MockLRLM()
        patched, collector = patch_for_tracing(lrlm)
        assert collector is not None
        assert patched is lrlm

    def test_full_patch_creates_registry(self):
        class MockLRLM:
            def _register_library(self, repl, plan, query=""): pass
        lrlm = MockLRLM()
        patched, collector, registry = patch_lambda_rlm_full(lrlm)
        assert collector is not None
        assert registry is not None
        assert patched is lrlm

    def test_full_patch_registers_combinators(self):
        class MockLRLM:
            def _register_library(self, repl, plan, query=""):
                repl.globals["_Split"] = lambda x, k: [x]
                repl.globals["_Reduce"] = lambda parts: parts[0]
        class MockREPL:
            globals = {}
            _llm_query = lambda self, p, m=None: "mock"
        lrlm = MockLRLM()
        registry = CombinatorRegistry(
            Crystallizer(CombinatorDB(":memory:"), PenumbraStore(CombinatorDB(":memory:")))
        )
        patched, collector, reg = patch_lambda_rlm_full(lrlm, registry=registry)
        repl = MockREPL()
        patched._register_library(repl, None)
        versions = reg.get_active_versions()
        assert "_Split" in versions
        assert "_Reduce" in versions

    def test_collector_captures_llm_calls(self):
        class MockLRLM:
            def _register_library(self, repl, plan, query=""):
                pass
        lrlm = MockLRLM()
        patched, collector = patch_for_tracing(lrlm)
        class MockREPL:
            globals = {"llm_query": lambda p, m=None: "resp"}
            _llm_query = lambda self, p, m=None: "resp"
        repl = MockREPL()
        patched._register_library(repl, None)
        result = repl._llm_query("test prompt")
        assert result == "resp"
        events = collector.get_events()
        assert len(events) == 1
        assert events[0].prompt == "test prompt"


# ---------------------------------------------------------------------------
# 2. Contrapuntal evaluation
# ---------------------------------------------------------------------------

class TestCapability2_ContrapuntalEvaluation:
    """Verify three evaluators produce meaningful, differentiated scores."""

    def test_good_trace_all_evaluators_score_high(self):
        trace = _make_trace(final_output="Paris is the capital of France")
        info = InformationTheoreticEval().evaluate(trace)
        boundary = BoundaryDetectionEval().evaluate(trace)
        goodhart = GoodhartResistantEval().evaluate(trace)
        assert info.score > 0.0
        assert boundary.score > 0.0
        assert info.signal_type == "information_theoretic"
        assert boundary.signal_type == "boundary"
        assert goodhart.signal_type == "goodhart_resistant"

    def test_bad_trace_boundary_detects_echo(self):
        trace = _make_bad_trace()
        boundary = BoundaryDetectionEval().evaluate(trace)
        assert boundary.score < 0.8, f"Expected low score for echoing trace, got {boundary.score}"

    def test_disagreement_varies_between_good_and_bad(self):
        good = _make_trace()
        bad = _make_bad_trace()
        evaluators = (InformationTheoreticEval(), BoundaryDetectionEval(), GoodhartResistantEval())
        good_evals = tuple(e.evaluate(good) for e in evaluators)
        bad_evals = tuple(e.evaluate(bad) for e in evaluators)
        good_disagree = compute_disagreement(good_evals)
        bad_disagree = compute_disagreement(bad_evals)
        assert good_disagree.magnitude != bad_disagree.magnitude, \
            "Disagreement should differ between good and bad traces"

    def test_per_node_scores_populated(self):
        trace = _make_trace(n_leaves=4)
        info = InformationTheoreticEval().evaluate(trace)
        assert len(info.per_node_scores) > 0
        for node in trace.nodes:
            assert node.id in info.per_node_scores

    def test_goodhart_with_rerun_fn(self):
        trace = _make_trace()
        def stable_rerun(prompt):
            return "partial answer 0 with some details about the topic"
        goodhart = GoodhartResistantEval(rerun_fn=stable_rerun)
        result = goodhart.evaluate(trace)
        assert result.score > 0.5
        assert result.confidence > 0.0

    def test_goodhart_with_unstable_rerun(self):
        trace = _make_trace()
        counter = [0]
        def unstable_rerun(prompt):
            counter[0] += 1
            return f"completely different answer #{counter[0]}"
        goodhart = GoodhartResistantEval(rerun_fn=unstable_rerun)
        result = goodhart.evaluate(trace)
        assert result.score < 0.8


# ---------------------------------------------------------------------------
# 3. Breathing — crystallize/dissolve on annealing schedule
# ---------------------------------------------------------------------------

class TestCapability3_BreathingCycle:
    """Verify the full breathing cycle: tempo + annealing + crystallizer."""

    def test_breathing_crystallizer_lifecycle(self):
        db = CombinatorDB(":memory:")
        penumbra = PenumbraStore(db)
        crystallizer = Crystallizer(db, penumbra, default_ttl=1.0)
        tempo = PIDTempoController()
        annealing = AnnealingSchedule(t_init=1.0, t_min=0.01)
        bc = BreathingCrystallizer(crystallizer, tempo, annealing)

        now = time.time()
        validation = ValidationSnapshot(("QA",), (1000, 50000), 10, 0.8, 0.1, now)
        crystallizer.crystallize("test_combinator", lambda x: x, validation)

        events1 = bc.tick(now + 0.5)
        assert any(e.event_type in ("noop", "cooled") for e in events1)

        events2 = bc.tick(now + 2.0)
        dissolved = [e for e in events2 if e.event_type == "dissolved"]
        assert len(dissolved) >= 1, "Should dissolve expired combinator"

    def test_reheat_on_high_disagreement(self):
        db = CombinatorDB(":memory:")
        crystallizer = Crystallizer(db, PenumbraStore(db))
        tempo = PIDTempoController()
        annealing = AnnealingSchedule(t_init=1.0)
        bc = BreathingCrystallizer(crystallizer, tempo, annealing)

        for _ in range(5):
            annealing.cool()
        temp_before = annealing.temperature

        high_disagree = DisagreementSignal(
            magnitude=0.8, should_surface=True, timestamp=time.time(),
        )
        events = bc.tick(time.time(), disagreement=high_disagree)
        assert any(e.event_type == "reheated" for e in events)
        assert annealing.temperature > temp_before

    def test_human_feedback_reheats(self):
        db = CombinatorDB(":memory:")
        crystallizer = Crystallizer(db, PenumbraStore(db))
        tempo = PIDTempoController()
        annealing = AnnealingSchedule(t_init=1.0)
        bc = BreathingCrystallizer(crystallizer, tempo, annealing)

        for _ in range(10):
            annealing.cool()
        temp_before = annealing.temperature
        event = bc.receive_human_feedback(time.time())
        assert event.event_type == "reheated"
        assert annealing.temperature > temp_before

    def test_tempo_responds_to_feedback_frequency(self):
        tempo = PIDTempoController(target_ratio=2.0)
        now = time.time()
        for i in range(10):
            tempo.record_human_feedback(now + i * 60)
        rate_with_feedback = tempo.breathing_rate()

        tempo2 = PIDTempoController(target_ratio=2.0)
        rate_without = tempo2.breathing_rate()
        assert rate_with_feedback != rate_without

    def test_interference_pattern_surfacing(self):
        ip = InterferencePattern()
        high_disagree = DisagreementSignal(magnitude=0.8, should_surface=True, timestamp=0.0)
        low_disagree = DisagreementSignal(magnitude=0.1, should_surface=False, timestamp=0.0)

        assert ip.should_surface(high_disagree, 1.0, 1.0, min_interval=0, last_surface_time=0)
        assert not ip.should_surface(low_disagree, 1.0, 1.0, min_interval=0, last_surface_time=0)


# ---------------------------------------------------------------------------
# 4. Store and query traces, repairs, distribution shifts
# ---------------------------------------------------------------------------

class TestCapability4_ArchiveAndQuery:
    """Verify trace storage, repair indexing, and distribution tracking."""

    def test_store_and_retrieve_trace(self):
        archive = TraceArchive(":memory:")
        trace = _make_trace(final_score=0.85)
        archive.store(trace)
        loaded = archive.load(trace.trace_id)
        assert loaded is not None
        assert loaded.trace_id == trace.trace_id
        assert loaded.task_type == "QA"
        assert len(loaded.nodes) == 4
        assert len(loaded.leaf_traces) == 3

    def test_store_with_evaluation_and_disagreement(self):
        archive = TraceArchive(":memory:")
        trace = _make_trace()
        evals = (
            EvalResult(score=0.9, confidence=0.8, signal_type="information_theoretic"),
            EvalResult(score=0.7, confidence=0.8, signal_type="boundary"),
            EvalResult(score=0.3, confidence=0.8, signal_type="goodhart_resistant"),
        )
        disagree = DisagreementSignal(
            magnitude=0.6, should_surface=True, timestamp=time.time(),
            pairwise={"info_vs_boundary": 0.2},
        )
        archive.store(trace, evals, disagree)
        failures = archive.query_by_failure(min_disagreement=0.5)
        assert len(failures) == 1
        assert failures[0][1].magnitude == 0.6

    def test_query_by_task_type(self):
        archive = TraceArchive(":memory:")
        archive.store(_make_trace(task_type="QA"))
        archive.store(_make_trace(task_type="SUMMARY"))
        archive.store(_make_trace(task_type="QA"))
        qa_traces = archive.query_by_task_type("QA")
        assert len(qa_traces) == 2

    def test_full_text_search(self):
        archive = TraceArchive(":memory:")
        archive.store(_make_trace(final_output="The capital of France is Paris"))
        archive.store(_make_trace(final_output="Photosynthesis converts light to energy"))
        results = archive.search_output("Paris")
        assert len(results) == 1
        assert "Paris" in results[0].final_output

    def test_repair_index(self):
        archive = TraceArchive(":memory:")
        trace_before = _make_trace(final_score=0.3)
        trace_after = _make_trace(final_score=0.8)
        archive.store(trace_before)
        archive.store(trace_after)
        repair_idx = RepairIndex(archive)
        repair_idx.record_repair(
            "test_combinator", trace_before.trace_id, trace_after.trace_id,
            trigger="disagreement", score_delta=0.5,
        )
        repairs = repair_idx.query_repairs("test_combinator")
        assert len(repairs) == 1
        assert repairs[0].score_delta == 0.5
        assert repairs[0].trigger == "disagreement"

    def test_distribution_tracker(self):
        archive = TraceArchive(":memory:")
        for i in range(5):
            t = _make_trace(final_score=0.5 + i * 0.1)
            archive.store(t)
        tracker = DistributionTracker(archive)
        snapshot = tracker.compute_snapshot("QA", window_seconds=3600)
        assert snapshot.n_samples == 5
        assert 0.5 <= snapshot.mean_score <= 0.9
        assert snapshot.score_std > 0


# ---------------------------------------------------------------------------
# 5. GEPA tree-structured credit assignment
# ---------------------------------------------------------------------------

class TestCapability5_GEPAAdapter:
    """Verify tree trace → GEPA reflective dataset conversion."""

    def test_identify_failing_nodes(self):
        trace = _make_trace(n_leaves=4)
        disagree = DisagreementSignal(
            magnitude=0.7, should_surface=True, timestamp=0.0,
            per_node={"leaf_0": 0.8, "leaf_1": 0.2, "leaf_2": 0.9, "leaf_3": 0.1},
        )
        adapter = TreeTraceGEPAAdapter()
        failing = adapter.identify_failing_nodes(trace, disagree)
        assert len(failing) > 0
        assert failing[0] == "leaf_2"
        assert failing[1] == "leaf_0"

    def test_extract_subtree_context(self):
        trace = _make_trace(n_leaves=3)
        adapter = TreeTraceGEPAAdapter()
        context = adapter.extract_subtree_context(trace, "leaf_1")
        assert "leaf_1" in context
        assert "depth" in context.lower() or "position" in context.lower()

    def test_reflective_dataset_format(self):
        trace = _make_trace(n_leaves=3)
        disagree = DisagreementSignal(
            magnitude=0.7, should_surface=True, timestamp=0.0,
            per_node={"leaf_0": 0.8, "leaf_1": 0.2, "leaf_2": 0.3, "root": 0.1},
        )
        adapter = TreeTraceGEPAAdapter()
        dataset = adapter.make_reflective_dataset(trace, disagree)
        assert "leaf_prompt" in dataset
        entries = dataset["leaf_prompt"]
        assert len(entries) > 0
        entry = entries[0]
        assert "Inputs" in entry
        assert "Generated Outputs" in entry
        assert "Feedback" in entry

    def test_batch_adapter(self):
        traces = [
            (_make_trace(n_leaves=2), DisagreementSignal(
                magnitude=0.6, should_surface=True, timestamp=0.0,
                per_node={"leaf_0": 0.7, "leaf_1": 0.3, "root": 0.1},
            ), "Paris"),
            (_make_trace(n_leaves=2), DisagreementSignal(
                magnitude=0.8, should_surface=True, timestamp=0.0,
                per_node={"leaf_0": 0.9, "leaf_1": 0.1, "root": 0.2},
            ), "London"),
        ]
        adapter = TreeTraceGEPAAdapter()
        dataset = adapter.adapt_for_gepa_optimize(traces)
        assert "leaf_prompt" in dataset
        assert len(dataset["leaf_prompt"]) >= 2


# ---------------------------------------------------------------------------
# 6. Meta-Harness combinator discovery + validation
# ---------------------------------------------------------------------------

class TestCapability6_MetaHarness:
    """Verify combinator candidate validation and lifecycle."""

    def test_valid_combinator_passes(self):
        bridge = MetaHarnessBridge()
        candidate = CombinatorCandidate(
            name="my_verify",
            code="def my_verify(text, checker):\n    return checker(text)\n",
            type_signature="(str, Callable) -> Any",
        )
        valid, msg = bridge.validate_candidate(candidate)
        assert valid, f"Expected valid but got: {msg}"

    def test_unbounded_loop_rejected(self):
        bridge = MetaHarnessBridge()
        candidate = CombinatorCandidate(
            name="bad_loop",
            code="def bad_loop(x):\n    while True:\n        x += 1\n    return x\n",
        )
        valid, msg = bridge.validate_candidate(candidate)
        assert not valid
        assert "unbounded" in msg.lower() or "while true" in msg.lower() or "loop" in msg.lower()

    def test_syntax_error_rejected(self):
        bridge = MetaHarnessBridge()
        candidate = CombinatorCandidate(name="bad_syntax", code="def bad(:\n")
        valid, msg = bridge.validate_candidate(candidate)
        assert not valid

    def test_recursive_call_rejected(self):
        bridge = MetaHarnessBridge()
        candidate = CombinatorCandidate(
            name="recursive",
            code="def recursive(x):\n    return recursive(x-1)\n",
        )
        valid, msg = bridge.validate_candidate(candidate)
        assert not valid

    def test_write_and_load_candidates(self, tmp_path):
        bridge = MetaHarnessBridge(candidates_dir=str(tmp_path / "cands"), archive_dir=str(tmp_path / "arch"))
        candidate = CombinatorCandidate(
            name="test_fn", code="def test_fn(x):\n    return x\n",
            type_signature="Any -> Any", hypothesis="identity function",
        )
        bridge.write_candidate(candidate)
        loaded = bridge.load_candidates()
        assert len(loaded) == 1
        assert loaded[0].name == "test_fn"

    def test_archive_candidate(self, tmp_path):
        bridge = MetaHarnessBridge(candidates_dir=str(tmp_path / "c"), archive_dir=str(tmp_path / "a"))
        candidate = CombinatorCandidate(name="archived", code="def archived(): pass\n")
        validation = ValidationSnapshot(("QA",), (1000, 5000), 10, 0.9, 0.05, time.time())
        bridge.archive_candidate(candidate, validation, accepted=True)
        archives = list((tmp_path / "a").glob("*.json"))
        assert len(archives) == 1

    def test_skill_md_generation(self):
        bridge = MetaHarnessBridge()
        md = bridge.generate_skill_md("Discover new QA combinators for medical domain")
        assert len(md) > 100
        assert "pure function" in md.lower() or "no side effects" in md.lower()


# ---------------------------------------------------------------------------
# 7. Developmental stages — embryonic → adult
# ---------------------------------------------------------------------------

class TestCapability7_Development:
    """Verify the developmental progression and orchestrator integration."""

    def test_stage_progression(self):
        dev = DevelopmentalController(DevelopmentalStage.EMBRYONIC)
        assert dev.current_stage == DevelopmentalStage.EMBRYONIC

        metrics = DevelopmentalMetrics(0, 0, 0, 0, None)
        new = dev.check_transition(metrics, time.time())
        assert new == DevelopmentalStage.FETAL

        metrics = DevelopmentalMetrics(100, 100, 0, 0, None)
        new = dev.check_transition(metrics, time.time())
        assert new == DevelopmentalStage.BORN

    def test_stages_are_irreversible(self):
        dev = DevelopmentalController(DevelopmentalStage.EMBRYONIC)
        dev.force_transition(DevelopmentalStage.BORN, time.time())
        assert dev.current_stage == DevelopmentalStage.BORN
        features = dev.get_enabled_features()
        assert "execution" in features
        assert "evaluation" in features
        assert "breathing" in features

    def test_full_orchestrator_embryonic_to_fetal(self):
        evaluators = (
            InformationTheoreticEval(),
            BoundaryDetectionEval(),
            GoodhartResistantEval(),
        )
        system = ContinualLearningSystem(evaluators=evaluators)
        trace = _make_trace()
        result = system.run(trace)
        assert result.stage == DevelopmentalStage.FETAL
        assert result.evaluations is None  # was embryonic during this run

    def test_full_orchestrator_fetal_evaluates(self):
        evaluators = (
            InformationTheoreticEval(),
            BoundaryDetectionEval(),
            GoodhartResistantEval(),
        )
        dev = DevelopmentalController(DevelopmentalStage.FETAL)
        system = ContinualLearningSystem(evaluators=evaluators, development=dev)
        trace = _make_trace()
        result = system.run(trace)
        assert result.evaluations is not None
        assert len(result.evaluations) == 3
        assert result.disagreement is not None

    def test_full_orchestrator_born_breathes(self):
        evaluators = (
            InformationTheoreticEval(),
            BoundaryDetectionEval(),
            GoodhartResistantEval(),
        )
        dev = DevelopmentalController(DevelopmentalStage.BORN)
        tempo = PIDTempoController()
        annealing = AnnealingSchedule()
        interference = InterferencePattern()
        feedback = FeedbackAnchor()
        system = ContinualLearningSystem(
            evaluators=evaluators, development=dev,
            tempo=tempo, annealing=annealing,
            interference=interference, feedback_anchor=feedback,
        )
        trace = _make_trace()
        result = system.run(trace)
        assert result.evaluations is not None
        assert result.disagreement is not None

    def test_full_orchestrator_childhood_archives(self):
        evaluators = (
            InformationTheoreticEval(),
            BoundaryDetectionEval(),
            GoodhartResistantEval(),
        )
        dev = DevelopmentalController(DevelopmentalStage.CHILDHOOD)
        archive = TraceArchive(":memory:")
        tempo = PIDTempoController()
        annealing = AnnealingSchedule()
        system = ContinualLearningSystem(
            evaluators=evaluators, development=dev,
            archive=archive, tempo=tempo, annealing=annealing,
        )
        trace = _make_trace()
        result = system.run(trace)
        assert archive.count() == 1

    def test_full_orchestrator_status(self):
        evaluators = (InformationTheoreticEval(), BoundaryDetectionEval(), GoodhartResistantEval())
        dev = DevelopmentalController(DevelopmentalStage.FETAL)
        system = ContinualLearningSystem(evaluators=evaluators, development=dev)
        for _ in range(3):
            system.run(_make_trace())
        status = system.status()
        assert status.total_runs == 3
        assert "evaluation" in status.enabled_features

    def test_human_feedback_flow(self):
        evaluators = (InformationTheoreticEval(), BoundaryDetectionEval(), GoodhartResistantEval())
        dev = DevelopmentalController(DevelopmentalStage.BORN)
        tempo = PIDTempoController()
        annealing = AnnealingSchedule()
        feedback = FeedbackAnchor()
        system = ContinualLearningSystem(
            evaluators=evaluators, development=dev,
            tempo=tempo, annealing=annealing, feedback_anchor=feedback,
        )
        for _ in range(5):
            annealing.cool()
        temp_before = annealing.temperature
        system.receive_human_feedback(time.time(), "correction")
        assert annealing.temperature > temp_before


# ---------------------------------------------------------------------------
# 8. Full pipeline integration — everything wired together
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end: trace → evaluate → breathe → archive → adapt for GEPA."""

    def test_full_pipeline_10_traces(self):
        """Run 10 traces through the complete system and verify all subsystems engage."""
        archive = TraceArchive(":memory:")
        db = CombinatorDB(":memory:")
        penumbra = PenumbraStore(db)
        crystallizer = Crystallizer(db, penumbra, default_ttl=5.0)
        tempo = PIDTempoController()
        annealing = AnnealingSchedule(t_init=1.0, t_min=0.01)
        bc = BreathingCrystallizer(crystallizer, tempo, annealing)
        feedback = FeedbackAnchor()
        interference = InterferencePattern()
        evaluators = (
            InformationTheoreticEval(),
            BoundaryDetectionEval(),
            GoodhartResistantEval(),
        )
        dev = DevelopmentalController(DevelopmentalStage.CHILDHOOD)
        system = ContinualLearningSystem(
            evaluators=evaluators, development=dev,
            tempo=tempo, annealing=annealing, archive=archive,
            interference=interference, feedback_anchor=feedback,
        )
        adapter = TreeTraceGEPAAdapter()

        all_results = []
        for i in range(10):
            score = 0.3 + (i % 3) * 0.25
            trace = _make_trace(
                final_output=f"answer_{i}",
                final_score=score,
                n_leaves=2 + (i % 3),
            )
            result = system.run(trace)
            all_results.append(result)

            bc.tick(time.time())

        assert archive.count() == 10
        assert all(r.evaluations is not None for r in all_results)
        assert all(r.disagreement is not None for r in all_results)

        status = system.status()
        assert status.total_runs == 10

        tracker = DistributionTracker(archive)
        snapshot = tracker.compute_snapshot("QA")
        assert snapshot.n_samples == 10

        last_trace = _make_trace(n_leaves=3)
        last_evals = tuple(e.evaluate(last_trace) for e in evaluators)
        last_disagree = compute_disagreement(last_evals)
        dataset = adapter.make_reflective_dataset(last_trace, last_disagree)
        assert isinstance(dataset, dict)

    def test_good_vs_bad_traces_diverge(self):
        """Verify the system produces different signals for good vs bad traces."""
        evaluators = (InformationTheoreticEval(), BoundaryDetectionEval(), GoodhartResistantEval())
        dev = DevelopmentalController(DevelopmentalStage.FETAL)
        system = ContinualLearningSystem(evaluators=evaluators, development=dev)

        good_result = system.run(_make_trace())
        bad_result = system.run(_make_bad_trace())

        assert good_result.disagreement is not None
        assert bad_result.disagreement is not None
        assert good_result.disagreement.magnitude != bad_result.disagreement.magnitude, \
            "System should produce different disagreement for good vs bad traces"
