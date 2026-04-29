"""Tests for rsocas.archive — TraceArchive, RepairIndex, DistributionTracker."""

from __future__ import annotations

import time

import pytest

from rsocas.archive.distribution_tracker import DistributionTracker
from rsocas.archive.repair_index import RepairEpisode, RepairIndex
from rsocas.archive.trace_archive import TraceArchive
from rsocas.contracts.evaluation import DisagreementSignal, EvalResult
from rsocas.contracts.traces import LeafTrace, NodeTrace, TreeTrace


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_node(
    node_id: str = "n0",
    depth: int = 0,
    position: int = 0,
    combinator: str = "split",
    input_size: int = 100,
    output: str = "node output",
    score: float | None = 0.8,
) -> NodeTrace:
    return NodeTrace(
        id=node_id,
        depth=depth,
        position=position,
        combinator=combinator,
        input_size=input_size,
        output=output,
        children=(),
        llm_calls=1,
        latency_ms=50.0,
        score=score,
    )


def _make_leaf(
    node_id: str = "n0",
    prompt: str = "test prompt",
    response: str = "test response",
) -> LeafTrace:
    return LeafTrace(
        node_id=node_id,
        prompt=prompt,
        response=response,
        tokens_in=10,
        tokens_out=20,
        model="test-model",
        confidence=0.9,
    )


def _make_trace(
    trace_id: str = "t-001",
    task_type: str = "summarization",
    final_output: str = "The final summary.",
    final_score: float | None = 0.85,
    timestamp: float | None = None,
    combinator_versions: dict[str, str] | None = None,
    metadata: dict | None = None,
    input_size: int = 100,
) -> TreeTrace:
    ts = timestamp if timestamp is not None else time.time()
    node = _make_node(input_size=input_size)
    leaf = _make_leaf()
    return TreeTrace(
        trace_id=trace_id,
        task_type=task_type,
        k=3,
        depth=2,
        tau=5,
        cost_estimate=0.05,
        nodes=(node,),
        leaf_traces=(leaf,),
        final_output=final_output,
        timestamp=ts,
        execution_time_ms=120.0,
        total_llm_calls=3,
        total_tokens=100,
        final_score=final_score,
        combinator_versions=combinator_versions or {},
        metadata=metadata or {},
    )


def _make_eval(
    signal_type: str = "coherence",
    score: float = 0.9,
    confidence: float = 0.8,
) -> EvalResult:
    return EvalResult(
        score=score,
        confidence=confidence,
        signal_type=signal_type,
        per_node_scores={"n0": score},
        explanation=f"{signal_type} looks good",
    )


def _make_disagreement(
    magnitude: float = 0.5,
    should_surface: bool = True,
) -> DisagreementSignal:
    return DisagreementSignal(
        magnitude=magnitude,
        pairwise={"coherence-fluency": 0.3},
        per_node={"n0": 0.4},
        outlier_voice="coherence",
        should_surface=should_surface,
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# TraceArchive tests
# ---------------------------------------------------------------------------


class TestTraceArchiveStoreAndLoad:
    """Store and load round-trip: TreeTrace survives serialization."""

    def test_store_and_load_roundtrip(self) -> None:
        archive = TraceArchive(":memory:")
        trace = _make_trace(
            combinator_versions={"split": "v1.2"},
            metadata={"source": "test"},
        )
        returned_id = archive.store(trace)
        assert returned_id == "t-001"

        loaded = archive.load("t-001")
        assert loaded is not None
        assert loaded.trace_id == trace.trace_id
        assert loaded.task_type == trace.task_type
        assert loaded.k == trace.k
        assert loaded.depth == trace.depth
        assert loaded.tau == trace.tau
        assert loaded.cost_estimate == trace.cost_estimate
        assert loaded.final_output == trace.final_output
        assert loaded.final_score == trace.final_score
        assert loaded.execution_time_ms == trace.execution_time_ms
        assert loaded.total_llm_calls == trace.total_llm_calls
        assert loaded.total_tokens == trace.total_tokens
        assert loaded.combinator_versions == {"split": "v1.2"}
        assert loaded.metadata == {"source": "test"}
        archive.close()

    def test_nodes_survive_roundtrip(self) -> None:
        archive = TraceArchive(":memory:")
        trace = _make_trace()
        archive.store(trace)
        loaded = archive.load("t-001")
        assert loaded is not None
        assert len(loaded.nodes) == 1
        node = loaded.nodes[0]
        assert node.id == "n0"
        assert node.depth == 0
        assert node.combinator == "split"
        assert node.input_size == 100
        assert node.score == 0.8
        assert isinstance(node.children, tuple)
        archive.close()

    def test_leaf_traces_survive_roundtrip(self) -> None:
        archive = TraceArchive(":memory:")
        trace = _make_trace()
        archive.store(trace)
        loaded = archive.load("t-001")
        assert loaded is not None
        assert len(loaded.leaf_traces) == 1
        leaf = loaded.leaf_traces[0]
        assert leaf.node_id == "n0"
        assert leaf.prompt == "test prompt"
        assert leaf.response == "test response"
        assert leaf.tokens_in == 10
        assert leaf.tokens_out == 20
        assert leaf.model == "test-model"
        assert leaf.confidence == 0.9
        archive.close()

    def test_load_nonexistent_returns_none(self) -> None:
        archive = TraceArchive(":memory:")
        assert archive.load("nonexistent") is None
        archive.close()


class TestTraceArchiveQueryByTaskType:
    """query_by_task_type returns correct traces."""

    def test_filters_by_task_type(self) -> None:
        archive = TraceArchive(":memory:")
        archive.store(_make_trace(trace_id="t-1", task_type="summarization"))
        archive.store(_make_trace(trace_id="t-2", task_type="qa"))
        archive.store(_make_trace(trace_id="t-3", task_type="summarization"))

        results = archive.query_by_task_type("summarization")
        assert len(results) == 2
        ids = {t.trace_id for t in results}
        assert ids == {"t-1", "t-3"}
        archive.close()

    def test_respects_limit(self) -> None:
        archive = TraceArchive(":memory:")
        for i in range(5):
            archive.store(
                _make_trace(trace_id=f"t-{i}", task_type="qa")
            )
        results = archive.query_by_task_type("qa", limit=3)
        assert len(results) == 3
        archive.close()

    def test_empty_result_for_unknown_type(self) -> None:
        archive = TraceArchive(":memory:")
        results = archive.query_by_task_type("unknown_type")
        assert results == []
        archive.close()


class TestTraceArchiveQueryByFailure:
    """query_by_failure returns traces with high disagreement."""

    def test_returns_high_disagreement_traces(self) -> None:
        archive = TraceArchive(":memory:")
        trace_low = _make_trace(trace_id="low")
        trace_high = _make_trace(trace_id="high")

        archive.store(trace_low, disagreement=_make_disagreement(magnitude=0.1))
        archive.store(trace_high, disagreement=_make_disagreement(magnitude=0.5))

        results = archive.query_by_failure(min_disagreement=0.3)
        assert len(results) == 1
        trace, signal = results[0]
        assert trace.trace_id == "high"
        assert signal.magnitude == 0.5
        archive.close()

    def test_ordered_by_magnitude_desc(self) -> None:
        archive = TraceArchive(":memory:")
        for i, mag in enumerate([0.4, 0.8, 0.6]):
            archive.store(
                _make_trace(trace_id=f"t-{i}"),
                disagreement=_make_disagreement(magnitude=mag),
            )
        results = archive.query_by_failure(min_disagreement=0.3)
        magnitudes = [s.magnitude for _, s in results]
        assert magnitudes == sorted(magnitudes, reverse=True)
        archive.close()

    def test_empty_when_no_disagreements(self) -> None:
        archive = TraceArchive(":memory:")
        archive.store(_make_trace())
        results = archive.query_by_failure()
        assert results == []
        archive.close()


class TestTraceArchiveSearchOutput:
    """search_output finds traces by text content via FTS5."""

    def test_finds_matching_output(self) -> None:
        archive = TraceArchive(":memory:")
        archive.store(
            _make_trace(trace_id="t-1", final_output="The quantum cat sat on the mat")
        )
        archive.store(
            _make_trace(trace_id="t-2", final_output="A dog ran in the park")
        )

        results = archive.search_output("quantum")
        assert len(results) == 1
        assert results[0].trace_id == "t-1"
        archive.close()

    def test_empty_for_no_match(self) -> None:
        archive = TraceArchive(":memory:")
        archive.store(_make_trace(final_output="hello world"))
        results = archive.search_output("nonexistent")
        assert results == []
        archive.close()


class TestTraceArchiveCount:
    """count returns correct count."""

    def test_count_empty(self) -> None:
        archive = TraceArchive(":memory:")
        assert archive.count() == 0
        archive.close()

    def test_count_after_inserts(self) -> None:
        archive = TraceArchive(":memory:")
        for i in range(3):
            archive.store(_make_trace(trace_id=f"t-{i}"))
        assert archive.count() == 3
        archive.close()


class TestTraceArchiveCombinatorVersion:
    """query_by_combinator_version finds traces containing a version id."""

    def test_finds_matching_version(self) -> None:
        archive = TraceArchive(":memory:")
        archive.store(
            _make_trace(
                trace_id="t-1",
                combinator_versions={"split": "v1.0"},
            )
        )
        archive.store(
            _make_trace(
                trace_id="t-2",
                combinator_versions={"split": "v2.0"},
            )
        )
        results = archive.query_by_combinator_version("v1.0")
        assert len(results) == 1
        assert results[0].trace_id == "t-1"
        archive.close()


# ---------------------------------------------------------------------------
# RepairIndex tests
# ---------------------------------------------------------------------------


class TestRepairIndex:
    """RepairIndex: record and query repair episodes."""

    def test_record_and_query(self) -> None:
        archive = TraceArchive(":memory:")
        index = RepairIndex(archive)

        index.record_repair(
            combinator_name="split",
            before_trace_id="t-before",
            after_trace_id="t-after",
            trigger="high disagreement",
            score_delta=0.15,
        )

        repairs = index.query_repairs("split")
        assert len(repairs) == 1
        ep = repairs[0]
        assert isinstance(ep, RepairEpisode)
        assert ep.combinator_name == "split"
        assert ep.before_trace_id == "t-before"
        assert ep.after_trace_id == "t-after"
        assert ep.trigger == "high disagreement"
        assert ep.score_delta == 0.15
        assert ep.timestamp > 0
        archive.close()

    def test_query_recent_repairs(self) -> None:
        archive = TraceArchive(":memory:")
        index = RepairIndex(archive)

        index.record_repair("split", "b1", "a1", "trigger1", 0.1)
        index.record_repair("merge", "b2", "a2", "trigger2", 0.2)

        recent = index.query_recent_repairs(window_seconds=60.0)
        assert len(recent) == 2
        archive.close()

    def test_query_repairs_empty(self) -> None:
        archive = TraceArchive(":memory:")
        index = RepairIndex(archive)
        assert index.query_repairs("nonexistent") == []
        archive.close()

    def test_query_recent_empty_archive(self) -> None:
        archive = TraceArchive(":memory:")
        index = RepairIndex(archive)
        assert index.query_recent_repairs() == []
        archive.close()


# ---------------------------------------------------------------------------
# DistributionTracker tests
# ---------------------------------------------------------------------------


class TestDistributionTracker:
    """DistributionTracker: compute_snapshot from stored traces."""

    def test_compute_snapshot_basic(self) -> None:
        archive = TraceArchive(":memory:")
        tracker = DistributionTracker(archive)

        now = time.time()
        archive.store(
            _make_trace(
                trace_id="t-1",
                task_type="qa",
                final_score=0.8,
                timestamp=now,
                input_size=50,
            )
        )
        archive.store(
            _make_trace(
                trace_id="t-2",
                task_type="qa",
                final_score=0.6,
                timestamp=now,
                input_size=200,
            )
        )

        snap = tracker.compute_snapshot("qa", window_seconds=60.0)
        assert snap.task_types == ("qa",)
        assert snap.n_samples == 2
        assert snap.mean_score == pytest.approx(0.7)
        assert snap.score_std > 0
        assert snap.input_size_range == (50, 200)
        assert snap.timestamp > 0
        archive.close()

    def test_compute_snapshot_no_scores(self) -> None:
        archive = TraceArchive(":memory:")
        tracker = DistributionTracker(archive)

        archive.store(
            _make_trace(
                trace_id="t-1",
                task_type="qa",
                final_score=None,
                timestamp=time.time(),
            )
        )
        snap = tracker.compute_snapshot("qa", window_seconds=60.0)
        assert snap.n_samples == 1
        assert snap.mean_score == 0.0
        assert snap.score_std == 0.0
        archive.close()

    def test_compute_snapshot_empty(self) -> None:
        archive = TraceArchive(":memory:")
        tracker = DistributionTracker(archive)

        snap = tracker.compute_snapshot("unknown", window_seconds=60.0)
        assert snap.n_samples == 0
        assert snap.mean_score == 0.0
        assert snap.score_std == 0.0
        assert snap.input_size_range == (0, 0)
        archive.close()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Empty archive returns empty results, not errors."""

    def test_empty_archive_load(self) -> None:
        archive = TraceArchive(":memory:")
        assert archive.load("anything") is None
        archive.close()

    def test_empty_archive_query_by_task_type(self) -> None:
        archive = TraceArchive(":memory:")
        assert archive.query_by_task_type("anything") == []
        archive.close()

    def test_empty_archive_query_by_failure(self) -> None:
        archive = TraceArchive(":memory:")
        assert archive.query_by_failure() == []
        archive.close()

    def test_empty_archive_search_output(self) -> None:
        archive = TraceArchive(":memory:")
        assert archive.search_output("anything") == []
        archive.close()

    def test_empty_archive_count(self) -> None:
        archive = TraceArchive(":memory:")
        assert archive.count() == 0
        archive.close()

    def test_empty_archive_combinator_version(self) -> None:
        archive = TraceArchive(":memory:")
        assert archive.query_by_combinator_version("v1.0") == []
        archive.close()

    def test_store_with_empty_nodes_and_leaves(self) -> None:
        archive = TraceArchive(":memory:")
        trace = TreeTrace(
            trace_id="empty-nodes",
            task_type="test",
            k=1,
            depth=0,
            tau=1,
            cost_estimate=0.0,
            nodes=(),
            leaf_traces=(),
            final_output="nothing",
            timestamp=time.time(),
            execution_time_ms=0.0,
        )
        archive.store(trace)
        loaded = archive.load("empty-nodes")
        assert loaded is not None
        assert loaded.nodes == ()
        assert loaded.leaf_traces == ()
        archive.close()
