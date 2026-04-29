"""Tests for the GEPA tree adapter — all synthetic data, no real GEPA needed."""

from __future__ import annotations

import pytest

from rsocas.adapters.gepa_tree_adapter import TreeTraceGEPAAdapter
from rsocas.contracts.evaluation import DisagreementSignal
from rsocas.contracts.traces import LeafTrace, NodeTrace, TreeTrace


# ---------------------------------------------------------------------------
# Fixtures — synthetic traces
# ---------------------------------------------------------------------------


def _make_trace(
    k: int = 2,
    depth: int = 1,
    task_type: str = "qa",
) -> TreeTrace:
    """Build a small synthetic tree trace with k leaves and one root."""
    leaves: list[NodeTrace] = []
    leaf_traces: list[LeafTrace] = []

    for i in range(k):
        nid = f"leaf_{i}"
        leaves.append(
            NodeTrace(
                id=nid,
                depth=depth,
                position=i,
                combinator="leaf",
                input_size=100 + i * 10,
                output=f"leaf_output_{i}",
                children=(),
                llm_calls=1,
                latency_ms=50.0,
            )
        )
        leaf_traces.append(
            LeafTrace(
                node_id=nid,
                prompt=f"Summarize chunk {i}: " + "x" * 200,
                response=f"leaf_output_{i}",
                tokens_in=50,
                tokens_out=30,
                model="test-model",
            )
        )

    root = NodeTrace(
        id="root_0",
        depth=0,
        position=0,
        combinator="reduce",
        input_size=k,
        output="final combined output",
        children=tuple(f"leaf_{i}" for i in range(k)),
        llm_calls=1,
        latency_ms=100.0,
    )

    all_nodes = tuple(leaves) + (root,)
    return TreeTrace(
        trace_id="trace_001",
        task_type=task_type,
        k=k,
        depth=depth,
        tau=512,
        cost_estimate=0.05,
        nodes=all_nodes,
        leaf_traces=tuple(leaf_traces),
        final_output="final combined output",
        timestamp=1000.0,
        execution_time_ms=200.0,
        total_llm_calls=k + 1,
        total_tokens=(50 + 30) * k,
    )


def _make_disagreement(
    per_node: dict[str, float] | None = None,
    magnitude: float = 0.5,
) -> DisagreementSignal:
    """Build a synthetic disagreement signal."""
    return DisagreementSignal(
        magnitude=magnitude,
        pairwise={"eval_a_vs_eval_b": magnitude},
        per_node=per_node or {},
        outlier_voice="eval_a",
        should_surface=magnitude >= 0.3,
        timestamp=1000.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIdentifyFailingNodes:
    def test_identify_failing_nodes(self) -> None:
        """Trace with per-node disagreement returns highest-disagreement nodes."""
        adapter = TreeTraceGEPAAdapter()
        trace = _make_trace(k=3, depth=1)
        disagreement = _make_disagreement(
            per_node={"leaf_0": 0.8, "leaf_1": 0.2, "leaf_2": 0.6}
        )

        result = adapter.identify_failing_nodes(trace, disagreement)

        assert result == ["leaf_0", "leaf_2", "leaf_1"]
        assert result[0] == "leaf_0"  # highest disagreement first

    def test_identify_failing_nodes_no_disagreement(self) -> None:
        """Empty per_node returns empty list."""
        adapter = TreeTraceGEPAAdapter()
        trace = _make_trace()
        disagreement = _make_disagreement(per_node={})

        result = adapter.identify_failing_nodes(trace, disagreement)

        assert result == []

    def test_identify_failing_nodes_zero_scores_excluded(self) -> None:
        """Nodes with zero disagreement score are excluded."""
        adapter = TreeTraceGEPAAdapter()
        trace = _make_trace(k=2)
        disagreement = _make_disagreement(
            per_node={"leaf_0": 0.5, "leaf_1": 0.0}
        )

        result = adapter.identify_failing_nodes(trace, disagreement)

        assert result == ["leaf_0"]


class TestExtractSubtreeContext:
    def test_extract_subtree_context_includes_siblings(self) -> None:
        """Context for a leaf mentions sibling outputs."""
        adapter = TreeTraceGEPAAdapter()
        trace = _make_trace(k=3, depth=1)

        context = adapter.extract_subtree_context(trace, "leaf_0")

        assert "leaf_0" in context
        assert "depth [1]" in context
        assert "Sibling nodes produced:" in context
        # Should mention other leaf outputs
        assert "leaf_output_1" in context or "leaf_output_2" in context

    def test_extract_subtree_context_leaf_node(self) -> None:
        """Context for a leaf includes prompt excerpt."""
        adapter = TreeTraceGEPAAdapter()
        trace = _make_trace(k=2, depth=1)

        context = adapter.extract_subtree_context(trace, "leaf_0")

        assert "This node received:" in context
        assert "Summarize chunk 0" in context
        assert "Combinator: [leaf]" in context

    def test_extract_subtree_context_includes_parent(self) -> None:
        """Context mentions parent combinator."""
        adapter = TreeTraceGEPAAdapter()
        trace = _make_trace(k=2, depth=1)

        context = adapter.extract_subtree_context(trace, "leaf_0")

        assert "Parent node used [reduce] to combine results." in context

    def test_extract_subtree_context_missing_node(self) -> None:
        """Missing node returns 'not found' message."""
        adapter = TreeTraceGEPAAdapter()
        trace = _make_trace()

        context = adapter.extract_subtree_context(trace, "nonexistent")

        assert "not found" in context


class TestMakeReflectiveDataset:
    def test_make_reflective_dataset_format(self) -> None:
        """Verify output matches GEPA's expected format."""
        adapter = TreeTraceGEPAAdapter()
        trace = _make_trace(k=2, depth=1)
        disagreement = _make_disagreement(
            per_node={"leaf_0": 0.7, "leaf_1": 0.3}
        )

        dataset = adapter.make_reflective_dataset(trace, disagreement)

        assert "leaf_prompt" in dataset
        entries = dataset["leaf_prompt"]
        assert len(entries) == 2

        entry = entries[0]
        assert "Inputs" in entry
        assert "Generated Outputs" in entry
        assert "Feedback" in entry
        assert "context" in entry["Inputs"]
        assert "answer" in entry["Generated Outputs"]

    def test_make_reflective_dataset_custom_component(self) -> None:
        """Custom component name is used as the key."""
        adapter = TreeTraceGEPAAdapter()
        trace = _make_trace(k=2, depth=1)
        disagreement = _make_disagreement(per_node={"leaf_0": 0.5})

        dataset = adapter.make_reflective_dataset(
            trace, disagreement, component_name="my_module"
        )

        assert "my_module" in dataset
        assert "leaf_prompt" not in dataset

    def test_make_reflective_dataset_empty_on_no_failures(self) -> None:
        """No failing nodes produces empty dataset."""
        adapter = TreeTraceGEPAAdapter()
        trace = _make_trace()
        disagreement = _make_disagreement(per_node={})

        dataset = adapter.make_reflective_dataset(trace, disagreement)

        assert dataset == {"leaf_prompt": []}


class TestAdaptForGEPAOptimize:
    def test_adapt_for_gepa_optimize_batch(self) -> None:
        """Multiple traces are aggregated into a single dataset."""
        adapter = TreeTraceGEPAAdapter()

        trace1 = _make_trace(k=2, depth=1)
        dis1 = _make_disagreement(per_node={"leaf_0": 0.8})

        trace2 = _make_trace(k=3, depth=1)
        dis2 = _make_disagreement(per_node={"leaf_1": 0.6, "leaf_2": 0.4})

        traces = [
            (trace1, dis1, "ground truth 1"),
            (trace2, dis2, None),
        ]

        dataset = adapter.adapt_for_gepa_optimize(traces)

        assert "leaf_prompt" in dataset
        entries = dataset["leaf_prompt"]
        # trace1 contributes 1, trace2 contributes 2
        assert len(entries) == 3

    def test_adapt_for_gepa_optimize_empty(self) -> None:
        """Empty traces list produces empty dataset."""
        adapter = TreeTraceGEPAAdapter()

        dataset = adapter.adapt_for_gepa_optimize([])

        assert dataset == {"leaf_prompt": []}
