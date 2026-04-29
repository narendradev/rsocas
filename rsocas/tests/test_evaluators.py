"""Tests for the three contrapuntal evaluators."""

from __future__ import annotations

import pytest

from rsocas.contracts.evaluation import EvalResult
from rsocas.contracts.traces import LeafTrace, NodeTrace, TreeTrace
from rsocas.evaluation.boundary_detection import BoundaryDetectionEval
from rsocas.evaluation.goodhart_resistant import GoodhartResistantEval
from rsocas.evaluation.info_theoretic import InformationTheoreticEval


def make_trace(
    nodes: tuple[NodeTrace, ...],
    leaves: tuple[LeafTrace, ...],
    final_output: str,
    trace_id: str = "test-trace",
) -> TreeTrace:
    """Build a TreeTrace with given data for testing."""
    return TreeTrace(
        trace_id=trace_id,
        task_type="test",
        k=len(nodes),
        depth=max((n.depth for n in nodes), default=0),
        tau=1,
        cost_estimate=0.0,
        nodes=nodes,
        leaf_traces=leaves,
        final_output=final_output,
        timestamp=0.0,
        execution_time_ms=0.0,
    )


def _make_diverse_text(seed: str, length: int = 500) -> str:
    """Generate a long diverse text that compresses poorly (high information)."""
    words = [
        "quantum", "entropy", "divergence", "topology", "manifold",
        "gradient", "stochastic", "orthogonal", "eigenvalue", "isomorphism",
        "category", "functor", "monad", "lattice", "algebra",
        "hypothesis", "theorem", "conjecture", "lemma", "corollary",
    ]
    result: list[str] = [seed]
    idx = 0
    while len(" ".join(result)) < length:
        result.append(words[idx % len(words)])
        idx += 1
        # Mix in numbers for diversity
        if idx % 3 == 0:
            result.append(str(idx * 17 + 42))
    return " ".join(result)


def _make_repetitive_text(word: str = "hello", repeats: int = 100) -> str:
    """Generate a repetitive text that compresses well (low information)."""
    return " ".join([word] * repeats)


# ---------------------------------------------------------------------------
# InformationTheoreticEval tests
# ---------------------------------------------------------------------------


class TestInformationTheoreticEval:
    """Tests for information preservation evaluator."""

    def test_high_info_preservation(self) -> None:
        """Trace with high information preservation should score > 0.5.

        When a parent node preserves most child information, the compressed size
        of the parent output should be a substantial fraction of the sum of
        children's compressed sizes. Due to zlib cross-text deduplication, the
        ratio is typically ~0.5-0.7 even for full concatenation, so we check > 0.5.
        """
        child_a_text = _make_diverse_text("alpha", 300)
        child_b_text = _make_diverse_text("beta", 300)
        # Parent output combines both children's information
        parent_text = f"{child_a_text} Furthermore, {child_b_text}"

        nodes = (
            NodeTrace(
                id="root",
                depth=0,
                position=0,
                combinator="merge",
                input_size=len(parent_text),
                output=parent_text,
                children=("child_a", "child_b"),
            ),
            NodeTrace(
                id="child_a",
                depth=1,
                position=0,
                combinator="leaf",
                input_size=len(child_a_text),
                output=child_a_text,
            ),
            NodeTrace(
                id="child_b",
                depth=1,
                position=1,
                combinator="leaf",
                input_size=len(child_b_text),
                output=child_b_text,
            ),
        )
        trace = make_trace(nodes, (), parent_text)

        evaluator = InformationTheoreticEval()
        result = evaluator.evaluate(trace)

        assert result.score > 0.5, f"Expected score > 0.5, got {result.score}"
        assert result.signal_type == "information_theoretic"
        assert result.confidence > 0
        # Leaf nodes should have score 1.0
        assert result.per_node_scores["child_a"] == 1.0
        assert result.per_node_scores["child_b"] == 1.0

    def test_info_loss(self) -> None:
        """Trace with info loss (short generic output from long input) should score < 0.4."""
        child_a_text = _make_diverse_text("alpha", 500)
        child_b_text = _make_diverse_text("beta", 500)
        # Parent output is very short -- massive info loss
        parent_text = "Summary: things happened."

        nodes = (
            NodeTrace(
                id="root",
                depth=0,
                position=0,
                combinator="merge",
                input_size=len(parent_text),
                output=parent_text,
                children=("child_a", "child_b"),
            ),
            NodeTrace(
                id="child_a",
                depth=1,
                position=0,
                combinator="leaf",
                input_size=len(child_a_text),
                output=child_a_text,
            ),
            NodeTrace(
                id="child_b",
                depth=1,
                position=1,
                combinator="leaf",
                input_size=len(child_b_text),
                output=child_b_text,
            ),
        )
        trace = make_trace(nodes, (), parent_text)

        evaluator = InformationTheoreticEval()
        result = evaluator.evaluate(trace)

        assert result.score < 0.4, f"Expected score < 0.4, got {result.score}"
        assert result.signal_type == "information_theoretic"

    def test_leaf_only_trace(self) -> None:
        """Trace with only leaf nodes should score 1.0."""
        nodes = (
            NodeTrace(
                id="leaf_only",
                depth=0,
                position=0,
                combinator="leaf",
                input_size=100,
                output="Some output text here.",
            ),
        )
        trace = make_trace(nodes, (), "Some output text here.")

        evaluator = InformationTheoreticEval()
        result = evaluator.evaluate(trace)

        assert result.score == 1.0
        assert result.per_node_scores["leaf_only"] == 1.0


# ---------------------------------------------------------------------------
# BoundaryDetectionEval tests
# ---------------------------------------------------------------------------


class TestBoundaryDetectionEval:
    """Tests for boundary detection (echo detection) evaluator."""

    def test_output_copies_child_verbatim(self) -> None:
        """When output copies one child verbatim, score should be < 0.2."""
        child_text = "This is the child output with detailed analysis and findings."
        nodes = (
            NodeTrace(
                id="root",
                depth=0,
                position=0,
                combinator="merge",
                input_size=len(child_text),
                output=child_text,  # Exact copy!
                children=("child_a",),
            ),
            NodeTrace(
                id="child_a",
                depth=1,
                position=0,
                combinator="leaf",
                input_size=len(child_text),
                output=child_text,
            ),
        )
        trace = make_trace(nodes, (), child_text)

        evaluator = BoundaryDetectionEval()
        result = evaluator.evaluate(trace)

        assert result.score < 0.2, f"Expected score < 0.2, got {result.score}"
        assert result.signal_type == "boundary"

    def test_output_synthesizes_children(self) -> None:
        """When output synthesizes from children, score should be > 0.7."""
        child_a_text = (
            "The quantum computing approach uses qubits and superposition "
            "to perform parallel computations on exponentially large state spaces."
        )
        child_b_text = (
            "Traditional classical computers use binary transistors with "
            "deterministic logic gates for sequential processing of data."
        )
        # Synthesized output with different vocabulary and structure
        parent_text = (
            "Comparing computational paradigms reveals fundamental differences: "
            "whereas one leverages probabilistic quantum states for massive parallelism, "
            "the other relies on deterministic binary operations executed sequentially. "
            "Each approach has distinct advantages for different problem classes."
        )
        nodes = (
            NodeTrace(
                id="root",
                depth=0,
                position=0,
                combinator="merge",
                input_size=len(parent_text),
                output=parent_text,
                children=("child_a", "child_b"),
            ),
            NodeTrace(
                id="child_a",
                depth=1,
                position=0,
                combinator="leaf",
                input_size=len(child_a_text),
                output=child_a_text,
            ),
            NodeTrace(
                id="child_b",
                depth=1,
                position=1,
                combinator="leaf",
                input_size=len(child_b_text),
                output=child_b_text,
            ),
        )
        trace = make_trace(nodes, (), parent_text)

        evaluator = BoundaryDetectionEval()
        result = evaluator.evaluate(trace)

        assert result.score > 0.7, f"Expected score > 0.7, got {result.score}"
        assert result.signal_type == "boundary"
        # Leaf nodes should have 1.0
        assert result.per_node_scores["child_a"] == 1.0
        assert result.per_node_scores["child_b"] == 1.0

    def test_custom_similarity_fn(self) -> None:
        """Custom similarity function should be used when provided."""
        call_count = 0

        def mock_similarity(text_a: str, text_b: str) -> float:
            nonlocal call_count
            call_count += 1
            return 0.5

        nodes = (
            NodeTrace(
                id="root", depth=0, position=0, combinator="merge",
                input_size=10, output="parent", children=("child",),
            ),
            NodeTrace(
                id="child", depth=1, position=0, combinator="leaf",
                input_size=10, output="child",
            ),
        )
        trace = make_trace(nodes, (), "parent")

        evaluator = BoundaryDetectionEval(similarity_fn=mock_similarity)
        result = evaluator.evaluate(trace)

        assert call_count > 0, "Custom similarity fn should have been called"
        assert result.per_node_scores["root"] == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# GoodhartResistantEval tests
# ---------------------------------------------------------------------------


class TestGoodhartResistantEval:
    """Tests for perturbation robustness evaluator."""

    def test_no_rerun_fn_returns_neutral(self) -> None:
        """With no rerun_fn, should return score=0.5, confidence=0.0."""
        nodes = (
            NodeTrace(
                id="leaf", depth=0, position=0, combinator="leaf",
                input_size=10, output="output",
            ),
        )
        leaves = (
            LeafTrace(
                node_id="leaf", prompt="test prompt", response="test response",
            ),
        )
        trace = make_trace(nodes, leaves, "output")

        evaluator = GoodhartResistantEval(rerun_fn=None)
        result = evaluator.evaluate(trace)

        assert result.score == 0.5
        assert result.confidence == 0.0
        assert result.signal_type == "goodhart_resistant"

    def test_stable_rerun_returns_high_score(self) -> None:
        """Mock rerun_fn that returns same answer should yield score=1.0."""

        def stable_rerun(prompt: str) -> str:
            return "consistent response regardless of input"

        nodes = (
            NodeTrace(
                id="leaf1", depth=0, position=0, combinator="leaf",
                input_size=10, output="consistent response regardless of input",
            ),
        )
        leaves = (
            LeafTrace(
                node_id="leaf1",
                prompt="What is the capital of France?",
                response="consistent response regardless of input",
            ),
        )
        trace = make_trace(nodes, leaves, "consistent response regardless of input")

        evaluator = GoodhartResistantEval(rerun_fn=stable_rerun)
        result = evaluator.evaluate(trace)

        assert result.score == 1.0, f"Expected score=1.0, got {result.score}"
        assert result.signal_type == "goodhart_resistant"

    def test_unstable_rerun_returns_low_score(self) -> None:
        """Mock rerun_fn returning very different answer should yield score < 0.5."""
        call_counter = 0

        def unstable_rerun(prompt: str) -> str:
            nonlocal call_counter
            call_counter += 1
            # Return completely different text each time
            return f"completely different response number {call_counter} with unique words"

        original_response = (
            "The capital of France is Paris, a major European city known for "
            "the Eiffel Tower and rich cultural heritage dating back centuries."
        )
        nodes = (
            NodeTrace(
                id="leaf1", depth=0, position=0, combinator="leaf",
                input_size=50, output=original_response,
            ),
        )
        leaves = (
            LeafTrace(
                node_id="leaf1",
                prompt="What is the capital of France?",
                response=original_response,
            ),
        )
        trace = make_trace(nodes, leaves, original_response)

        evaluator = GoodhartResistantEval(rerun_fn=unstable_rerun)
        result = evaluator.evaluate(trace)

        assert result.score < 0.5, f"Expected score < 0.5, got {result.score}"

    def test_internal_nodes_get_child_mean(self) -> None:
        """Internal nodes should receive mean of their children's scores."""

        def stable_rerun(prompt: str) -> str:
            return "stable answer for everything"

        nodes = (
            NodeTrace(
                id="root", depth=0, position=0, combinator="merge",
                input_size=30, output="merged output",
                children=("leaf1", "leaf2"),
            ),
            NodeTrace(
                id="leaf1", depth=1, position=0, combinator="leaf",
                input_size=10, output="stable answer for everything",
            ),
            NodeTrace(
                id="leaf2", depth=1, position=1, combinator="leaf",
                input_size=10, output="stable answer for everything",
            ),
        )
        leaves = (
            LeafTrace(
                node_id="leaf1",
                prompt="Question A?",
                response="stable answer for everything",
            ),
            LeafTrace(
                node_id="leaf2",
                prompt="Question B?",
                response="stable answer for everything",
            ),
        )
        trace = make_trace(nodes, leaves, "merged output")

        evaluator = GoodhartResistantEval(rerun_fn=stable_rerun)
        result = evaluator.evaluate(trace)

        assert "root" in result.per_node_scores
        # Root should be mean of children
        root_score = result.per_node_scores["root"]
        leaf1_score = result.per_node_scores["leaf1"]
        leaf2_score = result.per_node_scores["leaf2"]
        expected_root = (leaf1_score + leaf2_score) / 2
        assert root_score == pytest.approx(expected_root, abs=0.01)
