"""Boundary detection evaluator: detects self-referential loops (system echoing itself)."""

from __future__ import annotations

from typing import Callable

from rsocas.contracts.evaluation import EvalResult
from rsocas.contracts.traces import TreeTrace


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard word similarity between two texts.

    Returns set intersection / set union on whitespace-tokenized words.
    """
    if not text_a or not text_b:
        return 0.0
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    intersection = words_a & words_b
    union = words_a | words_b
    if not union:
        return 0.0
    return len(intersection) / len(union)


class BoundaryDetectionEval:
    """Detects when a node's output simply echoes one of its children.

    For each internal node, compares its output to each child's output.
    If max_similarity > 0.95: echoing (just copying one child's output).

    Score = 1.0 - max_echo_ratio across all internal nodes.
    Per-node: each internal node gets 1.0 - its echo_ratio.
    Leaf nodes get 1.0.
    """

    signal_type: str = "boundary"

    def __init__(self, similarity_fn: Callable[[str, str], float] | None = None):
        """Initialize with optional custom similarity function.

        If similarity_fn is None, uses Jaccard word similarity (no dependencies).
        Otherwise accepts a function(text_a, text_b) -> float.
        """
        self._similarity_fn = similarity_fn or _jaccard_similarity

    def evaluate(
        self, trace: TreeTrace, ground_truth: str | None = None
    ) -> EvalResult:
        """Evaluate boundary integrity across the trace tree."""
        node_map: dict[str, object] = {node.id: node for node in trace.nodes}
        per_node_scores: dict[str, float] = {}
        max_echo_ratio: float = 0.0
        internal_count: int = 0

        for node in trace.nodes:
            if not node.children:
                # Leaf node: no echoing possible
                per_node_scores[node.id] = 1.0
                continue

            internal_count += 1
            # Find max similarity between this node's output and any child output
            node_max_sim = 0.0
            for child_id in node.children:
                child_node = node_map.get(child_id)
                if child_node is not None:
                    sim = self._similarity_fn(node.output, child_node.output)
                    node_max_sim = max(node_max_sim, sim)

            echo_ratio = node_max_sim if node_max_sim > 0.95 else node_max_sim
            per_node_scores[node.id] = max(0.0, min(1.0, 1.0 - echo_ratio))
            max_echo_ratio = max(max_echo_ratio, echo_ratio)

        if internal_count > 0:
            score = max(0.0, min(1.0, 1.0 - max_echo_ratio))
        else:
            # No internal nodes — evaluate leaf response vs prompt similarity.
            # A good response should NOT echo the prompt verbatim.
            # It should synthesize/answer, not parrot.
            leaf_scores = []
            for lt in trace.leaf_traces:
                if lt.response and lt.prompt:
                    echo = self._similarity_fn(lt.response, lt.prompt)
                    leaf_score = max(0.0, min(1.0, 1.0 - echo))
                    leaf_scores.append(leaf_score)
                    per_node_scores[lt.node_id] = leaf_score
            score = sum(leaf_scores) / len(leaf_scores) if leaf_scores else 1.0
            max_echo_ratio = 1.0 - score

        return EvalResult(
            score=score,
            confidence=0.85 if internal_count > 0 else 0.7,
            signal_type=self.signal_type,
            per_node_scores=per_node_scores,
            explanation=(
                f"Boundary detection across {internal_count} internal node(s), "
                f"{len(trace.leaf_traces)} leaf(s). "
                f"Max echo ratio={max_echo_ratio:.3f}, score={score:.3f}."
            ),
        )
