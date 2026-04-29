"""Goodhart-resistant evaluator: perturbation robustness check."""

from __future__ import annotations

import random
from typing import Callable

from rsocas.contracts.evaluation import EvalResult
from rsocas.contracts.traces import TreeTrace


_DISTRACTOR_SENTENCE = "The weather in Tokyo is pleasant today."


def _default_perturb(text: str) -> str:
    """Insert an irrelevant distractor sentence at a random position in the text."""
    if not text:
        return _DISTRACTOR_SENTENCE
    words = text.split()
    if len(words) <= 1:
        return f"{text} {_DISTRACTOR_SENTENCE}"
    # Use a seeded insertion point based on text length for reproducibility
    # within the same text, but varied across different texts
    rng = random.Random(len(text))
    insert_pos = rng.randint(0, len(words))
    words_copy = list(words)
    words_copy.insert(insert_pos, _DISTRACTOR_SENTENCE)
    return " ".join(words_copy)


def _response_stability(original: str, perturbed_response: str) -> float:
    """Measure how stable the response is under perturbation.

    Returns 1.0 if responses are identical, 0.0 if completely different.
    Uses Jaccard word similarity.
    """
    if not original and not perturbed_response:
        return 1.0
    if not original or not perturbed_response:
        return 0.0
    words_a = set(original.lower().split())
    words_b = set(perturbed_response.lower().split())
    if not words_a and not words_b:
        return 1.0
    intersection = words_a & words_b
    union = words_a | words_b
    if not union:
        return 0.0
    return len(intersection) / len(union)


class GoodhartResistantEval:
    """Perturbation robustness evaluator.

    For each leaf trace:
    - Perturb the input (insert distractor)
    - Re-run the leaf call via rerun_fn
    - Compare original response to perturbed response
    - If response changes significantly on irrelevant perturbation: fragile

    Score = fraction of leaves that are stable.
    Internal nodes get the mean of their children's scores.
    If rerun_fn is None, returns neutral result (score=0.5, confidence=0.0).
    """

    signal_type: str = "goodhart_resistant"

    def __init__(
        self,
        perturb_fn: Callable[[str], str] | None = None,
        rerun_fn: Callable[[str], str] | None = None,
    ):
        """Initialize with optional perturbation and rerun functions.

        perturb_fn: (text) -> perturbed_text. Default: insert irrelevant sentence.
        rerun_fn: (prompt) -> response. If None, evaluator is disabled.
        """
        self._perturb_fn = perturb_fn or _default_perturb
        self._rerun_fn = rerun_fn

    def evaluate(
        self, trace: TreeTrace, ground_truth: str | None = None
    ) -> EvalResult:
        """Evaluate perturbation robustness across the trace tree."""
        if self._rerun_fn is None:
            return EvalResult(
                score=0.5,
                confidence=0.0,
                signal_type=self.signal_type,
                per_node_scores={},
                explanation="Evaluator disabled: no rerun_fn provided.",
            )

        # Map leaf traces by node_id
        leaf_map: dict[str, object] = {lt.node_id: lt for lt in trace.leaf_traces}
        node_map: dict[str, object] = {node.id: node for node in trace.nodes}

        per_node_scores: dict[str, float] = {}
        leaf_stability_scores: list[float] = []

        # First pass: score all leaf nodes
        for leaf_trace in trace.leaf_traces:
            perturbed_prompt = self._perturb_fn(leaf_trace.prompt)
            perturbed_response = self._rerun_fn(perturbed_prompt)
            stability = _response_stability(
                leaf_trace.response, perturbed_response
            )
            per_node_scores[leaf_trace.node_id] = stability
            leaf_stability_scores.append(stability)

        # Second pass: score internal nodes as mean of children
        # Process nodes from deepest to shallowest
        sorted_nodes = sorted(trace.nodes, key=lambda n: -n.depth)
        for node in sorted_nodes:
            if node.id in per_node_scores:
                continue
            if not node.children:
                # Leaf node without a leaf_trace; give neutral score
                per_node_scores[node.id] = 0.5
                continue
            child_scores = [
                per_node_scores[cid]
                for cid in node.children
                if cid in per_node_scores
            ]
            if child_scores:
                per_node_scores[node.id] = sum(child_scores) / len(child_scores)
            else:
                per_node_scores[node.id] = 0.5

        if leaf_stability_scores:
            # Score = fraction of leaves that are stable (> 0.7 similarity)
            stable_count = sum(
                1 for s in leaf_stability_scores if s > 0.7
            )
            score = stable_count / len(leaf_stability_scores)
        else:
            score = 0.5

        return EvalResult(
            score=score,
            confidence=0.7,
            signal_type=self.signal_type,
            per_node_scores=per_node_scores,
            explanation=(
                f"Perturbation robustness across {len(leaf_stability_scores)} leaf "
                f"node(s). {score:.1%} are stable under irrelevant perturbation."
            ),
        )
