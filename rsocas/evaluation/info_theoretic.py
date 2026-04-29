"""Information-theoretic evaluator: measures information preservation through the tree."""

from __future__ import annotations

import zlib

from rsocas.contracts.evaluation import EvalResult
from rsocas.contracts.traces import TreeTrace


def _compressed_size(text: str) -> int:
    """Return the size of the zlib-compressed representation of text."""
    if not text:
        return 0
    return len(zlib.compress(text.encode("utf-8")))


def _compression_ratio(text: str) -> float:
    """Return compression ratio: compressed_size / raw_size. Lower = more redundant."""
    if not text:
        return 0.0
    raw = len(text.encode("utf-8"))
    compressed = _compressed_size(text)
    return compressed / raw


class InformationTheoreticEval:
    """Measures information preservation through the composition tree.

    For each internal node (node with children), computes:
    - info_ratio = output_info / sum(child_info)
    where info = compressed size of the output text.

    Score = mean info_ratio across internal nodes, clamped to [0, 1].
    Leaf nodes get 1.0 (no information loss at leaves).
    """

    signal_type: str = "information_theoretic"

    def evaluate(
        self, trace: TreeTrace, ground_truth: str | None = None
    ) -> EvalResult:
        """Evaluate information preservation across the trace tree."""
        node_map: dict[str, object] = {node.id: node for node in trace.nodes}
        per_node_scores: dict[str, float] = {}
        internal_ratios: list[float] = []

        for node in trace.nodes:
            if not node.children:
                # Leaf node: no information loss
                per_node_scores[node.id] = 1.0
                continue

            # Internal node: compare output info to sum of children info
            output_info = _compressed_size(node.output)
            child_info_sum = 0
            for child_id in node.children:
                child_node = node_map.get(child_id)
                if child_node is not None:
                    child_info_sum += _compressed_size(child_node.output)

            if child_info_sum == 0:
                # Avoid division by zero; if children have no output, ratio is 1.0
                info_ratio = 1.0
            else:
                info_ratio = output_info / child_info_sum

            clamped = max(0.0, min(1.0, info_ratio))
            per_node_scores[node.id] = clamped
            internal_ratios.append(clamped)

        if internal_ratios:
            score = sum(internal_ratios) / len(internal_ratios)
        else:
            # No internal nodes — evaluate leaf response density instead.
            # A good leaf response should be informationally dense relative
            # to its input (high compression ratio = more unique content).
            leaf_scores = []
            for lt in trace.leaf_traces:
                if lt.response and lt.prompt:
                    resp_ratio = _compression_ratio(lt.response)
                    prompt_ratio = _compression_ratio(lt.prompt)
                    # Good: response has high info density relative to prompt
                    # Bad: response is very short/empty or just echoes prompt
                    density = min(1.0, resp_ratio / max(prompt_ratio, 0.01))
                    # Also penalize very short responses relative to prompt
                    length_ratio = min(1.0, len(lt.response) / max(len(lt.prompt) * 0.01, 1))
                    leaf_score = density * 0.7 + length_ratio * 0.3
                    leaf_scores.append(leaf_score)
                    per_node_scores[lt.node_id] = leaf_score
            score = sum(leaf_scores) / len(leaf_scores) if leaf_scores else 1.0

        score = max(0.0, min(1.0, score))

        return EvalResult(
            score=score,
            confidence=0.9,
            signal_type=self.signal_type,
            per_node_scores=per_node_scores,
            explanation=(
                f"Information preservation across {len(internal_ratios)} internal "
                f"node(s). Mean info_ratio={score:.3f}."
            ),
        )
