"""Disagreement computation: pure function computing disagreement from evaluator results."""

from __future__ import annotations

import math

from rsocas.contracts.evaluation import DisagreementSignal, EvalResult


def compute_disagreement(
    results: tuple[EvalResult, ...],
    threshold: float = 0.3,
    timestamp: float = 0.0,
) -> DisagreementSignal:
    """Pure function computing disagreement across evaluator results.

    No side effects. Deterministic for same inputs.

    magnitude = max pairwise |score_a - score_b| across all evaluator pairs
    pairwise = dict of "typeA_vs_typeB" -> |score_a - score_b|
    per_node = for each node_id, compute variance of per_node_scores across evaluators
    outlier_voice = signal_type of evaluator whose score differs most from mean
    should_surface = magnitude >= threshold
    """
    if not results:
        return DisagreementSignal(
            magnitude=0.0,
            pairwise={},
            per_node={},
            outlier_voice=None,
            should_surface=False,
            timestamp=timestamp,
        )

    if len(results) == 1:
        return DisagreementSignal(
            magnitude=0.0,
            pairwise={},
            per_node={},
            outlier_voice=None,
            should_surface=False,
            timestamp=timestamp,
        )

    # Compute pairwise differences
    pairwise: dict[str, float] = {}
    magnitude: float = 0.0

    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            key = f"{results[i].signal_type}_vs_{results[j].signal_type}"
            diff = abs(results[i].score - results[j].score)
            pairwise[key] = diff
            magnitude = max(magnitude, diff)

    # Find outlier: evaluator whose score differs most from mean
    mean_score = sum(r.score for r in results) / len(results)
    max_deviation = 0.0
    outlier_voice: str | None = None

    for result in results:
        deviation = abs(result.score - mean_score)
        if deviation > max_deviation:
            max_deviation = deviation
            outlier_voice = result.signal_type

    # Compute per-node disagreement (variance across evaluators for each node)
    all_node_ids: set[str] = set()
    for result in results:
        all_node_ids.update(result.per_node_scores.keys())

    per_node: dict[str, float] = {}
    for node_id in all_node_ids:
        scores_for_node: list[float] = []
        for result in results:
            if node_id in result.per_node_scores:
                scores_for_node.append(result.per_node_scores[node_id])

        if len(scores_for_node) < 2:
            per_node[node_id] = 0.0
            continue

        node_mean = sum(scores_for_node) / len(scores_for_node)
        variance = sum(
            (s - node_mean) ** 2 for s in scores_for_node
        ) / len(scores_for_node)
        per_node[node_id] = variance

    should_surface = magnitude >= threshold

    return DisagreementSignal(
        magnitude=magnitude,
        pairwise=pairwise,
        per_node=per_node,
        outlier_voice=outlier_voice,
        should_surface=should_surface,
        timestamp=timestamp,
    )
