"""DistributionTracker — running distribution statistics for staleness detection.

Computes ValidationSnapshot from recent traces to detect when a combinator's
performance distribution has drifted enough to trigger dissolution.
"""

from __future__ import annotations

import json
import math
import time

from rsocas.archive.trace_archive import TraceArchive
from rsocas.contracts.combinators import ValidationSnapshot


class DistributionTracker:
    """Computes distribution snapshots from stored traces."""

    def __init__(self, archive: TraceArchive) -> None:
        self._archive = archive

    def compute_snapshot(
        self, task_type: str, window_seconds: float = 3600.0
    ) -> ValidationSnapshot:
        """Compute a ValidationSnapshot from recent traces.

        Aggregates traces of the given task_type within the time window:
        - task_types: (task_type,)
        - input_size_range: (min, max) of node input_sizes across traces
        - n_samples: count of traces in window
        - mean_score: mean of final_score values (0.0 if none have scores)
        - score_std: population standard deviation of final_scores
        - timestamp: current time
        """
        cutoff = time.time() - window_seconds
        cur = self._archive._conn.cursor()
        cur.execute(
            """SELECT final_score, nodes_json
               FROM traces
               WHERE task_type = ? AND timestamp >= ?
               ORDER BY timestamp DESC""",
            (task_type, cutoff),
        )
        rows = cur.fetchall()

        scores: list[float] = []
        min_input_size = 0
        max_input_size = 0
        has_nodes = False

        for row in rows:
            final_score = row[0]
            if final_score is not None:
                scores.append(final_score)

            nodes_data = json.loads(row[1])
            for node in nodes_data:
                size = node.get("input_size", 0)
                if not has_nodes:
                    min_input_size = size
                    max_input_size = size
                    has_nodes = True
                else:
                    min_input_size = min(min_input_size, size)
                    max_input_size = max(max_input_size, size)

        n_samples = len(rows)
        mean_score = 0.0
        score_std = 0.0

        if scores:
            mean_score = sum(scores) / len(scores)
            if len(scores) > 1:
                variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
                score_std = math.sqrt(variance)

        return ValidationSnapshot(
            task_types=(task_type,),
            input_size_range=(min_input_size, max_input_size),
            n_samples=n_samples,
            mean_score=mean_score,
            score_std=score_std,
            timestamp=time.time(),
        )
