"""RepairIndex — tracks repair episodes (growth plates / kintsugi seams).

Links before and after traces to record how combinators evolve through
repair cycles triggered by disagreement signals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from rsocas.archive.trace_archive import TraceArchive


@dataclass(frozen=True)
class RepairEpisode:
    """A single repair episode linking before/after traces."""

    combinator_name: str
    before_trace_id: str
    after_trace_id: str
    trigger: str
    score_delta: float
    timestamp: float


class RepairIndex:
    """Index over repair episodes stored in TraceArchive's repairs table."""

    def __init__(self, archive: TraceArchive) -> None:
        self._archive = archive

    def record_repair(
        self,
        combinator_name: str,
        before_trace_id: str,
        after_trace_id: str,
        trigger: str,
        score_delta: float,
    ) -> None:
        """Record a repair episode: before and after traces linked."""
        cur = self._archive._conn.cursor()
        cur.execute(
            """INSERT INTO repairs
               (combinator_name, before_trace_id, after_trace_id,
                trigger, score_delta, timestamp)
               VALUES (?,?,?,?,?,?)""",
            (
                combinator_name,
                before_trace_id,
                after_trace_id,
                trigger,
                score_delta,
                time.time(),
            ),
        )
        self._archive._conn.commit()

    def query_repairs(
        self, combinator_name: str, limit: int = 10
    ) -> list[RepairEpisode]:
        """Return repair episodes for a combinator, newest first."""
        cur = self._archive._conn.cursor()
        cur.execute(
            """SELECT combinator_name, before_trace_id, after_trace_id,
                      trigger, score_delta, timestamp
               FROM repairs
               WHERE combinator_name = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (combinator_name, limit),
        )
        return [
            RepairEpisode(
                combinator_name=row[0],
                before_trace_id=row[1],
                after_trace_id=row[2],
                trigger=row[3],
                score_delta=row[4],
                timestamp=row[5],
            )
            for row in cur.fetchall()
        ]

    def query_recent_repairs(
        self, window_seconds: float = 86400.0
    ) -> list[RepairEpisode]:
        """All repairs in the last window (default 24 hours)."""
        cutoff = time.time() - window_seconds
        cur = self._archive._conn.cursor()
        cur.execute(
            """SELECT combinator_name, before_trace_id, after_trace_id,
                      trigger, score_delta, timestamp
               FROM repairs
               WHERE timestamp >= ?
               ORDER BY timestamp DESC""",
            (cutoff,),
        )
        return [
            RepairEpisode(
                combinator_name=row[0],
                before_trace_id=row[1],
                after_trace_id=row[2],
                trigger=row[3],
                score_delta=row[4],
                timestamp=row[5],
            )
            for row in cur.fetchall()
        ]
