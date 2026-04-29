"""TraceArchive — SQLite + FTS5 persistent storage for traces.

Stores TreeTrace, EvalResult, DisagreementSignal, and repair episodes
with full-text search on final_output via FTS5.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict

from rsocas.contracts.combinators import ValidationSnapshot
from rsocas.contracts.evaluation import DisagreementSignal, EvalResult
from rsocas.contracts.traces import LeafTrace, NodeTrace, TreeTrace


def _node_to_dict(node: NodeTrace) -> dict:
    return asdict(node)


def _leaf_to_dict(leaf: LeafTrace) -> dict:
    return asdict(leaf)


def _dict_to_node(d: dict) -> NodeTrace:
    return NodeTrace(
        id=d["id"],
        depth=d["depth"],
        position=d["position"],
        combinator=d["combinator"],
        input_size=d["input_size"],
        output=d["output"],
        children=tuple(d.get("children", ())),
        llm_calls=d.get("llm_calls", 0),
        latency_ms=d.get("latency_ms", 0.0),
        score=d.get("score"),
    )


def _dict_to_leaf(d: dict) -> LeafTrace:
    return LeafTrace(
        node_id=d["node_id"],
        prompt=d["prompt"],
        response=d["response"],
        tokens_in=d.get("tokens_in", 0),
        tokens_out=d.get("tokens_out", 0),
        model=d.get("model", ""),
        confidence=d.get("confidence"),
    )


class TraceArchive:
    """SQLite-backed archive for TreeTrace and related data."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                k INTEGER NOT NULL,
                depth INTEGER NOT NULL,
                tau INTEGER NOT NULL,
                cost_estimate REAL NOT NULL,
                final_output TEXT NOT NULL,
                final_score REAL,
                timestamp REAL NOT NULL,
                execution_time_ms REAL NOT NULL,
                total_llm_calls INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                nodes_json TEXT NOT NULL,
                leaf_traces_json TEXT NOT NULL,
                combinator_versions_json TEXT NOT NULL DEFAULT '{}',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                score REAL NOT NULL,
                confidence REAL NOT NULL,
                per_node_json TEXT NOT NULL DEFAULT '{}',
                explanation TEXT NOT NULL DEFAULT '',
                FOREIGN KEY(trace_id) REFERENCES traces(trace_id)
            );

            CREATE TABLE IF NOT EXISTS disagreements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                magnitude REAL NOT NULL,
                pairwise_json TEXT NOT NULL DEFAULT '{}',
                per_node_json TEXT NOT NULL DEFAULT '{}',
                outlier_voice TEXT,
                should_surface INTEGER NOT NULL DEFAULT 0,
                timestamp REAL NOT NULL,
                FOREIGN KEY(trace_id) REFERENCES traces(trace_id)
            );

            CREATE TABLE IF NOT EXISTS repairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                combinator_name TEXT NOT NULL,
                before_trace_id TEXT NOT NULL,
                after_trace_id TEXT NOT NULL,
                trigger TEXT NOT NULL,
                score_delta REAL NOT NULL,
                timestamp REAL NOT NULL
            );
        """)

        # FTS5 virtual table — created separately since IF NOT EXISTS
        # is not supported for virtual tables in older SQLite versions.
        try:
            cur.execute("""
                CREATE VIRTUAL TABLE trace_fts USING fts5(
                    trace_id, final_output, task_type,
                    content='traces',
                    content_rowid='rowid'
                );
            """)
        except sqlite3.OperationalError:
            pass  # already exists

        self._conn.commit()

    def store(
        self,
        trace: TreeTrace,
        evaluations: tuple[EvalResult, ...] = (),
        disagreement: DisagreementSignal | None = None,
    ) -> str:
        """Store a trace with optional evaluations and disagreement signal.

        Returns the trace_id.
        """
        cur = self._conn.cursor()

        nodes_json = json.dumps([_node_to_dict(n) for n in trace.nodes])
        leaf_json = json.dumps([_leaf_to_dict(lf) for lf in trace.leaf_traces])
        cv_json = json.dumps(trace.combinator_versions)
        meta_json = json.dumps(trace.metadata)

        cur.execute(
            """INSERT OR REPLACE INTO traces
               (trace_id, task_type, k, depth, tau, cost_estimate,
                final_output, final_score, timestamp, execution_time_ms,
                total_llm_calls, total_tokens, nodes_json, leaf_traces_json,
                combinator_versions_json, metadata_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                trace.trace_id,
                trace.task_type,
                trace.k,
                trace.depth,
                trace.tau,
                trace.cost_estimate,
                trace.final_output,
                trace.final_score,
                trace.timestamp,
                trace.execution_time_ms,
                trace.total_llm_calls,
                trace.total_tokens,
                nodes_json,
                leaf_json,
                cv_json,
                meta_json,
            ),
        )

        # Update FTS index
        cur.execute(
            "INSERT INTO trace_fts (trace_id, final_output, task_type) VALUES (?,?,?)",
            (trace.trace_id, trace.final_output, trace.task_type),
        )

        for ev in evaluations:
            cur.execute(
                """INSERT INTO evaluations
                   (trace_id, signal_type, score, confidence,
                    per_node_json, explanation)
                   VALUES (?,?,?,?,?,?)""",
                (
                    trace.trace_id,
                    ev.signal_type,
                    ev.score,
                    ev.confidence,
                    json.dumps(ev.per_node_scores),
                    ev.explanation,
                ),
            )

        if disagreement is not None:
            cur.execute(
                """INSERT INTO disagreements
                   (trace_id, magnitude, pairwise_json, per_node_json,
                    outlier_voice, should_surface, timestamp)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    trace.trace_id,
                    disagreement.magnitude,
                    json.dumps(disagreement.pairwise),
                    json.dumps(disagreement.per_node),
                    disagreement.outlier_voice,
                    1 if disagreement.should_surface else 0,
                    disagreement.timestamp,
                ),
            )

        self._conn.commit()
        return trace.trace_id

    def _row_to_trace(self, row: sqlite3.Row) -> TreeTrace:
        """Deserialize a database row into a TreeTrace."""
        nodes = tuple(_dict_to_node(d) for d in json.loads(row["nodes_json"]))
        leaves = tuple(_dict_to_leaf(d) for d in json.loads(row["leaf_traces_json"]))
        cv = json.loads(row["combinator_versions_json"])
        meta = json.loads(row["metadata_json"])

        return TreeTrace(
            trace_id=row["trace_id"],
            task_type=row["task_type"],
            k=row["k"],
            depth=row["depth"],
            tau=row["tau"],
            cost_estimate=row["cost_estimate"],
            nodes=nodes,
            leaf_traces=leaves,
            final_output=row["final_output"],
            timestamp=row["timestamp"],
            execution_time_ms=row["execution_time_ms"],
            total_llm_calls=row["total_llm_calls"],
            total_tokens=row["total_tokens"],
            final_score=row["final_score"],
            combinator_versions=cv,
            metadata=meta,
        )

    def _row_to_disagreement(self, row: sqlite3.Row) -> DisagreementSignal:
        """Deserialize a database row into a DisagreementSignal."""
        return DisagreementSignal(
            magnitude=row["magnitude"],
            pairwise=json.loads(row["pairwise_json"]),
            per_node=json.loads(row["per_node_json"]),
            outlier_voice=row["outlier_voice"],
            should_surface=bool(row["should_surface"]),
            timestamp=row["timestamp"],
        )

    def load(self, trace_id: str) -> TreeTrace | None:
        """Load a single trace by ID, or None if not found."""
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_trace(row)

    def query_by_task_type(
        self, task_type: str, limit: int = 50
    ) -> list[TreeTrace]:
        """Return traces matching a task_type, newest first."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM traces WHERE task_type = ? ORDER BY timestamp DESC LIMIT ?",
            (task_type, limit),
        )
        return [self._row_to_trace(row) for row in cur.fetchall()]

    def query_by_failure(
        self, min_disagreement: float = 0.3, limit: int = 20
    ) -> list[tuple[TreeTrace, DisagreementSignal]]:
        """Return traces paired with high-magnitude disagreement signals."""
        cur = self._conn.cursor()
        cur.execute(
            """SELECT t.*, d.id AS d_id, d.magnitude, d.pairwise_json,
                      d.per_node_json, d.outlier_voice, d.should_surface,
                      d.timestamp AS d_timestamp
               FROM traces t
               JOIN disagreements d ON t.trace_id = d.trace_id
               WHERE d.magnitude >= ?
               ORDER BY d.magnitude DESC
               LIMIT ?""",
            (min_disagreement, limit),
        )
        results: list[tuple[TreeTrace, DisagreementSignal]] = []
        for row in cur.fetchall():
            trace = self._row_to_trace(row)
            signal = DisagreementSignal(
                magnitude=row["magnitude"],
                pairwise=json.loads(row["pairwise_json"]),
                per_node=json.loads(row["per_node_json"]),
                outlier_voice=row["outlier_voice"],
                should_surface=bool(row["should_surface"]),
                timestamp=row["d_timestamp"],
            )
            results.append((trace, signal))
        return results

    def query_by_combinator_version(
        self, version_id: str, limit: int = 50
    ) -> list[TreeTrace]:
        """Return traces whose combinator_versions contain the given version_id."""
        cur = self._conn.cursor()
        # Use JSON contains via LIKE on the serialized dict
        pattern = f'%"{version_id}"%'
        cur.execute(
            """SELECT * FROM traces
               WHERE combinator_versions_json LIKE ?
               ORDER BY timestamp DESC LIMIT ?""",
            (pattern, limit),
        )
        return [self._row_to_trace(row) for row in cur.fetchall()]

    def search_output(self, query: str, limit: int = 20) -> list[TreeTrace]:
        """Full-text search on final_output via FTS5."""
        cur = self._conn.cursor()
        cur.execute(
            """SELECT t.* FROM trace_fts fts
               JOIN traces t ON fts.trace_id = t.trace_id
               WHERE trace_fts MATCH ?
               LIMIT ?""",
            (query, limit),
        )
        return [self._row_to_trace(row) for row in cur.fetchall()]

    def count(self) -> int:
        """Return total number of stored traces."""
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM traces")
        result = cur.fetchone()
        return result[0] if result else 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
