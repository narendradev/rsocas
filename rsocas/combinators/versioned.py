"""CombinatorDB -- SQLite store for versioned combinators and penumbra variants."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from rsocas.contracts.combinators import (
    RepairRecord,
    ValidationSnapshot,
    VersionedCombinator,
)


def _validation_to_json(v: ValidationSnapshot) -> str:
    return json.dumps(asdict(v))


def _validation_from_json(raw: str) -> ValidationSnapshot:
    d = json.loads(raw)
    return ValidationSnapshot(
        task_types=tuple(d["task_types"]),
        input_size_range=tuple(d["input_size_range"]),
        n_samples=d["n_samples"],
        mean_score=d["mean_score"],
        score_std=d["score_std"],
        timestamp=d["timestamp"],
    )


def _repairs_to_json(repairs: tuple[RepairRecord, ...]) -> str:
    return json.dumps([asdict(r) for r in repairs])


def _repairs_from_json(raw: str) -> tuple[RepairRecord, ...]:
    items = json.loads(raw)
    return tuple(
        RepairRecord(
            timestamp=r["timestamp"],
            trigger=r["trigger"],
            from_version=r["from_version"],
            change_summary=r["change_summary"],
            score_delta=r["score_delta"],
        )
        for r in items
    )


class CombinatorDB:
    """SQLite store for versioned combinators and penumbra variants.

    Use ``':memory:'`` for tests, a file path for production.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS combinators (
                version_id     TEXT PRIMARY KEY,
                name           TEXT NOT NULL,
                code_hash      TEXT NOT NULL,
                status         TEXT NOT NULL,
                created_at     REAL NOT NULL,
                expires_at     REAL NOT NULL,
                validation_json TEXT NOT NULL,
                repairs_json   TEXT NOT NULL,
                cost_constant  REAL NOT NULL,
                type_signature TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS penumbra (
                variant_id     TEXT PRIMARY KEY,
                parent_name    TEXT NOT NULL,
                version_id     TEXT NOT NULL,
                fitness_delta  REAL NOT NULL,
                created_at     REAL NOT NULL,
                FOREIGN KEY(version_id) REFERENCES combinators(version_id)
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Combinator CRUD
    # ------------------------------------------------------------------

    def store(self, vc: VersionedCombinator) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO combinators
                (version_id, name, code_hash, status, created_at, expires_at,
                 validation_json, repairs_json, cost_constant, type_signature)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                vc.version_id,
                vc.name,
                vc.code_hash,
                vc.status,
                vc.created_at,
                vc.expires_at,
                _validation_to_json(vc.validation),
                _repairs_to_json(vc.repairs),
                vc.cost_constant,
                vc.type_signature,
            ),
        )
        self._conn.commit()

    def load(self, version_id: str) -> VersionedCombinator | None:
        cur = self._conn.execute(
            "SELECT * FROM combinators WHERE version_id = ?",
            (version_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_vc(row)

    def load_active(self, name: str) -> VersionedCombinator | None:
        cur = self._conn.execute(
            """
            SELECT * FROM combinators
            WHERE name = ? AND status IN ('fluid', 'crystallized')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (name,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_vc(row)

    def update_status(
        self, version_id: str, new_status: str
    ) -> VersionedCombinator:
        existing = self.load(version_id)
        if existing is None:
            raise KeyError(f"No combinator with version_id={version_id!r}")
        updated = replace(existing, status=new_status)
        self.store(updated)
        return updated

    def list_by_status(self, status: str) -> list[VersionedCombinator]:
        cur = self._conn.execute(
            "SELECT * FROM combinators WHERE status = ?",
            (status,),
        )
        return [self._row_to_vc(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Penumbra
    # ------------------------------------------------------------------

    def store_penumbra(
        self,
        parent_name: str,
        variant: VersionedCombinator,
        fitness_delta: float,
    ) -> None:
        self.store(variant)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO penumbra
                (variant_id, parent_name, version_id, fitness_delta, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                parent_name,
                variant.version_id,
                fitness_delta,
                time.time(),
            ),
        )
        self._conn.commit()

    def load_penumbra(
        self, parent_name: str, limit: int = 5
    ) -> list[VersionedCombinator]:
        cur = self._conn.execute(
            """
            SELECT c.* FROM penumbra p
            JOIN combinators c ON p.version_id = c.version_id
            WHERE p.parent_name = ?
            ORDER BY p.fitness_delta DESC
            LIMIT ?
            """,
            (parent_name, limit),
        )
        return [self._row_to_vc(row) for row in cur.fetchall()]

    def prune_penumbra(
        self, parent_name: str, max_variants: int = 10
    ) -> int:
        cur = self._conn.execute(
            """
            SELECT p.rowid, p.version_id FROM penumbra p
            WHERE p.parent_name = ?
            ORDER BY p.fitness_delta DESC
            """,
            (parent_name,),
        )
        rows = cur.fetchall()
        if len(rows) <= max_variants:
            return 0

        to_remove = rows[max_variants:]
        removed = 0
        for rowid, version_id in to_remove:
            self._conn.execute("DELETE FROM penumbra WHERE rowid = ?", (rowid,))
            self._conn.execute(
                "DELETE FROM combinators WHERE version_id = ?", (version_id,)
            )
            removed += 1

        self._conn.commit()
        return removed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_vc(row: tuple) -> VersionedCombinator:
        (
            version_id,
            name,
            code_hash,
            status,
            created_at,
            expires_at,
            validation_json,
            repairs_json,
            cost_constant,
            type_signature,
        ) = row
        return VersionedCombinator(
            name=name,
            version_id=version_id,
            code_hash=code_hash,
            status=status,
            created_at=created_at,
            expires_at=expires_at,
            validation=_validation_from_json(validation_json),
            repairs=_repairs_from_json(repairs_json),
            cost_constant=cost_constant,
            type_signature=type_signature,
        )

    def close(self) -> None:
        self._conn.close()
