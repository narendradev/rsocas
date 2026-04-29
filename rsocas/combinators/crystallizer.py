"""Crystallizer -- lifecycle state machine for combinator versioning.

State machine: fluid -> crystallized -> dissolving -> expired
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import replace

from rsocas.contracts.combinators import (
    RepairRecord,
    ValidationSnapshot,
    VersionedCombinator,
)

from rsocas.combinators.penumbra import PenumbraStore
from rsocas.combinators.versioned import CombinatorDB

_VALID_TRANSITIONS: dict[str, set[str]] = {
    "fluid": {"crystallized"},
    "crystallized": {"dissolving"},
    "dissolving": {"expired"},
    "expired": set(),
}


def _compute_code_hash(fn: object) -> str:
    """Deterministic hash of a callable's representation."""
    return hashlib.sha256(repr(fn).encode("utf-8")).hexdigest()


def _make_fluid_validation() -> ValidationSnapshot:
    """Default empty validation for fluid combinators."""
    return ValidationSnapshot(
        task_types=(),
        input_size_range=(0, 0),
        n_samples=0,
        mean_score=0.0,
        score_std=0.0,
        timestamp=time.time(),
    )


class Crystallizer:
    """Lifecycle state machine for versioned combinators.

    Manages the transitions: fluid -> crystallized -> dissolving -> expired.
    """

    def __init__(
        self,
        db: CombinatorDB,
        penumbra: PenumbraStore,
        default_ttl: float = 86400.0,
    ) -> None:
        self._db = db
        self._penumbra = penumbra
        self._default_ttl = default_ttl

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def crystallize(
        self,
        name: str,
        fn: object,
        validation: ValidationSnapshot,
    ) -> VersionedCombinator:
        """Promote a fluid pattern to *crystallized* status.

        Creates a new ``VersionedCombinator`` with a fresh UUID,
        a SHA-256 code hash derived from *fn*, and an expiry based
        on ``default_ttl``.
        """
        now = time.time()
        vc = VersionedCombinator(
            name=name,
            version_id=str(uuid.uuid4()),
            code_hash=_compute_code_hash(fn),
            status="crystallized",
            created_at=now,
            expires_at=now + self._default_ttl,
            validation=validation,
        )
        self._db.store(vc)
        return vc

    def check_staleness(
        self, version_id: str, current: ValidationSnapshot
    ) -> float:
        """Compare stored validation against a current distribution.

        Returns::

            |stored.mean_score - current.mean_score| / max(stored.score_std, 0.01)

        Result is 0.0 (fresh) to 1.0+ (stale).  Values > 1.0 indicate
        significant distribution drift.
        """
        vc = self._db.load(version_id)
        if vc is None:
            raise KeyError(f"No combinator with version_id={version_id!r}")

        mean_diff = abs(vc.validation.mean_score - current.mean_score)
        denominator = max(vc.validation.score_std, 0.01)
        return mean_diff / denominator

    def dissolve(
        self, version_id: str, reason: str
    ) -> VersionedCombinator:
        """Begin dissolution: status -> 'dissolving'.

        Appends a ``RepairRecord`` documenting the reason.
        """
        vc = self._db.load(version_id)
        if vc is None:
            raise KeyError(f"No combinator with version_id={version_id!r}")

        self._assert_transition(vc.status, "dissolving")

        repair = RepairRecord(
            timestamp=time.time(),
            trigger="dissolve",
            from_version=vc.version_id,
            change_summary=reason,
            score_delta=0.0,
        )
        updated = replace(
            vc,
            status="dissolving",
            repairs=vc.repairs + (repair,),
        )
        self._db.store(updated)
        return updated

    def expire(self, version_id: str) -> VersionedCombinator:
        """Final expiry: status -> 'expired'."""
        vc = self._db.load(version_id)
        if vc is None:
            raise KeyError(f"No combinator with version_id={version_id!r}")

        self._assert_transition(vc.status, "expired")
        return self._db.update_status(version_id, "expired")

    def tick(self, current_time: float) -> list[VersionedCombinator]:
        """Check all crystallized combinators for time-based expiry.

        Any combinator whose ``expires_at < current_time`` is
        transitioned to 'dissolving' with an auto-expiry repair record.
        Returns the list of newly-dissolving combinators.
        """
        crystallized = self._db.list_by_status("crystallized")
        dissolved: list[VersionedCombinator] = []
        for vc in crystallized:
            if vc.expires_at < current_time:
                updated = self.dissolve(vc.version_id, "TTL expired")
                dissolved.append(updated)
        return dissolved

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_or_create(
        self, name: str, fn: object
    ) -> VersionedCombinator:
        """Get the active combinator for *name*, or create a fluid one."""
        existing = self._db.load_active(name)
        if existing is not None:
            return existing

        now = time.time()
        vc = VersionedCombinator(
            name=name,
            version_id=str(uuid.uuid4()),
            code_hash=_compute_code_hash(fn),
            status="fluid",
            created_at=now,
            expires_at=now + self._default_ttl,
            validation=_make_fluid_validation(),
        )
        self._db.store(vc)
        return vc

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_transition(current: str, target: str) -> None:
        allowed = _VALID_TRANSITIONS.get(current, set())
        if target not in allowed:
            raise ValueError(
                f"Invalid transition: {current!r} -> {target!r}. "
                f"Allowed targets from {current!r}: {allowed}"
            )
