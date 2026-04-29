"""Contract 5: Combinator Lifecycle — versioning, penumbra, crystallization.

VersionedCombinator carries its validation distribution, repair history,
and expiration timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ValidationSnapshot:
    task_types: tuple[str, ...]
    input_size_range: tuple[int, int]
    n_samples: int
    mean_score: float
    score_std: float
    timestamp: float


@dataclass(frozen=True)
class RepairRecord:
    timestamp: float
    trigger: str
    from_version: str
    change_summary: str
    score_delta: float


@dataclass(frozen=True)
class VersionedCombinator:
    name: str
    version_id: str
    code_hash: str
    status: str
    created_at: float
    expires_at: float
    validation: ValidationSnapshot
    repairs: tuple[RepairRecord, ...] = ()
    cost_constant: float = 1.0
    type_signature: str = ""


class CombinatorStore(Protocol):
    def crystallize(self, name: str, fn: object, validation: ValidationSnapshot) -> VersionedCombinator: ...
    def dissolve(self, version_id: str, reason: str) -> VersionedCombinator: ...
    def get_active(self, name: str) -> VersionedCombinator | None: ...
    def get_penumbra(self, name: str, limit: int = 5) -> list[VersionedCombinator]: ...
    def check_staleness(self, version_id: str, current: ValidationSnapshot) -> float: ...


class TempoController(Protocol):
    def record_human_feedback(self, timestamp: float) -> None: ...
    def record_system_event(self, timestamp: float, event: str) -> None: ...
    def breathing_rate(self) -> float: ...
    def should_crystallize(self) -> bool: ...
    def should_dissolve(self) -> bool: ...
