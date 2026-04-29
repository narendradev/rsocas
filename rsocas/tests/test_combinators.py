"""Tests for the combinator lifecycle module.

Covers CombinatorDB CRUD, crystallizer state machine, penumbra
relevance ranking, and the CombinatorRegistry facade.
"""

from __future__ import annotations

import time
import uuid

import pytest

from rsocas.contracts.combinators import (
    RepairRecord,
    ValidationSnapshot,
    VersionedCombinator,
)
from rsocas.combinators.crystallizer import Crystallizer
from rsocas.combinators.penumbra import PenumbraStore
from rsocas.combinators.registry import CombinatorRegistry
from rsocas.combinators.versioned import CombinatorDB


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_validation(
    mean: float = 0.85,
    std: float = 0.05,
    n: int = 100,
) -> ValidationSnapshot:
    return ValidationSnapshot(
        task_types=("qa", "summarize"),
        input_size_range=(10, 500),
        n_samples=n,
        mean_score=mean,
        score_std=std,
        timestamp=time.time(),
    )


def _make_vc(
    name: str = "map_reduce",
    status: str = "fluid",
    mean: float = 0.85,
    std: float = 0.05,
) -> VersionedCombinator:
    return VersionedCombinator(
        name=name,
        version_id=str(uuid.uuid4()),
        code_hash="abc123",
        status=status,
        created_at=time.time(),
        expires_at=time.time() + 86400,
        validation=_make_validation(mean=mean, std=std),
    )


def _make_stack() -> tuple[CombinatorDB, PenumbraStore, Crystallizer]:
    db = CombinatorDB(":memory:")
    pen = PenumbraStore(db)
    cryst = Crystallizer(db, pen, default_ttl=100.0)
    return db, pen, cryst


# ------------------------------------------------------------------
# CombinatorDB: store / load / update_status
# ------------------------------------------------------------------


class TestCombinatorDB:
    def test_store_load_roundtrip(self) -> None:
        db = CombinatorDB(":memory:")
        vc = _make_vc()
        db.store(vc)
        loaded = db.load(vc.version_id)

        assert loaded is not None
        assert loaded.name == vc.name
        assert loaded.version_id == vc.version_id
        assert loaded.status == vc.status
        assert loaded.validation.mean_score == vc.validation.mean_score
        assert loaded.validation.task_types == ("qa", "summarize")
        db.close()

    def test_load_missing_returns_none(self) -> None:
        db = CombinatorDB(":memory:")
        assert db.load("nonexistent") is None
        db.close()

    def test_update_status(self) -> None:
        db = CombinatorDB(":memory:")
        vc = _make_vc(status="crystallized")
        db.store(vc)

        updated = db.update_status(vc.version_id, "dissolving")
        assert updated.status == "dissolving"
        assert updated.name == vc.name

        reloaded = db.load(vc.version_id)
        assert reloaded is not None
        assert reloaded.status == "dissolving"
        db.close()

    def test_update_status_missing_raises(self) -> None:
        db = CombinatorDB(":memory:")
        with pytest.raises(KeyError):
            db.update_status("nonexistent", "expired")
        db.close()

    def test_list_by_status(self) -> None:
        db = CombinatorDB(":memory:")
        fluid = _make_vc(status="fluid")
        crystallized = _make_vc(name="other", status="crystallized")
        db.store(fluid)
        db.store(crystallized)

        results = db.list_by_status("fluid")
        assert len(results) == 1
        assert results[0].version_id == fluid.version_id
        db.close()

    def test_load_active(self) -> None:
        db = CombinatorDB(":memory:")
        vc = _make_vc(status="crystallized")
        db.store(vc)

        active = db.load_active("map_reduce")
        assert active is not None
        assert active.version_id == vc.version_id

        assert db.load_active("nonexistent") is None
        db.close()

    def test_repairs_roundtrip(self) -> None:
        db = CombinatorDB(":memory:")
        repair = RepairRecord(
            timestamp=time.time(),
            trigger="staleness",
            from_version="v1",
            change_summary="retrained",
            score_delta=0.02,
        )
        vc = VersionedCombinator(
            name="test",
            version_id=str(uuid.uuid4()),
            code_hash="h",
            status="dissolving",
            created_at=time.time(),
            expires_at=time.time() + 100,
            validation=_make_validation(),
            repairs=(repair,),
        )
        db.store(vc)
        loaded = db.load(vc.version_id)
        assert loaded is not None
        assert len(loaded.repairs) == 1
        assert loaded.repairs[0].trigger == "staleness"
        assert loaded.repairs[0].score_delta == 0.02
        db.close()


# ------------------------------------------------------------------
# CombinatorDB: penumbra store / load / prune
# ------------------------------------------------------------------


class TestCombinatorDBPenumbra:
    def test_store_and_load_penumbra(self) -> None:
        db = CombinatorDB(":memory:")
        v1 = _make_vc(name="variant_a", mean=0.80)
        v2 = _make_vc(name="variant_b", mean=0.82)

        db.store_penumbra("parent", v1, fitness_delta=0.05)
        db.store_penumbra("parent", v2, fitness_delta=0.03)

        loaded = db.load_penumbra("parent", limit=10)
        assert len(loaded) == 2
        # Ordered by fitness_delta DESC
        assert loaded[0].validation.mean_score == 0.80  # delta 0.05
        assert loaded[1].validation.mean_score == 0.82  # delta 0.03
        db.close()

    def test_prune_penumbra(self) -> None:
        db = CombinatorDB(":memory:")
        for i in range(5):
            v = _make_vc(name=f"var_{i}", mean=0.70 + i * 0.01)
            db.store_penumbra("parent", v, fitness_delta=float(i))

        removed = db.prune_penumbra("parent", max_variants=3)
        assert removed == 2

        remaining = db.load_penumbra("parent", limit=10)
        assert len(remaining) == 3
        db.close()

    def test_prune_noop_when_under_limit(self) -> None:
        db = CombinatorDB(":memory:")
        v = _make_vc()
        db.store_penumbra("parent", v, fitness_delta=1.0)
        assert db.prune_penumbra("parent", max_variants=5) == 0
        db.close()


# ------------------------------------------------------------------
# Crystallizer: full lifecycle
# ------------------------------------------------------------------


class TestCrystallizer:
    def test_crystallize(self) -> None:
        _db, _pen, cryst = _make_stack()
        val = _make_validation(mean=0.90)

        vc = cryst.crystallize("my_combinator", lambda x: x, val)

        assert vc.status == "crystallized"
        assert vc.name == "my_combinator"
        assert vc.validation.mean_score == 0.90
        assert len(vc.code_hash) == 64  # SHA-256 hex

    def test_full_lifecycle(self) -> None:
        _db, _pen, cryst = _make_stack()
        val = _make_validation()

        # crystallize
        vc = cryst.crystallize("lc", lambda x: x, val)
        assert vc.status == "crystallized"

        # dissolve
        dissolved = cryst.dissolve(vc.version_id, "distribution shifted")
        assert dissolved.status == "dissolving"
        assert len(dissolved.repairs) == 1
        assert dissolved.repairs[0].change_summary == "distribution shifted"

        # expire
        expired = cryst.expire(dissolved.version_id)
        assert expired.status == "expired"

    def test_invalid_transition_raises(self) -> None:
        _db, _pen, cryst = _make_stack()
        val = _make_validation()
        vc = cryst.crystallize("x", lambda: None, val)

        # cannot go crystallized -> expired directly
        with pytest.raises(ValueError, match="Invalid transition"):
            cryst.expire(vc.version_id)

    def test_check_staleness_fresh(self) -> None:
        _db, _pen, cryst = _make_stack()
        val = _make_validation(mean=0.85, std=0.05)
        vc = cryst.crystallize("s", lambda: None, val)

        # Same distribution => staleness ~0
        staleness = cryst.check_staleness(vc.version_id, val)
        assert staleness == pytest.approx(0.0, abs=1e-9)

    def test_check_staleness_drifted(self) -> None:
        _db, _pen, cryst = _make_stack()
        val = _make_validation(mean=0.85, std=0.05)
        vc = cryst.crystallize("s", lambda: None, val)

        drifted = _make_validation(mean=0.70, std=0.05)
        staleness = cryst.check_staleness(vc.version_id, drifted)
        # |0.85 - 0.70| / max(0.05, 0.01) = 0.15 / 0.05 = 3.0
        assert staleness == pytest.approx(3.0, abs=1e-9)

    def test_check_staleness_missing_raises(self) -> None:
        _db, _pen, cryst = _make_stack()
        with pytest.raises(KeyError):
            cryst.check_staleness("nope", _make_validation())

    def test_tick_expires_stale_combinators(self) -> None:
        db, pen, cryst = _make_stack()
        val = _make_validation()

        vc1 = cryst.crystallize("a", lambda: 1, val)
        vc2 = cryst.crystallize("b", lambda: 2, val)

        # Advance time past expiry (ttl = 100)
        future = time.time() + 200
        dissolved = cryst.tick(future)

        assert len(dissolved) == 2
        for d in dissolved:
            assert d.status == "dissolving"
            assert len(d.repairs) == 1
            assert d.repairs[0].change_summary == "TTL expired"

    def test_tick_ignores_fresh_combinators(self) -> None:
        _db, _pen, cryst = _make_stack()
        val = _make_validation()
        cryst.crystallize("fresh", lambda: 1, val)

        dissolved = cryst.tick(time.time())
        assert dissolved == []

    def test_get_or_create_new(self) -> None:
        _db, _pen, cryst = _make_stack()

        vc = cryst.get_or_create("new_one", lambda: 42)
        assert vc.status == "fluid"
        assert vc.name == "new_one"

    def test_get_or_create_existing(self) -> None:
        db, pen, cryst = _make_stack()
        val = _make_validation()
        original = cryst.crystallize("existing", lambda: 1, val)

        found = cryst.get_or_create("existing", lambda: 1)
        assert found.version_id == original.version_id


# ------------------------------------------------------------------
# PenumbraStore: retrieve_candidates relevance ranking
# ------------------------------------------------------------------


class TestPenumbraStoreRelevance:
    def test_retrieve_candidates_sorted(self) -> None:
        db = CombinatorDB(":memory:")
        pen = PenumbraStore(db)

        # Variant close to mean=0.80
        close = _make_vc(name="close", mean=0.81, std=0.04)
        # Variant far from mean=0.80
        far = _make_vc(name="far", mean=0.50, std=0.10)

        pen.store_variant("parent", close, fitness_delta=0.9)
        pen.store_variant("parent", far, fitness_delta=0.8)

        target = _make_validation(mean=0.80, std=0.04)
        candidates = pen.retrieve_candidates("parent", target)

        assert len(candidates) == 2
        # close variant should rank first
        assert candidates[0].name == "close"
        assert candidates[1].name == "far"

    def test_retrieve_empty(self) -> None:
        db = CombinatorDB(":memory:")
        pen = PenumbraStore(db)
        result = pen.retrieve_candidates("nobody", _make_validation())
        assert result == []


# ------------------------------------------------------------------
# CombinatorRegistry
# ------------------------------------------------------------------


class TestCombinatorRegistry:
    def test_register_with_validation(self) -> None:
        _db, _pen, cryst = _make_stack()
        reg = CombinatorRegistry(cryst)

        val = _make_validation()
        vc = reg.register("my_fn", lambda: 1, validation=val)

        assert vc.status == "crystallized"
        assert reg.get_active("my_fn") is not None
        assert "my_fn" in reg.get_active_versions()

    def test_register_without_validation(self) -> None:
        _db, _pen, cryst = _make_stack()
        reg = CombinatorRegistry(cryst)

        vc = reg.register("lazy_fn", lambda: 2)
        assert vc.status == "fluid"
        assert reg.get_active("lazy_fn") is vc

    def test_get_active_versions(self) -> None:
        _db, _pen, cryst = _make_stack()
        reg = CombinatorRegistry(cryst)

        reg.register("a", lambda: 1, _make_validation())
        reg.register("b", lambda: 2)

        versions = reg.get_active_versions()
        assert set(versions.keys()) == {"a", "b"}

    def test_get_active_missing(self) -> None:
        _db, _pen, cryst = _make_stack()
        reg = CombinatorRegistry(cryst)
        assert reg.get_active("nonexistent") is None
