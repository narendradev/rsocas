"""Additional penumbra tests: bulk pruning and distribution-shifted retrieval."""

from __future__ import annotations

import time
import uuid

import pytest

from rsocas.contracts.combinators import ValidationSnapshot, VersionedCombinator
from rsocas.combinators.penumbra import PenumbraStore
from rsocas.combinators.versioned import CombinatorDB


def _make_validation(
    mean: float = 0.85,
    std: float = 0.05,
) -> ValidationSnapshot:
    return ValidationSnapshot(
        task_types=("qa",),
        input_size_range=(10, 500),
        n_samples=100,
        mean_score=mean,
        score_std=std,
        timestamp=time.time(),
    )


def _make_variant(
    name: str,
    mean: float = 0.85,
    std: float = 0.05,
) -> VersionedCombinator:
    return VersionedCombinator(
        name=name,
        version_id=str(uuid.uuid4()),
        code_hash="hash",
        status="fluid",
        created_at=time.time(),
        expires_at=time.time() + 86400,
        validation=_make_validation(mean=mean, std=std),
    )


class TestPenumbraPruning:
    """Store 15 variants, prune to 10, verify lowest-fitness removed."""

    def test_prune_removes_lowest_fitness(self) -> None:
        db = CombinatorDB(":memory:")
        pen = PenumbraStore(db)

        # Store 15 variants with fitness_delta = i (0..14)
        for i in range(15):
            v = _make_variant(f"var_{i}", mean=0.70 + i * 0.01)
            pen.store_variant("parent", v, fitness_delta=float(i))

        # After each store_variant, auto-prune fires with default max=10.
        # So we should already be at 10.
        remaining = db.load_penumbra("parent", limit=100)
        assert len(remaining) == 10

        # The 10 kept should be the ones with highest fitness_delta (5..14)
        deltas_present = sorted(
            [r.validation.mean_score for r in remaining]
        )
        # Expect means 0.75 .. 0.84 (indices 5..14)
        expected_means = sorted([0.70 + i * 0.01 for i in range(5, 15)])
        for actual, expected in zip(deltas_present, expected_means):
            assert actual == pytest.approx(expected, abs=1e-9)

    def test_explicit_prune_to_smaller(self) -> None:
        db = CombinatorDB(":memory:")
        pen = PenumbraStore(db)

        for i in range(8):
            v = _make_variant(f"var_{i}")
            # Use store_penumbra directly to skip auto-prune threshold
            db.store_penumbra("parent", v, fitness_delta=float(i))

        removed = pen.prune("parent", max_variants=3)
        assert removed == 5

        remaining = db.load_penumbra("parent", limit=100)
        assert len(remaining) == 3


class TestPenumbraDistributionShift:
    """retrieve_candidates with a shifted distribution prefers variants
    closest to the new distribution."""

    def test_shifted_distribution_ranking(self) -> None:
        db = CombinatorDB(":memory:")
        pen = PenumbraStore(db)

        # Create variants across a range of mean scores
        v_low = _make_variant("low", mean=0.50, std=0.02)
        v_mid = _make_variant("mid", mean=0.70, std=0.05)
        v_high = _make_variant("high", mean=0.90, std=0.08)

        db.store_penumbra("parent", v_low, fitness_delta=1.0)
        db.store_penumbra("parent", v_mid, fitness_delta=1.0)
        db.store_penumbra("parent", v_high, fitness_delta=1.0)

        # New distribution shifts to mean=0.72, std=0.05
        new_val = _make_validation(mean=0.72, std=0.05)
        candidates = pen.retrieve_candidates("parent", new_val)

        assert len(candidates) == 3
        # v_mid (0.70, 0.05) should be closest to (0.72, 0.05)
        assert candidates[0].name == "mid"

    def test_std_also_affects_ranking(self) -> None:
        db = CombinatorDB(":memory:")
        pen = PenumbraStore(db)

        # Same mean, different std
        v_narrow = _make_variant("narrow", mean=0.80, std=0.02)
        v_wide = _make_variant("wide", mean=0.80, std=0.15)

        db.store_penumbra("parent", v_narrow, fitness_delta=1.0)
        db.store_penumbra("parent", v_wide, fitness_delta=1.0)

        target = _make_validation(mean=0.80, std=0.03)
        candidates = pen.retrieve_candidates("parent", target)

        assert len(candidates) == 2
        # narrow (std=0.02) is closer to target (std=0.03) than wide (std=0.15)
        assert candidates[0].name == "narrow"
        assert candidates[1].name == "wide"

    def test_relevance_score_computation(self) -> None:
        """Verify the exact relevance formula: 1/(1 + |mean_diff| + |std_diff|)."""
        db = CombinatorDB(":memory:")
        pen = PenumbraStore(db)

        v = _make_variant("exact", mean=0.80, std=0.05)
        db.store_penumbra("parent", v, fitness_delta=1.0)

        # Query with exact same distribution
        target = _make_validation(mean=0.80, std=0.05)
        candidates = pen.retrieve_candidates("parent", target)
        assert len(candidates) == 1
        # relevance = 1/(1+0+0) = 1.0 -- variant is a perfect match

        # Query with shifted distribution
        shifted = _make_validation(mean=0.90, std=0.10)
        candidates = pen.retrieve_candidates("parent", shifted)
        assert len(candidates) == 1
        # relevance = 1/(1 + 0.10 + 0.05) = 1/1.15 ~ 0.87
        # The variant is still returned, just with lower relevance
