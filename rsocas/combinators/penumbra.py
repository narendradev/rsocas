"""PenumbraStore -- higher-level interface for near-miss variant management."""

from __future__ import annotations

from rsocas.contracts.combinators import ValidationSnapshot, VersionedCombinator

from rsocas.combinators.versioned import CombinatorDB


class PenumbraStore:
    """Manages near-miss combinator variants (the 'penumbra' around a
    crystallized combinator).

    Variants are stored with a fitness delta that indicates how close
    they came to the parent combinator's performance.  When a new
    validation distribution arrives, candidates are ranked by
    relevance to that distribution.
    """

    def __init__(self, db: CombinatorDB) -> None:
        self._db = db

    def store_variant(
        self,
        parent_name: str,
        variant: VersionedCombinator,
        fitness_delta: float,
    ) -> None:
        """Store a near-miss variant, then prune if over the limit."""
        self._db.store_penumbra(parent_name, variant, fitness_delta)
        self._db.prune_penumbra(parent_name)

    def retrieve_candidates(
        self,
        parent_name: str,
        new_validation: ValidationSnapshot,
    ) -> list[VersionedCombinator]:
        """Retrieve variants sorted by relevance to *new_validation*.

        Relevance is defined as::

            1.0 / (1.0 + |mean_score_diff| + |std_diff|)

        Higher relevance means the variant's stored validation
        distribution is closer to the incoming distribution.
        """
        variants = self._db.load_penumbra(parent_name, limit=100)
        if not variants:
            return []

        scored: list[tuple[float, VersionedCombinator]] = []
        for v in variants:
            mean_diff = abs(v.validation.mean_score - new_validation.mean_score)
            std_diff = abs(v.validation.score_std - new_validation.score_std)
            relevance = 1.0 / (1.0 + mean_diff + std_diff)
            scored.append((relevance, v))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [v for _, v in scored]

    def prune(self, parent_name: str, max_variants: int = 10) -> int:
        """Remove lowest-fitness variants.  Returns count removed."""
        return self._db.prune_penumbra(parent_name, max_variants)
