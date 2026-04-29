"""CombinatorRegistry -- top-level facade for combinator management."""

from __future__ import annotations

from rsocas.contracts.combinators import ValidationSnapshot, VersionedCombinator

from rsocas.combinators.crystallizer import Crystallizer


class CombinatorRegistry:
    """Wraps the crystallizer to provide a simple registration API.

    Maintains an in-memory cache of active combinators keyed by name.
    """

    def __init__(self, crystallizer: Crystallizer) -> None:
        self._crystallizer = crystallizer
        self._active: dict[str, VersionedCombinator] = {}

    def register(
        self,
        name: str,
        fn: object,
        validation: ValidationSnapshot | None = None,
    ) -> VersionedCombinator:
        """Register a combinator.

        If *validation* is provided the combinator is crystallized
        immediately.  Otherwise a fluid combinator is created via
        ``get_or_create``.
        """
        if validation is not None:
            vc = self._crystallizer.crystallize(name, fn, validation)
        else:
            vc = self._crystallizer.get_or_create(name, fn)

        self._active[name] = vc
        return vc

    def get_active_versions(self) -> dict[str, str]:
        """Return ``{name: version_id}`` for all active combinators."""
        return {name: vc.version_id for name, vc in self._active.items()}

    def get_active(self, name: str) -> VersionedCombinator | None:
        """Return the active combinator for *name*, or ``None``."""
        return self._active.get(name)
