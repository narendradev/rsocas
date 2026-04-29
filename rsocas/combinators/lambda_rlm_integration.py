"""Lambda-RLM integration patch -- applies tracing and combinator versioning.

Patches a LambdaRLM instance so that:
1. All LLM calls are traced via ``patch_for_tracing``.
2. Combinators registered in ``_register_library`` are version-stamped
   via the ``CombinatorRegistry``.
"""

from __future__ import annotations

from typing import Any

from rsocas.combinators.crystallizer import Crystallizer
from rsocas.combinators.penumbra import PenumbraStore
from rsocas.combinators.registry import CombinatorRegistry
from rsocas.combinators.versioned import CombinatorDB
from rsocas.tracing.collector import TraceCollector
from rsocas.tracing.patch import patch_for_tracing


_COMBINATOR_NAMES = ("_Split", "_Peek", "_Reduce", "_FilterRelevant")


def _make_default_registry() -> CombinatorRegistry:
    """Create a registry backed by an in-memory SQLite DB."""
    db = CombinatorDB(":memory:")
    penumbra = PenumbraStore(db)
    crystallizer = Crystallizer(db, penumbra)
    return CombinatorRegistry(crystallizer)


def patch_lambda_rlm_full(
    lrlm: Any,
    registry: CombinatorRegistry | None = None,
) -> tuple[Any, TraceCollector, CombinatorRegistry]:
    """Apply both tracing and combinator versioning patches to a LambdaRLM instance.

    Args:
        lrlm: A ``LambdaRLM`` instance (from ``rlm.lambda_rlm``).
        registry: Optional pre-configured registry.  If ``None``, creates
            one backed by an in-memory SQLite database.

    Returns:
        A tuple of ``(lrlm, collector, registry)``.
    """
    # Step 1: Apply tracing patch
    lrlm, collector = patch_for_tracing(lrlm)

    # Step 2: Ensure we have a registry
    effective_registry = registry if registry is not None else _make_default_registry()

    # Step 3: Wrap _register_library to version-stamp combinators
    original_register = lrlm._register_library

    def versioned_register(repl: Any, plan: Any, query: str = "") -> None:
        original_register(repl, plan, query)
        # After original registers combinators, wrap them with version info
        for name in _COMBINATOR_NAMES:
            fn = repl.globals.get(name)
            if fn is not None:
                effective_registry.register(name, fn)

    lrlm._register_library = versioned_register

    return lrlm, collector, effective_registry
