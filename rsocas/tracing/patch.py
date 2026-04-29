"""Monkey-patch Lambda-RLM to emit TreeTrace objects.

The critical detail: ``_Reduce`` and ``_FilterRelevant`` closures inside
``_register_library`` capture ``repl._llm_query`` *directly* at definition
time (line ~466: ``_llm = repl._llm_query``).  They do **not** use
``repl.globals["llm_query"]``.

Therefore we must wrap ``repl._llm_query`` **before** ``_register_library``
captures it into closures.  We achieve this by wrapping
``_register_library`` itself: the wrapper replaces ``repl._llm_query``
with a traced version, then calls the original method (which snapshots
the now-traced function into its closures), and finally attaches the
collector to the repl for later retrieval.

The ``llm_query`` function in ``repl.globals`` (used by leaf calls in the
Phi executor code) is *also* backed by ``repl._llm_query``, so wrapping
the underlying method covers both paths.
"""

from __future__ import annotations

import functools
from typing import Any

from rsocas.tracing.collector import TraceCollector


def patch_for_tracing(lrlm: Any) -> tuple[Any, TraceCollector]:
    """Monkey-patch a ``LambdaRLM`` instance to emit traces.

    Strategy
    --------
    1. Save the original ``_register_library`` method.
    2. Replace it with a wrapper that, for each invocation:
       a. Saves the REPL's original ``_llm_query``.
       b. Replaces ``repl._llm_query`` with a traced version that logs
          every call to the shared ``TraceCollector``.
       c. Calls the original ``_register_library`` (which now captures the
          traced ``_llm_query`` into ``_Reduce`` / ``_FilterRelevant``).
       d. Re-binds ``repl.globals["llm_query"]`` so that the Phi executor
          code's leaf calls also go through the traced wrapper.
    3. Returns ``(patched_lrlm, collector)``.

    Args:
        lrlm: A ``LambdaRLM`` instance (from ``rlm.lambda_rlm``).

    Returns:
        A tuple of ``(lrlm, collector)`` where *lrlm* is the same object
        (now patched in-place) and *collector* accumulates events across
        calls to ``completion()``.
    """
    collector = TraceCollector()
    original_register = lrlm._register_library

    @functools.wraps(original_register)
    def _traced_register_library(repl: Any, plan: Any, query: str = "") -> None:
        # --- (a) Save the real _llm_query ----------------------------------
        original_llm_query = repl._llm_query

        # --- (b) Build traced wrapper that infers call_context from prompt --
        def _traced_llm_query(prompt: str, model: str | None = None) -> str:
            context = _infer_call_context(prompt)
            call_id = collector.start_call(prompt, model, context)
            response = original_llm_query(prompt, model)
            collector.end_call(call_id, response)
            return response

        # Replace the bound method so _register_library's closures capture it.
        repl._llm_query = _traced_llm_query

        # --- (c) Call the original (captures traced _llm_query) ------------
        original_register(repl, plan, query)

        # --- (d) Also update repl.globals so leaf calls in Phi use traced --
        repl.globals["llm_query"] = _traced_llm_query

    # Patch in place.
    lrlm._register_library = _traced_register_library

    return lrlm, collector


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_REDUCE_MARKERS = frozenset({
    "Merge these partial summaries",
    "Synthesise these partial answers",
    "Combine these partial analyses",
})

_FILTER_MARKERS = frozenset({
    "Does this excerpt contain information relevant",
    "Reply YES or NO only",
})


def _infer_call_context(prompt: str) -> str:
    """Heuristically classify a prompt as ``leaf``, ``reduce``, or ``filter``.

    The classification relies on known prompt prefixes injected by
    ``_register_library`` in Lambda-RLM.  If none match, the call is
    assumed to be a leaf.
    """
    for marker in _REDUCE_MARKERS:
        if marker in prompt:
            return "reduce"
    for marker in _FILTER_MARKERS:
        if marker in prompt:
            return "filter"
    return "leaf"
