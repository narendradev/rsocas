"""Tests for Lambda-RLM integration patch -- tracing + combinator versioning."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from rsocas.combinators.crystallizer import Crystallizer
from rsocas.combinators.lambda_rlm_integration import patch_lambda_rlm_full
from rsocas.combinators.penumbra import PenumbraStore
from rsocas.combinators.registry import CombinatorRegistry
from rsocas.combinators.versioned import CombinatorDB
from rsocas.tracing.collector import TraceCollector


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_mock_lrlm() -> MagicMock:
    """Create a mock LambdaRLM instance with a working _register_library.

    The mock simulates registering combinators into repl.globals,
    which is the key behavior the integration patch wraps.
    """
    lrlm = MagicMock()

    def fake_register_library(repl: Any, plan: Any, query: str = "") -> None:
        """Simulate _register_library by populating repl.globals."""
        repl.globals["_Split"] = lambda text, k: [text]
        repl.globals["_Peek"] = lambda text, start, length: text[start : start + length]
        repl.globals["_Reduce"] = lambda parts: "\n".join(parts)
        repl.globals["_FilterRelevant"] = lambda q, items: [c for c, _ in items]

    lrlm._register_library = fake_register_library

    # Provide a real _llm_query so the tracing patch can wrap it
    lrlm._llm_query = lambda prompt, model=None: "mock response"

    return lrlm


def _make_registry() -> CombinatorRegistry:
    db = CombinatorDB(":memory:")
    penumbra = PenumbraStore(db)
    crystallizer = Crystallizer(db, penumbra)
    return CombinatorRegistry(crystallizer)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestPatchReturnValues:
    def test_patch_returns_triple(self) -> None:
        """patch_lambda_rlm_full returns (lrlm, collector, registry)."""
        lrlm = _make_mock_lrlm()
        registry = _make_registry()

        result = patch_lambda_rlm_full(lrlm, registry=registry)

        assert len(result) == 3
        patched_lrlm, collector, returned_registry = result
        assert patched_lrlm is lrlm
        assert isinstance(collector, TraceCollector)
        assert returned_registry is registry


class TestPatchWithoutRegistry:
    def test_patch_without_registry(self) -> None:
        """When registry=None, a default in-memory registry is created."""
        lrlm = _make_mock_lrlm()

        patched_lrlm, collector, registry = patch_lambda_rlm_full(lrlm, registry=None)

        assert patched_lrlm is lrlm
        assert isinstance(collector, TraceCollector)
        assert isinstance(registry, CombinatorRegistry)


class TestPatchWithRegistryRecordsCombinators:
    def test_patch_with_registry_records_combinators(self) -> None:
        """After calling _register_library, registry should have combinator entries."""
        lrlm = _make_mock_lrlm()
        registry = _make_registry()

        patched_lrlm, _collector, returned_registry = patch_lambda_rlm_full(
            lrlm, registry=registry,
        )

        # Simulate what LambdaRLM does: call _register_library
        mock_repl = MagicMock()
        mock_repl.globals = {}
        mock_repl._llm_query = lambda prompt, model=None: "mock"
        mock_plan = MagicMock()

        patched_lrlm._register_library(mock_repl, mock_plan, "test query")

        # Verify all four combinators are registered
        active = returned_registry.get_active_versions()
        for name in ("_Split", "_Peek", "_Reduce", "_FilterRelevant"):
            assert name in active, f"{name} not found in registry"
            vc = returned_registry.get_active(name)
            assert vc is not None
            assert vc.name == name

    def test_registry_not_populated_before_register_call(self) -> None:
        """Registry should be empty until _register_library is called."""
        lrlm = _make_mock_lrlm()
        registry = _make_registry()

        _patched_lrlm, _collector, returned_registry = patch_lambda_rlm_full(
            lrlm, registry=registry,
        )

        assert returned_registry.get_active_versions() == {}


class TestTracingStillWorks:
    def test_tracing_still_works_with_full_patch(self) -> None:
        """Tracing collector should record events after the full patch.

        The tracing patch wraps _register_library so that repl._llm_query
        is replaced with a traced version.  Verify this still works when
        the combinator versioning patch is also applied.
        """
        lrlm = _make_mock_lrlm()
        registry = _make_registry()

        patched_lrlm, collector, _registry = patch_lambda_rlm_full(
            lrlm, registry=registry,
        )

        # Simulate a registration call that exercises the traced _llm_query
        mock_repl = MagicMock()
        mock_repl.globals = {}
        mock_repl._llm_query = lambda prompt, model=None: "real response"
        mock_plan = MagicMock()

        patched_lrlm._register_library(mock_repl, mock_plan, "test")

        # The _llm_query on the repl should now be traced.
        # Calling it should produce a trace event.
        traced_llm = mock_repl._llm_query
        result = traced_llm("test prompt")

        assert result == "real response"

        events = collector.get_events()
        assert len(events) == 1
        assert events[0].prompt == "test prompt"
        assert events[0].response == "real response"
        assert events[0].call_context == "leaf"
