"""Tests for the DSPy leaf registry — no real DSPy needed."""

from __future__ import annotations

import pytest

from rsocas.adapters.dspy_leaf_registry import DSPyLeafRegistry, LeafModuleSpec


class TestLeafModuleSpec:
    def test_leaf_module_spec_defaults(self) -> None:
        """Verify default values for LeafModuleSpec."""
        spec = LeafModuleSpec(task_type="qa")

        assert spec.task_type == "qa"
        assert spec.signature == "context, question -> answer"
        assert spec.instruction == ""
        assert spec.demos == ()

    def test_leaf_module_spec_custom(self) -> None:
        """Custom values are stored correctly."""
        demos = ({"context": "ctx", "answer": "ans"},)
        spec = LeafModuleSpec(
            task_type="summarization",
            signature="text -> summary",
            instruction="Summarize the text.",
            demos=demos,
        )

        assert spec.task_type == "summarization"
        assert spec.signature == "text -> summary"
        assert spec.instruction == "Summarize the text."
        assert spec.demos == demos

    def test_leaf_module_spec_is_frozen(self) -> None:
        """LeafModuleSpec is immutable."""
        spec = LeafModuleSpec(task_type="qa")
        with pytest.raises(AttributeError):
            spec.task_type = "other"


class TestDSPyLeafRegistry:
    def test_register_spec(self) -> None:
        """Register and list task types."""
        registry = DSPyLeafRegistry()
        spec = LeafModuleSpec(task_type="qa")

        registry.register(spec)

        assert registry.list_registered() == ["qa"]

    def test_register_multiple(self) -> None:
        """Multiple specs are listed in sorted order."""
        registry = DSPyLeafRegistry()
        registry.register(LeafModuleSpec(task_type="qa"))
        registry.register(LeafModuleSpec(task_type="summarization"))
        registry.register(LeafModuleSpec(task_type="classification"))

        result = registry.list_registered()

        assert result == ["classification", "qa", "summarization"]

    def test_register_overwrites(self) -> None:
        """Re-registering the same task_type overwrites the spec."""
        registry = DSPyLeafRegistry()
        spec1 = LeafModuleSpec(task_type="qa", instruction="v1")
        spec2 = LeafModuleSpec(task_type="qa", instruction="v2")

        registry.register(spec1)
        registry.register(spec2)

        assert registry.list_registered() == ["qa"]

    def test_get_leaf_fn_without_dspy(self) -> None:
        """When DSPy is not available, the returned fn raises ImportError."""
        registry = DSPyLeafRegistry()
        spec = LeafModuleSpec(task_type="qa")
        registry.register(spec)

        fn = registry.get_leaf_fn("qa")

        # DSPy is not installed in the test environment
        with pytest.raises(ImportError, match="DSPy is not installed"):
            fn("test prompt")

    def test_get_leaf_fn_unregistered(self) -> None:
        """Getting a leaf fn for an unregistered task raises KeyError."""
        registry = DSPyLeafRegistry()

        with pytest.raises(KeyError, match="No spec registered"):
            registry.get_leaf_fn("nonexistent")

    def test_inject_into_repl(self) -> None:
        """Verify repl_globals['llm_query'] is replaced."""
        registry = DSPyLeafRegistry()
        spec = LeafModuleSpec(task_type="qa")
        registry.register(spec)

        repl_globals: dict = {"llm_query": lambda p: "original"}

        registry.inject_into_repl(repl_globals, "qa")

        assert "llm_query" in repl_globals
        # The injected function should NOT be the original
        # It should be the DSPy-wrapped version (which will raise ImportError)
        with pytest.raises(ImportError, match="DSPy is not installed"):
            repl_globals["llm_query"]("test")

    def test_create_dspy_module_without_dspy(self) -> None:
        """create_dspy_module raises ImportError when DSPy is missing."""
        registry = DSPyLeafRegistry()
        spec = LeafModuleSpec(task_type="qa")

        with pytest.raises(ImportError, match="DSPy is not installed"):
            registry.create_dspy_module(spec)

    def test_optimize_without_dspy(self) -> None:
        """optimize raises ImportError when DSPy is missing."""
        registry = DSPyLeafRegistry()
        spec = LeafModuleSpec(task_type="qa")
        registry.register(spec)

        with pytest.raises(ImportError, match="DSPy is not installed"):
            registry.optimize("qa", trainset=[], metric=lambda x: 1.0)

    def test_optimize_unregistered(self) -> None:
        """optimize raises KeyError for unregistered task_type."""
        registry = DSPyLeafRegistry()

        with pytest.raises(KeyError, match="No spec registered"):
            registry.optimize("nonexistent", trainset=[], metric=lambda x: 1.0)

    def test_list_registered_empty(self) -> None:
        """Empty registry returns empty list."""
        registry = DSPyLeafRegistry()

        assert registry.list_registered() == []
