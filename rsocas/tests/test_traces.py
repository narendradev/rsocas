"""Tests for the tracing module: collector, builder, and patch."""
from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from rsocas.contracts.traces import LeafTrace, NodeTrace, TreeTrace
from rsocas.tracing.collector import CallEvent, TraceCollector
from rsocas.tracing.builder import TreeTraceBuilder
from rsocas.tracing.patch import patch_for_tracing, _infer_call_context


# --- TraceCollector tests ---

class TestTraceCollector:
    """Unit tests for TraceCollector start/end/get_events."""

    def test_single_call_round_trip(self) -> None:
        collector = TraceCollector()
        cid = collector.start_call("Hello", "gpt-4o", "leaf")
        collector.end_call(cid, "World", tokens_in=5, tokens_out=3)
        events = collector.get_events()
        assert len(events) == 1
        ev = events[0]
        assert ev.call_id == cid
        assert ev.prompt == "Hello"
        assert ev.response == "World"
        assert ev.model == "gpt-4o"
        assert ev.call_context == "leaf"
        assert ev.tokens_in == 5
        assert ev.tokens_out == 3

    def test_multiple_calls_preserve_order(self) -> None:
        collector = TraceCollector()
        ids = []
        for i in range(5):
            cid = collector.start_call(f"prompt_{i}", None, "leaf")
            collector.end_call(cid, f"response_{i}")
            ids.append(cid)
        events = collector.get_events()
        assert len(events) == 5
        for i, ev in enumerate(events):
            assert ev.prompt == f"prompt_{i}"
            assert ev.response == f"response_{i}"
            assert ev.call_id == ids[i]

    def test_timing_monotonic(self) -> None:
        collector = TraceCollector()
        cid = collector.start_call("p", None, "leaf")
        time.sleep(0.001)
        collector.end_call(cid, "r")
        assert collector.get_events()[0].end_time > collector.get_events()[0].start_time

    def test_end_call_unknown_id_raises(self) -> None:
        collector = TraceCollector()
        with pytest.raises(KeyError):
            collector.end_call("nonexistent", "resp")

    def test_clear_resets_state(self) -> None:
        collector = TraceCollector()
        cid = collector.start_call("p", None, "leaf")
        collector.end_call(cid, "r")
        assert len(collector.get_events()) == 1
        collector.clear()
        assert len(collector.get_events()) == 0

    def test_get_events_returns_copy(self) -> None:
        collector = TraceCollector()
        cid = collector.start_call("p", None, "leaf")
        collector.end_call(cid, "r")
        assert collector.get_events() is not collector.get_events()
        assert collector.get_events() == collector.get_events()

    def test_call_event_is_frozen(self) -> None:
        collector = TraceCollector()
        cid = collector.start_call("p", None, "leaf")
        collector.end_call(cid, "r")
        with pytest.raises(AttributeError):
            collector.get_events()[0].prompt = "mutated"  # type: ignore[misc]

    def test_call_contexts(self) -> None:
        collector = TraceCollector()
        for ctx in ("leaf", "reduce", "filter"):
            cid = collector.start_call("p", None, ctx)
            collector.end_call(cid, "r")
        assert [e.call_context for e in collector.get_events()] == ["leaf", "reduce", "filter"]


# --- TreeTraceBuilder tests ---

@dataclass(frozen=True)
class FakePlan:
    """Minimal plan duck-type for builder tests."""

    k_star: int
    tau_star: int
    depth: int
    cost_estimate: float


class TestTreeTraceBuilder:
    """Unit tests for TreeTraceBuilder.build()."""

    def _make_event(
        self,
        prompt: str = "p",
        response: str = "r",
        call_context: str = "leaf",
        tokens_in: int = 10,
        tokens_out: int = 20,
        start_offset: float = 0.0,
        duration: float = 0.1,
    ) -> CallEvent:
        base = 1000.0
        return CallEvent(
            call_id=f"id_{prompt}_{call_context}",
            prompt=prompt,
            response=response,
            model="test-model",
            call_context=call_context,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            start_time=base + start_offset,
            end_time=base + start_offset + duration,
        )

    def test_k3_depth1_tree_structure(self) -> None:
        """k=3, depth=1: 3 leaf events + 1 reduce event."""
        plan = FakePlan(k_star=3, tau_star=100, depth=1, cost_estimate=5.0)

        leaf_events = [
            self._make_event(prompt=f"leaf_{i}", response=f"resp_{i}", call_context="leaf", start_offset=i * 0.1)
            for i in range(3)
        ]
        reduce_event = self._make_event(
            prompt="Merge these partial summaries into one",
            response="merged",
            call_context="reduce",
            start_offset=0.4,
        )
        events = leaf_events + [reduce_event]
        builder = TreeTraceBuilder()
        trace = builder.build(events, plan, "summarization", "merged", 1000.0, 1001.0)

        assert isinstance(trace, TreeTrace)
        assert (trace.task_type, trace.k, trace.depth, trace.tau) == ("summarization", 3, 1, 100)
        assert trace.final_output == "merged"
        # 3 leaf traces
        assert len(trace.leaf_traces) == 3
        for i, lt in enumerate(trace.leaf_traces):
            assert isinstance(lt, LeafTrace)
            assert lt.prompt == f"leaf_{i}" and lt.response == f"resp_{i}" and lt.model == "test-model"
        # 3 leaf nodes + 1 internal = 4 total
        assert len(trace.nodes) == 4
        leaf_nodes = [n for n in trace.nodes if n.combinator == "leaf"]
        internal_nodes = [n for n in trace.nodes if n.combinator != "leaf"]
        assert len(leaf_nodes) == 3 and len(internal_nodes) == 1
        root = internal_nodes[0]
        assert len(root.children) == 3 and root.depth == 0 and root.combinator == "reduce"
        assert trace.total_llm_calls == 4
        assert trace.total_tokens == (10 + 20) * 4

    def test_single_leaf_depth0(self) -> None:
        """tau >= n, depth=0: single leaf, no splitting."""
        plan = FakePlan(k_star=1, tau_star=500, depth=0, cost_estimate=1.0)
        event = self._make_event(prompt="the text", response="summary")
        builder = TreeTraceBuilder()
        trace = builder.build([event], plan, "summarization", "summary", 1000.0, 1000.5)
        assert trace.depth == 0 and trace.k == 1
        assert len(trace.leaf_traces) == 1 and len(trace.nodes) == 1
        node = trace.nodes[0]
        assert node.combinator == "leaf" and node.children == () and node.depth == 0
        assert trace.total_llm_calls == 1
        assert trace.execution_time_ms == pytest.approx(500.0)

    def test_empty_events(self) -> None:
        """Edge case: no events at all (should not crash)."""
        plan = FakePlan(k_star=2, tau_star=100, depth=0, cost_estimate=0.0)
        trace = TreeTraceBuilder().build([], plan, "general", "", 0.0, 0.0)
        assert trace.total_llm_calls == 0 and trace.total_tokens == 0

    def test_trace_id_is_unique(self) -> None:
        """Each build() call produces a unique trace_id."""
        plan = FakePlan(k_star=1, tau_star=100, depth=0, cost_estimate=1.0)
        ev = self._make_event()
        b = TreeTraceBuilder()
        ids = {b.build([ev], plan, "qa", "out", 0.0, 1.0).trace_id for _ in range(10)}
        assert len(ids) == 10

    def test_execution_time_computation(self) -> None:
        plan = FakePlan(k_star=1, tau_star=100, depth=0, cost_estimate=1.0)
        trace = TreeTraceBuilder().build([self._make_event()], plan, "qa", "out", 10.0, 12.5)
        assert trace.execution_time_ms == pytest.approx(2500.0)

    def test_token_aggregation(self) -> None:
        plan = FakePlan(k_star=2, tau_star=100, depth=1, cost_estimate=1.0)
        events = [
            self._make_event(tokens_in=10, tokens_out=20, call_context="leaf"),
            self._make_event(prompt="leaf2", tokens_in=15, tokens_out=25, call_context="leaf"),
            self._make_event(prompt="reduce", tokens_in=5, tokens_out=10, call_context="reduce"),
        ]
        trace = TreeTraceBuilder().build(events, plan, "qa", "out", 0.0, 1.0)
        assert trace.total_tokens == (10 + 20) + (15 + 25) + (5 + 10)


# --- patch_for_tracing tests ---

class TestPatchForTracing:
    """Unit tests for patch_for_tracing on mock objects."""

    def _make_mock_lrlm(self) -> SimpleNamespace:
        """Build a minimal mock that looks enough like LambdaRLM."""
        call_log: list[str] = []

        def mock_register_library(repl: Any, plan: Any, query: str = "") -> None:
            # Simulate what the real _register_library does:
            # capture repl._llm_query into a closure.
            captured_llm = repl._llm_query
            call_log.append("register_called")

            # Simulate a reduce closure that uses the captured llm.
            def mock_reduce(parts: list[str]) -> str:
                return captured_llm("Merge these partial summaries " + " ".join(parts))

            repl.globals["_Reduce"] = mock_reduce

        lrlm = SimpleNamespace(
            _register_library=mock_register_library,
        )
        lrlm._call_log = call_log
        return lrlm

    def _make_mock_repl(self) -> SimpleNamespace:
        """Build a minimal mock REPL."""
        def mock_llm_query(prompt: str, model: str | None = None) -> str:
            return f"LLM_RESPONSE({prompt[:30]})"

        repl = SimpleNamespace(
            _llm_query=mock_llm_query,
            globals={"llm_query": mock_llm_query},
        )
        return repl

    def _patch_and_register(self) -> tuple[SimpleNamespace, SimpleNamespace, TraceCollector]:
        """Helper: patch a mock lrlm, register library on a mock repl."""
        lrlm = self._make_mock_lrlm()
        patched, collector = patch_for_tracing(lrlm)
        repl = self._make_mock_repl()
        plan = FakePlan(k_star=3, tau_star=100, depth=1, cost_estimate=5.0)
        patched._register_library(repl, plan, "")
        return patched, repl, collector

    def test_patch_returns_same_object_and_collector(self) -> None:
        lrlm = self._make_mock_lrlm()
        patched, collector = patch_for_tracing(lrlm)
        assert patched is lrlm
        assert isinstance(collector, TraceCollector)

    def test_patch_does_not_crash(self) -> None:
        patched, _repl, _collector = self._patch_and_register()
        assert "register_called" in patched._call_log

    def test_traced_llm_query_records_events(self) -> None:
        _patched, repl, collector = self._patch_and_register()
        result = repl._llm_query("Summarize this text")
        assert "LLM_RESPONSE" in result
        events = collector.get_events()
        assert len(events) == 1
        assert events[0].prompt == "Summarize this text" and events[0].call_context == "leaf"

    def test_traced_reduce_records_events(self) -> None:
        _patched, repl, collector = self._patch_and_register()
        result = repl.globals["_Reduce"](["part1", "part2"])
        assert "LLM_RESPONSE" in result
        events = collector.get_events()
        assert len(events) == 1 and events[0].call_context == "reduce"

    def test_globals_llm_query_updated(self) -> None:
        _patched, repl, collector = self._patch_and_register()
        repl.globals["llm_query"]("Translate this text")
        events = collector.get_events()
        assert len(events) == 1
        assert events[0].prompt == "Translate this text" and events[0].call_context == "leaf"


# --- _infer_call_context tests ---

class TestInferCallContext:
    """Test the heuristic prompt classification."""

    def test_leaf_default(self) -> None:
        assert _infer_call_context("Summarize the following text") == "leaf"

    def test_reduce_merge_summaries(self) -> None:
        assert _infer_call_context(
            "Merge these partial summaries into one concise summary"
        ) == "reduce"

    def test_reduce_synthesise(self) -> None:
        assert _infer_call_context(
            "Synthesise these partial answers into one complete answer"
        ) == "reduce"

    def test_reduce_combine(self) -> None:
        assert _infer_call_context(
            "Combine these partial analyses into one report"
        ) == "reduce"

    def test_filter_relevance(self) -> None:
        assert _infer_call_context(
            "Does this excerpt contain information relevant to the question?\nReply YES or NO only."
        ) == "filter"

    def test_empty_string_is_leaf(self) -> None:
        assert _infer_call_context("") == "leaf"
