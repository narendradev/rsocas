"""TreeTraceBuilder — reconstruct a TreeTrace from flat CallEvents.

The builder takes the ordered list of ``CallEvent`` objects captured by
``TraceCollector`` and combines them with the ``LambdaPlan`` to produce a
``TreeTrace`` that faithfully represents the execution tree.

Tree-structure inference
~~~~~~~~~~~~~~~~~~~~~~~~
Lambda-RLM executes depth-first, left-to-right.  Given ``k`` (branching
factor) and ``depth``:

* The first ``k^depth`` events with ``call_context == "leaf"`` are the
  leaf calls.
* Subsequent events are ``"reduce"`` or ``"filter"`` calls that correspond
  to internal nodes.

When ``depth == 0`` (input fits in one context window), there is a single
leaf and no internal nodes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from rsocas.contracts.traces import LeafTrace, NodeTrace, TreeTrace
from rsocas.tracing.collector import CallEvent


class TreeTraceBuilder:
    """Construct a ``TreeTrace`` from flat call events and a plan."""

    def build(
        self,
        events: list[CallEvent],
        plan: _PlanLike,
        task_type: str,
        final_output: str,
        start_time: float,
        end_time: float,
    ) -> TreeTrace:
        """Build a ``TreeTrace`` from the recorded events.

        Args:
            events: Ordered list of ``CallEvent`` produced by a collector.
            plan: An object exposing ``k_star``, ``tau_star``, ``depth``,
                  and ``cost_estimate`` (e.g. a ``LambdaPlan``).
            task_type: Task type label (``"summarization"``, ``"qa"``, ...).
            final_output: The final text output of the Lambda-RLM run.
            start_time: ``time.monotonic()`` at the start of execution.
            end_time: ``time.monotonic()`` at the end of execution.

        Returns:
            A frozen ``TreeTrace`` instance.
        """
        k = plan.k_star
        depth = plan.depth
        tau = plan.tau_star
        cost_estimate = plan.cost_estimate

        leaf_events = [e for e in events if e.call_context == "leaf"]
        internal_events = [
            e for e in events if e.call_context in ("reduce", "filter")
        ]

        # --- Build LeafTrace objects ---
        leaf_traces: list[LeafTrace] = []
        leaf_node_ids: list[str] = []
        for idx, ev in enumerate(leaf_events):
            node_id = uuid.uuid4().hex
            leaf_node_ids.append(node_id)
            leaf_traces.append(
                LeafTrace(
                    node_id=node_id,
                    prompt=ev.prompt,
                    response=ev.response,
                    tokens_in=ev.tokens_in,
                    tokens_out=ev.tokens_out,
                    model=ev.model or "",
                )
            )

        # --- Build NodeTrace objects ---
        nodes: list[NodeTrace] = []

        if depth == 0:
            # Single leaf: one node, no children, no internal events.
            nid = leaf_node_ids[0] if leaf_node_ids else uuid.uuid4().hex
            nodes.append(
                NodeTrace(
                    id=nid,
                    depth=0,
                    position=0,
                    combinator="leaf",
                    input_size=tau,
                    output=final_output,
                    children=(),
                    llm_calls=1 if leaf_events else 0,
                    latency_ms=_event_latency_ms(leaf_events[0])
                    if leaf_events
                    else 0.0,
                )
            )
        else:
            # Build bottom-up.  Current layer starts as the leaf ids.
            current_layer_ids = list(leaf_node_ids)

            # Create a leaf NodeTrace for each leaf.
            for pos, lid in enumerate(leaf_node_ids):
                ev = leaf_events[pos] if pos < len(leaf_events) else None
                nodes.append(
                    NodeTrace(
                        id=lid,
                        depth=depth,
                        position=pos,
                        combinator="leaf",
                        input_size=tau,
                        output=ev.response if ev else "",
                        children=(),
                        llm_calls=1,
                        latency_ms=_event_latency_ms(ev) if ev else 0.0,
                    )
                )

            # Walk upward through each internal level.
            internal_idx = 0
            for d in range(depth - 1, -1, -1):
                next_layer_ids: list[str] = []
                # Group current_layer_ids into chunks of k.
                groups = _chunk_list(current_layer_ids, k)
                for pos, group in enumerate(groups):
                    nid = uuid.uuid4().hex
                    next_layer_ids.append(nid)

                    ie = (
                        internal_events[internal_idx]
                        if internal_idx < len(internal_events)
                        else None
                    )
                    combinator = ie.call_context if ie else "reduce"
                    latency = _event_latency_ms(ie) if ie else 0.0
                    llm_calls = 1 if ie else 0
                    output_text = ie.response if ie else ""
                    internal_idx += 1

                    nodes.append(
                        NodeTrace(
                            id=nid,
                            depth=d,
                            position=pos,
                            combinator=combinator,
                            input_size=len(group),
                            output=output_text,
                            children=tuple(group),
                            llm_calls=llm_calls,
                            latency_ms=latency,
                        )
                    )

                current_layer_ids = next_layer_ids

        # --- Aggregate totals ---
        total_llm_calls = len(events)
        total_tokens = sum(e.tokens_in + e.tokens_out for e in events)
        execution_time_ms = (end_time - start_time) * 1000.0

        return TreeTrace(
            trace_id=uuid.uuid4().hex,
            task_type=task_type,
            k=k,
            depth=depth,
            tau=tau,
            cost_estimate=cost_estimate,
            nodes=tuple(nodes),
            leaf_traces=tuple(leaf_traces),
            final_output=final_output,
            timestamp=start_time,
            execution_time_ms=execution_time_ms,
            total_llm_calls=total_llm_calls,
            total_tokens=total_tokens,
        )


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _event_latency_ms(event: CallEvent) -> float:
    """Compute latency in milliseconds from a single ``CallEvent``."""
    return (event.end_time - event.start_time) * 1000.0


def _chunk_list(items: list[str], k: int) -> list[list[str]]:
    """Split *items* into groups of at most *k* elements."""
    if k <= 0:
        return [items] if items else []
    return [items[i : i + k] for i in range(0, len(items), k)]


# ------------------------------------------------------------------
# Structural type for the plan parameter (avoids importing lambda_rlm)
# ------------------------------------------------------------------


@dataclass(frozen=True)
class _PlanLike:
    """Minimal structural expectation for the *plan* argument.

    Any object with these four attributes will work (duck typing).
    This dataclass is here solely for documentation — callers may pass
    a real ``LambdaPlan`` or any compatible object.
    """

    k_star: int
    tau_star: int
    depth: int
    cost_estimate: float
