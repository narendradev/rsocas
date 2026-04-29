"""TraceCollector — records LLM call events during Lambda-RLM execution.

Each call to the traced ``_llm_query`` wrapper produces a pair of events
(start + end) captured as a single frozen ``CallEvent`` once the call
completes.  Events are stored in order of *completion* and are later fed
to ``TreeTraceBuilder`` to reconstruct the execution tree.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class CallEvent:
    """One completed LLM call captured by the collector."""

    call_id: str
    prompt: str
    response: str
    model: str | None
    call_context: str  # "leaf", "reduce", or "filter"
    tokens_in: int
    tokens_out: int
    start_time: float
    end_time: float


class TraceCollector:
    """Thread-safe accumulator for ``CallEvent`` objects.

    Typical lifecycle::

        collector = TraceCollector()
        cid = collector.start_call(prompt, model, "leaf")
        # ... LLM call happens ...
        collector.end_call(cid, response, tokens_in, tokens_out)
        events = collector.get_events()
    """

    def __init__(self) -> None:
        self._pending: dict[str, _PendingCall] = {}
        self._events: list[CallEvent] = []
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_call(
        self,
        prompt: str,
        model: str | None = None,
        call_context: str = "leaf",
    ) -> str:
        """Record the start of an LLM call.

        Args:
            prompt: The prompt text sent to the model.
            model: Optional model identifier.
            call_context: One of ``"leaf"``, ``"reduce"``, or ``"filter"``.

        Returns:
            A unique *call_id* (UUID4 hex) to pass to :meth:`end_call`.
        """
        call_id = uuid.uuid4().hex
        pending = _PendingCall(
            call_id=call_id,
            prompt=prompt,
            model=model,
            call_context=call_context,
            start_time=time.monotonic(),
        )
        with self._lock:
            self._pending[call_id] = pending
        return call_id

    def end_call(
        self,
        call_id: str,
        response: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        """Record the completion of a previously-started LLM call.

        Args:
            call_id: The id returned by :meth:`start_call`.
            response: The model's response text.
            tokens_in: Input token count (0 if unknown).
            tokens_out: Output token count (0 if unknown).

        Raises:
            KeyError: If *call_id* was never started or already ended.
        """
        end_time = time.monotonic()
        with self._lock:
            pending = self._pending.pop(call_id)
            event = CallEvent(
                call_id=pending.call_id,
                prompt=pending.prompt,
                response=response,
                model=pending.model,
                call_context=pending.call_context,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                start_time=pending.start_time,
                end_time=end_time,
            )
            self._events.append(event)

    def get_events(self) -> list[CallEvent]:
        """Return a copy of all completed events in completion order."""
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        """Reset the collector, discarding all events and pending calls."""
        with self._lock:
            self._pending.clear()
            self._events.clear()


# ------------------------------------------------------------------
# Internal helper — mutable staging area (not part of public API)
# ------------------------------------------------------------------


@dataclass
class _PendingCall:
    """Mutable staging record for a call that has started but not ended."""

    call_id: str
    prompt: str
    model: str | None
    call_context: str
    start_time: float
