"""BreathingCrystallizer -- wires the breathing cycle to combinator lifecycle.

One heartbeat (tick):
1. Check tempo -> should we crystallize or dissolve?
2. Check annealing -> are we at a phase boundary? Cool or hold.
3. Check disagreement -> should we reheat?
4. If reheat: dissolve stale combinators, raise temperature
5. If crystallize: promote fluid patterns that have been stable
6. If dissolve: expire stale combinators
"""

from __future__ import annotations

from dataclasses import dataclass

from rsocas.breathing.annealing import AnnealingSchedule
from rsocas.breathing.tempo import PIDTempoController
from rsocas.combinators.crystallizer import Crystallizer
from rsocas.contracts.evaluation import DisagreementSignal


@dataclass(frozen=True)
class BreathingEvent:
    """Immutable record of a single event during a breathing tick."""

    timestamp: float
    event_type: str  # "crystallized" | "dissolved" | "reheated" | "cooled" | "noop"
    combinator_name: str | None = None
    detail: str = ""


class BreathingCrystallizer:
    """Connects breathing cycle to combinator lifecycle.

    Orchestrates the interplay between tempo control (PID),
    annealing (temperature schedule), and the crystallizer
    (combinator state machine).  Each call to ``tick`` represents
    one heartbeat of the system.
    """

    def __init__(
        self,
        crystallizer: Crystallizer,
        tempo: PIDTempoController,
        annealing: AnnealingSchedule,
        reheat_threshold: float = 0.5,
    ) -> None:
        self._crystallizer = crystallizer
        self._tempo = tempo
        self._annealing = annealing
        self._reheat_threshold = reheat_threshold

    def tick(
        self,
        current_time: float,
        disagreement: DisagreementSignal | None = None,
    ) -> list[BreathingEvent]:
        """One heartbeat.  Returns events that occurred."""
        events: list[BreathingEvent] = []

        # Record system event
        self._tempo.record_system_event(current_time, "tick")

        # Cool the annealing schedule
        self._annealing.cool()

        # If high disagreement, reheat
        if disagreement and disagreement.magnitude >= self._reheat_threshold:
            self._annealing.reheat(disagreement.magnitude * 0.5)
            events.append(
                BreathingEvent(
                    timestamp=current_time,
                    event_type="reheated",
                    detail=f"magnitude={disagreement.magnitude:.2f}",
                )
            )

        # Check for expired combinators
        expired = self._crystallizer.tick(current_time)
        for vc in expired:
            events.append(
                BreathingEvent(
                    timestamp=current_time,
                    event_type="dissolved",
                    combinator_name=vc.name,
                )
            )

        # Should we crystallize?
        if self._tempo.should_crystallize() and self._annealing.temperature < 0.5:
            events.append(
                BreathingEvent(
                    timestamp=current_time,
                    event_type="cooled",
                    detail=f"temp={self._annealing.temperature:.3f}",
                )
            )

        if not events:
            events.append(
                BreathingEvent(
                    timestamp=current_time,
                    event_type="noop",
                )
            )

        return events

    def receive_human_feedback(self, timestamp: float) -> BreathingEvent:
        """Human feedback triggers controlled reheat + tempo update."""
        self._tempo.record_human_feedback(timestamp)
        self._annealing.reheat(0.2)
        return BreathingEvent(
            timestamp=timestamp,
            event_type="reheated",
            detail="human_feedback",
        )
