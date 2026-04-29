"""AnnealingSchedule — cooling with phase boundary detection.

Manages simulated annealing temperature for combinator crystallization.
Detects phase boundaries via second derivative analysis and adjusts
cooling rate accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnnealingState:
    """Immutable snapshot of annealing schedule state."""

    temperature: float
    step_count: int
    at_phase_boundary: bool
    last_reheat_step: int | None


class AnnealingSchedule:
    """Simulated annealing schedule with phase boundary detection.

    Supports controlled reheating triggered by high disagreement or
    human feedback. Slows cooling at detected phase boundaries.
    """

    def __init__(
        self,
        t_init: float = 1.0,
        t_min: float = 0.01,
        cooling_rate: float = 0.95,
    ) -> None:
        """Initialize annealing schedule.

        Args:
            t_init: Initial temperature.
            t_min: Minimum temperature floor.
            cooling_rate: Multiplicative cooling factor per step.
        """
        self._temperature = t_init
        self._t_init = t_init
        self._t_min = t_min
        self._cooling_rate = cooling_rate
        self._history: list[float] = [t_init]
        self._step = 0
        self._last_reheat: int | None = None

    @property
    def temperature(self) -> float:
        """Current temperature."""
        return self._temperature

    def cool(self) -> float:
        """Cool by one step. Slow down at phase boundaries.

        At phase boundary: use cooling_rate^0.5 instead of cooling_rate.
        This preserves interesting structures near phase transitions.

        Returns:
            New temperature after cooling.
        """
        self._step += 1

        if self.at_phase_boundary():
            # Gentler cooling at phase boundaries
            effective_rate = self._cooling_rate ** 0.5
        else:
            effective_rate = self._cooling_rate

        self._temperature = max(
            self._t_min,
            self._temperature * effective_rate,
        )
        self._history.append(self._temperature)

        return self._temperature

    def reheat(self, amount: float = 0.3) -> float:
        """Controlled reheat. Triggered by high disagreement or human feedback.

        Args:
            amount: Temperature increase amount.

        Returns:
            New temperature after reheating.
        """
        self._temperature = min(self._t_init, self._temperature + amount)
        self._last_reheat = self._step
        self._history.append(self._temperature)

        return self._temperature

    def at_phase_boundary(self) -> bool:
        """Detect phase boundary via second derivative of temperature trajectory.

        Phase boundary = inflection point where the second derivative
        changes sign. Requires at least 5 history points for reliable
        detection.

        Returns:
            True if currently at a phase boundary.
        """
        if len(self._history) < 5:
            return False

        # Compute second differences over recent history
        recent = self._history[-5:]

        first_diffs = [
            recent[i + 1] - recent[i] for i in range(len(recent) - 1)
        ]
        second_diffs = [
            first_diffs[i + 1] - first_diffs[i]
            for i in range(len(first_diffs) - 1)
        ]

        # Check for sign change in second differences
        for i in range(len(second_diffs) - 1):
            if second_diffs[i] * second_diffs[i + 1] < 0:
                return True

        return False

    def entropy_budget(self, staleness: float) -> float:
        """Residual entropy per combinator.

        Higher temperature -> higher budget (more exploration allowed).
        Higher staleness -> higher budget (stale combinators need more change).

        Args:
            staleness: Staleness score of the combinator (0.0 to 1.0+).

        Returns:
            Entropy budget value.
        """
        return self._temperature * (1.0 + staleness)

    def get_state(self) -> AnnealingState:
        """Return an immutable snapshot of current state."""
        return AnnealingState(
            temperature=self._temperature,
            step_count=self._step,
            at_phase_boundary=self.at_phase_boundary(),
            last_reheat_step=self._last_reheat,
        )
