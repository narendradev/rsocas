"""PIDTempoController — PID controller anchored to human feedback frequency.

When humans intervene often, slow down and keep combinators fluid.
When humans go quiet, speed up crystallization.

The human's own behavior IS the tempo controller.
External signal — no Godelian regress.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class TempoState:
    """Immutable snapshot of tempo controller state."""

    breathing_rate: float  # target crystallizations per hour
    human_freq: float  # measured human feedback frequency
    system_freq: float  # measured system event frequency
    error: float  # PID error term
    phase: str  # "systole" | "diastole"


class PIDTempoController:
    """PID controller for breathing rate.

    When humans intervene often -> slow down, keep combinators fluid.
    When humans go quiet -> speed up crystallization.

    The human's own behavior IS the tempo controller.
    External signal -- no Godelian regress.
    """

    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.05,
        target_ratio: float = 2.0,
        window: float = 3600.0,
        half_life: float = 900.0,
    ) -> None:
        """Initialize PID tempo controller.

        Args:
            kp: Proportional gain.
            ki: Integral gain.
            kd: Derivative gain.
            target_ratio: System breathes target_ratio times faster
                          than human feedback.
            window: Time window in seconds for frequency computation.
            half_life: Half-life in seconds for exponential weighting.
        """
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._target_ratio = target_ratio
        self._window = window
        self._half_life = half_life

        self._human_events: list[float] = []
        self._system_events: list[float] = []

        self._integral: float = 0.0
        self._prev_error: float | None = None
        self._last_pid_time: float | None = None
        self._last_crystallize_time: float = 0.0

    def record_human_feedback(self, timestamp: float) -> None:
        """Record a human feedback event."""
        self._human_events.append(timestamp)

    def record_system_event(self, timestamp: float, event: str) -> None:
        """Record a system event (crystallization, dissolution, etc.)."""
        self._system_events.append(timestamp)

    def breathing_rate(self) -> float:
        """Compute target breathing rate via PID.

        setpoint = human_freq * target_ratio
        error = setpoint - current_system_freq
        output = kp*error + ki*integral + kd*derivative
        Return max(0.01, system_freq + output)
        """
        now = time.time()

        human_freq = self._compute_frequency(self._human_events)
        system_freq = self._compute_frequency(self._system_events)

        setpoint = human_freq * self._target_ratio
        error = setpoint - system_freq

        # Compute PID terms
        dt = 1.0
        if self._last_pid_time is not None:
            elapsed = now - self._last_pid_time
            if elapsed > 0:
                dt = elapsed

        self._integral += error * dt

        derivative = 0.0
        if self._prev_error is not None:
            derivative = (error - self._prev_error) / dt

        self._prev_error = error
        self._last_pid_time = now

        output = (
            self._kp * error
            + self._ki * self._integral
            + self._kd * derivative
        )

        return max(0.01, system_freq + output)

    def should_crystallize(self) -> bool:
        """True if breathing rate suggests crystallization is due."""
        rate = self.breathing_rate()
        if rate <= 0.01:
            return False

        # Interval in seconds between crystallizations
        interval = 3600.0 / rate
        now = time.time()
        return (now - self._last_crystallize_time) >= interval

    def should_dissolve(self) -> bool:
        """True if breathing rate suggests dissolution is due.

        Dissolution happens when system is running too fast relative
        to the human tempo -- i.e., breathing rate is very low.
        """
        rate = self.breathing_rate()
        return rate < 0.5

    def get_state(self) -> TempoState:
        """Return an immutable snapshot of current state."""
        human_freq = self._compute_frequency(self._human_events)
        system_freq = self._compute_frequency(self._system_events)
        rate = self.breathing_rate()
        error = self._prev_error if self._prev_error is not None else 0.0

        phase = "systole" if rate >= system_freq else "diastole"

        return TempoState(
            breathing_rate=rate,
            human_freq=human_freq,
            system_freq=system_freq,
            error=error,
            phase=phase,
        )

    def _compute_frequency(
        self,
        events: list[float],
        window: float | None = None,
    ) -> float:
        """Exponentially-weighted frequency: recent events count more.

        Each event contributes weight = exp(-(now - event_time) / half_life).
        Frequency = sum(weights) / window * 3600 (events per hour).
        """
        if not events:
            return 0.0

        now = time.time()
        effective_window = window if window is not None else self._window

        total_weight = 0.0
        for event_time in events:
            age = now - event_time
            if age < 0:
                age = 0.0
            if age <= effective_window:
                weight = math.exp(-age / self._half_life)
                total_weight += weight

        if effective_window <= 0:
            return 0.0

        # Convert to events per hour
        return (total_weight / effective_window) * 3600.0
