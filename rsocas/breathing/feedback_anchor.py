"""FeedbackAnchor — tracks human feedback events.

Provides the external anchor signal that grounds the breathing cycle.
Human feedback frequency is the reference clock against which system
tempo is measured.
"""

from __future__ import annotations

import math
import time


class FeedbackAnchor:
    """Tracks human feedback timestamps. Provides the external anchor signal.

    Uses exponentially-weighted frequency computation so recent feedback
    events count more than older ones.
    """

    def __init__(self, half_life: float = 1800.0) -> None:
        """Initialize feedback anchor.

        Args:
            half_life: Seconds for exponential weighting (30 min default).
        """
        self._events: list[tuple[float, str]] = []
        self._half_life = half_life

    def record(
        self,
        timestamp: float,
        feedback_type: str = "general",
    ) -> None:
        """Record a human feedback event.

        Args:
            timestamp: Unix timestamp of the event.
            feedback_type: Category of feedback (e.g. "general", "correction").
        """
        self._events.append((timestamp, feedback_type))

    def frequency(self, now: float | None = None) -> float:
        """Exponentially-weighted feedback frequency (events per hour).

        Each event contributes weight = exp(-(now - event_time) / half_life).
        Frequency = sum(weights) / window * 3600.

        Args:
            now: Current time. Defaults to time.time().

        Returns:
            Weighted frequency in events per hour.
        """
        if not self._events:
            return 0.0

        if now is None:
            now = time.time()

        total_weight = 0.0
        window = self._half_life * 2  # Use 2x half_life as effective window

        for event_time, _ in self._events:
            age = now - event_time
            if age < 0:
                age = 0.0
            if age <= window:
                weight = math.exp(-age / self._half_life)
                total_weight += weight

        if window <= 0:
            return 0.0

        return (total_weight / window) * 3600.0

    def time_since_last(self, now: float | None = None) -> float:
        """Seconds since last human feedback.

        Args:
            now: Current time. Defaults to time.time().

        Returns:
            Seconds since last feedback, or float('inf') if never.
        """
        if not self._events:
            return float("inf")

        if now is None:
            now = time.time()

        last_time = max(event_time for event_time, _ in self._events)
        return now - last_time

    def feedback_density(
        self,
        window: float = 3600.0,
        now: float | None = None,
    ) -> float:
        """Raw count of events in window.

        Args:
            window: Time window in seconds (default 1 hour).
            now: Current time. Defaults to time.time().

        Returns:
            Number of feedback events within the window.
        """
        if now is None:
            now = time.time()

        count = 0.0
        for event_time, _ in self._events:
            age = now - event_time
            if 0 <= age <= window:
                count += 1.0

        return count
