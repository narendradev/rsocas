"""InterferencePattern — standing wave between human and system tempo.

Models constructive/destructive interference to determine when
the system should surface for human feedback.
"""

from __future__ import annotations

import time

from rsocas.contracts.evaluation import DisagreementSignal


class InterferencePattern:
    """Models constructive/destructive interference between human and system tempo.

    When human and system frequencies are close (constructive interference),
    the system is in sync with the human. When frequencies diverge
    (destructive interference), the system is drifting away.
    """

    def compute(self, human_freq: float, system_freq: float) -> float:
        """Compute interference amplitude.

        Uses beat frequency: beat = |human_freq - system_freq|.
        Amplitude = 1.0 / (1.0 + beat).

        Values > 0.5 indicate constructive interference (frequencies close).
        Values < 0.5 indicate destructive interference (frequencies far apart).

        Args:
            human_freq: Human feedback frequency (events per hour).
            system_freq: System event frequency (events per hour).

        Returns:
            Interference amplitude in [0, 1].
        """
        beat = abs(human_freq - system_freq)
        return 1.0 / (1.0 + beat)

    def should_surface(
        self,
        disagreement: DisagreementSignal,
        human_freq: float,
        system_freq: float,
        min_interval: float = 300.0,
        last_surface_time: float = 0.0,
        now: float | None = None,
    ) -> bool:
        """Should the system surface for human feedback right now?

        True if ALL of:
        1. disagreement.should_surface is True
        2. Interference is constructive (amplitude > 0.5)
        3. Enough time since last surface (> min_interval)

        Args:
            disagreement: Current disagreement signal from evaluators.
            human_freq: Human feedback frequency (events per hour).
            system_freq: System event frequency (events per hour).
            min_interval: Minimum seconds between surfacing events.
            last_surface_time: Timestamp of last surfacing event.
            now: Current time. Defaults to time.time().

        Returns:
            True if the system should surface for feedback now.
        """
        if now is None:
            now = time.time()

        # Condition 1: Disagreement says to surface
        if not disagreement.should_surface:
            return False

        # Condition 2: Constructive interference
        amplitude = self.compute(human_freq, system_freq)
        if amplitude <= 0.5:
            return False

        # Condition 3: Enough time since last surface
        if (now - last_surface_time) <= min_interval:
            return False

        return True
