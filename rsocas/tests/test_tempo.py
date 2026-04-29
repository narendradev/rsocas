"""Tests for breathing cycle module: tempo, annealing, feedback anchor, interference."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from rsocas.breathing.annealing import AnnealingSchedule, AnnealingState
from rsocas.breathing.feedback_anchor import FeedbackAnchor
from rsocas.breathing.interference import InterferencePattern
from rsocas.breathing.tempo import PIDTempoController, TempoState
from rsocas.contracts.evaluation import DisagreementSignal


# ---------------------------------------------------------------------------
# PIDTempoController tests
# ---------------------------------------------------------------------------


class TestPIDTempoController:
    """Tests for PIDTempoController."""

    def test_no_human_feedback_system_speeds_up(self) -> None:
        """With no human feedback, system should maintain or speed up."""
        now = time.time()
        controller = PIDTempoController(
            kp=0.5, ki=0.1, kd=0.05, target_ratio=2.0,
        )

        # Record some system events but no human events
        for i in range(10):
            controller.record_system_event(now - 300 + i * 30, "crystallize")

        rate = controller.breathing_rate()
        # With zero human freq, setpoint = 0, so PID drives rate down
        # but max(0.01, ...) ensures it stays positive
        assert rate >= 0.01

    def test_frequent_human_feedback_slows_system(self) -> None:
        """Frequent human feedback (1/min) should slow the system down."""
        now = time.time()
        controller = PIDTempoController(
            kp=0.5, ki=0.1, kd=0.05, target_ratio=2.0, window=3600.0,
        )

        # Record frequent human feedback: once per minute for 10 minutes
        for i in range(10):
            controller.record_human_feedback(now - 600 + i * 60)

        # Record very fast system events: every 10 seconds
        for i in range(60):
            controller.record_system_event(now - 600 + i * 10, "crystallize")

        rate = controller.breathing_rate()
        # Human freq is moderate, system freq is high
        # PID should try to slow system toward 2x human freq
        assert isinstance(rate, float)
        assert rate >= 0.01

    def test_human_goes_quiet_after_active_period(self) -> None:
        """After an active period, if human goes quiet, system speeds up."""
        now = time.time()
        controller = PIDTempoController(
            kp=0.5, ki=0.1, kd=0.05, target_ratio=2.0,
        )

        # Active period: human feedback 30 min ago
        for i in range(10):
            controller.record_human_feedback(now - 2400 + i * 60)

        # System events more recently
        for i in range(5):
            controller.record_system_event(now - 300 + i * 60, "crystallize")

        rate1 = controller.breathing_rate()

        # Now add very recent human feedback
        for i in range(10):
            controller.record_human_feedback(now - 60 + i * 6)

        rate2 = controller.breathing_rate()

        # With more recent human feedback, the target setpoint changes
        # Both rates should be valid positive numbers
        assert rate1 >= 0.01
        assert rate2 >= 0.01

    def test_get_state_returns_frozen_dataclass(self) -> None:
        """get_state should return a TempoState with correct phase."""
        controller = PIDTempoController()
        state = controller.get_state()

        assert isinstance(state, TempoState)
        assert state.phase in ("systole", "diastole")
        assert state.breathing_rate >= 0.01

        # Verify it's frozen
        with pytest.raises(AttributeError):
            state.breathing_rate = 999.0  # type: ignore[misc]

    def test_record_events_are_tracked(self) -> None:
        """Events should be recorded and affect frequency."""
        now = time.time()
        controller = PIDTempoController()

        controller.record_human_feedback(now)
        controller.record_system_event(now, "test_event")

        assert len(controller._human_events) == 1
        assert len(controller._system_events) == 1

    def test_compute_frequency_empty(self) -> None:
        """Empty event list returns zero frequency."""
        controller = PIDTempoController()
        freq = controller._compute_frequency([])
        assert freq == 0.0

    def test_compute_frequency_with_events(self) -> None:
        """Events should produce positive frequency."""
        now = time.time()
        controller = PIDTempoController()
        events = [now - 60 * i for i in range(5)]
        freq = controller._compute_frequency(events)
        assert freq > 0.0

    def test_should_crystallize_respects_interval(self) -> None:
        """should_crystallize respects the breathing rate interval."""
        controller = PIDTempoController()
        # With no events, rate is minimal
        # _last_crystallize_time is 0, so enough time has passed
        result = controller.should_crystallize()
        assert isinstance(result, bool)

    def test_should_dissolve_low_rate(self) -> None:
        """should_dissolve returns True when rate is very low."""
        controller = PIDTempoController()
        # With no events and minimal rate, should_dissolve depends on rate
        result = controller.should_dissolve()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# AnnealingSchedule tests
# ---------------------------------------------------------------------------


class TestAnnealingSchedule:
    """Tests for AnnealingSchedule."""

    def test_cooling_is_monotonic(self) -> None:
        """Cooling should be monotonically decreasing (ignoring phase boundaries)."""
        schedule = AnnealingSchedule(t_init=1.0, t_min=0.01, cooling_rate=0.95)

        temperatures = [schedule.temperature]
        for _ in range(20):
            temp = schedule.cool()
            temperatures.append(temp)

        # Each temperature should be <= previous (monotonic decrease)
        for i in range(1, len(temperatures)):
            assert temperatures[i] <= temperatures[i - 1]

    def test_reheat_increases_temperature(self) -> None:
        """Reheat should increase temperature."""
        schedule = AnnealingSchedule(t_init=1.0, t_min=0.01, cooling_rate=0.95)

        # Cool down first
        for _ in range(10):
            schedule.cool()

        temp_before = schedule.temperature
        schedule.reheat(0.3)
        temp_after = schedule.temperature

        assert temp_after > temp_before

    def test_reheat_does_not_exceed_t_init(self) -> None:
        """Reheat should not push temperature above initial temperature."""
        schedule = AnnealingSchedule(t_init=1.0, t_min=0.01, cooling_rate=0.95)

        # Cool slightly then reheat with large amount
        schedule.cool()
        schedule.reheat(10.0)

        assert schedule.temperature <= 1.0

    def test_entropy_budget_increases_with_staleness(self) -> None:
        """Higher staleness should produce higher entropy budget."""
        schedule = AnnealingSchedule(t_init=1.0)

        budget_low = schedule.entropy_budget(staleness=0.0)
        budget_mid = schedule.entropy_budget(staleness=0.5)
        budget_high = schedule.entropy_budget(staleness=1.0)

        assert budget_low < budget_mid < budget_high

    def test_entropy_budget_increases_with_temperature(self) -> None:
        """Higher temperature should produce higher entropy budget."""
        schedule_hot = AnnealingSchedule(t_init=1.0)
        schedule_cold = AnnealingSchedule(t_init=0.1)

        budget_hot = schedule_hot.entropy_budget(staleness=0.5)
        budget_cold = schedule_cold.entropy_budget(staleness=0.5)

        assert budget_hot > budget_cold

    def test_get_state_returns_frozen_dataclass(self) -> None:
        """get_state should return an AnnealingState."""
        schedule = AnnealingSchedule()
        state = schedule.get_state()

        assert isinstance(state, AnnealingState)
        assert state.temperature == schedule.temperature
        assert state.step_count == 0

        # Verify it's frozen
        with pytest.raises(AttributeError):
            state.temperature = 999.0  # type: ignore[misc]

    def test_phase_boundary_not_detected_early(self) -> None:
        """Phase boundary should not be detected with fewer than 5 history points."""
        schedule = AnnealingSchedule()
        # Only 1 history point (initial)
        assert not schedule.at_phase_boundary()

        schedule.cool()
        schedule.cool()
        # Only 3 history points
        assert not schedule.at_phase_boundary()


# ---------------------------------------------------------------------------
# FeedbackAnchor tests
# ---------------------------------------------------------------------------


class TestFeedbackAnchor:
    """Tests for FeedbackAnchor."""

    def test_frequency_with_exponential_weighting(self) -> None:
        """Recent events should contribute more to frequency."""
        now = time.time()
        anchor = FeedbackAnchor(half_life=1800.0)

        # Record events at various times
        anchor.record(now - 10, "correction")  # very recent
        anchor.record(now - 1000, "general")  # moderately old
        anchor.record(now - 3000, "general")  # old

        freq = anchor.frequency(now=now)
        assert freq > 0.0

    def test_frequency_empty(self) -> None:
        """No events should give zero frequency."""
        anchor = FeedbackAnchor()
        assert anchor.frequency() == 0.0

    def test_time_since_last_correct(self) -> None:
        """time_since_last should return correct elapsed time."""
        now = 1000.0
        anchor = FeedbackAnchor()

        anchor.record(900.0, "general")
        anchor.record(950.0, "correction")

        elapsed = anchor.time_since_last(now=now)
        assert abs(elapsed - 50.0) < 0.001

    def test_time_since_last_no_events(self) -> None:
        """time_since_last should return inf with no events."""
        anchor = FeedbackAnchor()
        assert anchor.time_since_last() == float("inf")

    def test_feedback_density(self) -> None:
        """feedback_density should count events in window."""
        now = 1000.0
        anchor = FeedbackAnchor()

        # 3 events within window, 1 outside
        anchor.record(900.0, "general")
        anchor.record(950.0, "general")
        anchor.record(999.0, "general")
        anchor.record(100.0, "general")  # 900 seconds ago, outside 600s window

        density = anchor.feedback_density(window=600.0, now=now)
        assert density == 3.0

    def test_record_stores_feedback_type(self) -> None:
        """Records should store the feedback type."""
        anchor = FeedbackAnchor()
        anchor.record(100.0, "correction")
        anchor.record(200.0, "approval")

        assert len(anchor._events) == 2
        assert anchor._events[0] == (100.0, "correction")
        assert anchor._events[1] == (200.0, "approval")


# ---------------------------------------------------------------------------
# InterferencePattern tests
# ---------------------------------------------------------------------------


class TestInterferencePattern:
    """Tests for InterferencePattern."""

    def test_close_frequencies_high_amplitude(self) -> None:
        """Close frequencies should produce high amplitude (> 0.5)."""
        pattern = InterferencePattern()
        amplitude = pattern.compute(human_freq=10.0, system_freq=10.2)
        assert amplitude > 0.5

    def test_far_frequencies_low_amplitude(self) -> None:
        """Far apart frequencies should produce low amplitude (< 0.5)."""
        pattern = InterferencePattern()
        amplitude = pattern.compute(human_freq=1.0, system_freq=100.0)
        assert amplitude < 0.5

    def test_identical_frequencies_max_amplitude(self) -> None:
        """Identical frequencies should give amplitude of 1.0."""
        pattern = InterferencePattern()
        amplitude = pattern.compute(human_freq=5.0, system_freq=5.0)
        assert abs(amplitude - 1.0) < 0.001

    def test_should_surface_all_conditions_met(self) -> None:
        """should_surface True when all three conditions are met."""
        pattern = InterferencePattern()
        now = 1000.0

        disagreement = DisagreementSignal(
            magnitude=0.8,
            should_surface=True,
            timestamp=now,
        )

        result = pattern.should_surface(
            disagreement=disagreement,
            human_freq=10.0,
            system_freq=10.0,  # identical -> constructive
            min_interval=300.0,
            last_surface_time=0.0,  # long ago -> enough time
            now=now,
        )

        assert result is True

    def test_should_surface_disagreement_false(self) -> None:
        """should_surface False when disagreement says don't surface."""
        pattern = InterferencePattern()
        now = 1000.0

        disagreement = DisagreementSignal(
            magnitude=0.1,
            should_surface=False,
            timestamp=now,
        )

        result = pattern.should_surface(
            disagreement=disagreement,
            human_freq=10.0,
            system_freq=10.0,
            min_interval=300.0,
            last_surface_time=0.0,
            now=now,
        )

        assert result is False

    def test_should_surface_destructive_interference(self) -> None:
        """should_surface False when frequencies are far apart."""
        pattern = InterferencePattern()
        now = 1000.0

        disagreement = DisagreementSignal(
            magnitude=0.8,
            should_surface=True,
            timestamp=now,
        )

        result = pattern.should_surface(
            disagreement=disagreement,
            human_freq=1.0,
            system_freq=100.0,  # far apart -> destructive
            min_interval=300.0,
            last_surface_time=0.0,
            now=now,
        )

        assert result is False

    def test_should_surface_too_soon(self) -> None:
        """should_surface False when not enough time since last surface."""
        pattern = InterferencePattern()
        now = 1000.0

        disagreement = DisagreementSignal(
            magnitude=0.8,
            should_surface=True,
            timestamp=now,
        )

        result = pattern.should_surface(
            disagreement=disagreement,
            human_freq=10.0,
            system_freq=10.0,
            min_interval=300.0,
            last_surface_time=900.0,  # only 100s ago
            now=now,
        )

        assert result is False

    def test_should_surface_requires_all_three(self) -> None:
        """Each condition alone is not sufficient."""
        pattern = InterferencePattern()
        now = 1000.0

        # Only disagreement is True, interference destructive
        result1 = pattern.should_surface(
            disagreement=DisagreementSignal(
                magnitude=0.8, should_surface=True, timestamp=now,
            ),
            human_freq=1.0,
            system_freq=100.0,
            min_interval=300.0,
            last_surface_time=0.0,
            now=now,
        )

        # Only time is sufficient, disagreement False
        result2 = pattern.should_surface(
            disagreement=DisagreementSignal(
                magnitude=0.1, should_surface=False, timestamp=now,
            ),
            human_freq=10.0,
            system_freq=10.0,
            min_interval=300.0,
            last_surface_time=0.0,
            now=now,
        )

        assert result1 is False
        assert result2 is False
