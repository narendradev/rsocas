"""Tests for BreathingCrystallizer -- wiring between breathing cycle and combinator lifecycle."""

from __future__ import annotations

import time

import pytest

from rsocas.breathing.annealing import AnnealingSchedule
from rsocas.breathing.breathing_crystallizer import BreathingCrystallizer, BreathingEvent
from rsocas.breathing.tempo import PIDTempoController
from rsocas.combinators.crystallizer import Crystallizer
from rsocas.combinators.penumbra import PenumbraStore
from rsocas.combinators.versioned import CombinatorDB
from rsocas.contracts.combinators import ValidationSnapshot
from rsocas.contracts.evaluation import DisagreementSignal


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_validation(
    mean: float = 0.85,
    std: float = 0.05,
    n: int = 100,
) -> ValidationSnapshot:
    return ValidationSnapshot(
        task_types=("qa",),
        input_size_range=(10, 500),
        n_samples=n,
        mean_score=mean,
        score_std=std,
        timestamp=time.time(),
    )


def _make_components(
    default_ttl: float = 86400.0,
    t_init: float = 1.0,
    cooling_rate: float = 0.95,
    reheat_threshold: float = 0.5,
) -> tuple[BreathingCrystallizer, Crystallizer, PIDTempoController, AnnealingSchedule]:
    db = CombinatorDB(":memory:")
    penumbra = PenumbraStore(db)
    crystallizer = Crystallizer(db, penumbra, default_ttl=default_ttl)
    tempo = PIDTempoController()
    annealing = AnnealingSchedule(
        t_init=t_init,
        cooling_rate=cooling_rate,
    )
    bc = BreathingCrystallizer(
        crystallizer=crystallizer,
        tempo=tempo,
        annealing=annealing,
        reheat_threshold=reheat_threshold,
    )
    return bc, crystallizer, tempo, annealing


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestTickNoop:
    def test_tick_noop_on_no_events(self) -> None:
        """Tick with no disagreement and no expired combinators yields noop."""
        bc, _crystallizer, _tempo, _annealing = _make_components()
        now = time.time()

        events = bc.tick(now)

        assert len(events) == 1
        assert events[0].event_type == "noop"
        assert events[0].timestamp == now


class TestTickReheat:
    def test_tick_reheat_on_high_disagreement(self) -> None:
        """Disagreement with magnitude >= threshold triggers reheated event."""
        bc, _crystallizer, _tempo, _annealing = _make_components(reheat_threshold=0.5)
        now = time.time()

        disagreement = DisagreementSignal(
            magnitude=0.8,
            pairwise={"a_vs_b": 0.8},
            per_node={},
            outlier_voice="a",
            should_surface=True,
            timestamp=now,
        )

        events = bc.tick(now, disagreement=disagreement)

        event_types = [e.event_type for e in events]
        assert "reheated" in event_types

        reheated = [e for e in events if e.event_type == "reheated"][0]
        assert "magnitude=0.80" in reheated.detail

    def test_tick_no_reheat_on_low_disagreement(self) -> None:
        """Disagreement below threshold does not trigger reheat."""
        bc, _crystallizer, _tempo, _annealing = _make_components(reheat_threshold=0.5)
        now = time.time()

        disagreement = DisagreementSignal(
            magnitude=0.3,
            pairwise={},
            per_node={},
            should_surface=False,
            timestamp=now,
        )

        events = bc.tick(now, disagreement=disagreement)

        event_types = [e.event_type for e in events]
        assert "reheated" not in event_types


class TestTickDissolve:
    def test_tick_dissolves_expired(self) -> None:
        """A combinator past its TTL gets dissolved on tick."""
        bc, crystallizer, _tempo, _annealing = _make_components(default_ttl=1.0)

        # Create a crystallized combinator with a very short TTL
        validation = _make_validation()
        vc = crystallizer.crystallize("_Reduce", lambda x: x, validation)

        # Tick at a time well past the TTL
        future_time = vc.expires_at + 10.0
        events = bc.tick(future_time)

        event_types = [e.event_type for e in events]
        assert "dissolved" in event_types

        dissolved = [e for e in events if e.event_type == "dissolved"][0]
        assert dissolved.combinator_name == "_Reduce"


class TestReceiveHumanFeedback:
    def test_receive_human_feedback_reheats(self) -> None:
        """Human feedback triggers reheat and tempo update."""
        bc, _crystallizer, tempo, annealing = _make_components(
            t_init=0.5,
            cooling_rate=0.9,
        )

        # Cool down first
        for _ in range(5):
            annealing.cool()

        temp_before = annealing.temperature
        now = time.time()

        event = bc.receive_human_feedback(now)

        assert event.event_type == "reheated"
        assert event.detail == "human_feedback"
        assert event.timestamp == now

        # Temperature should have increased
        assert annealing.temperature > temp_before


class TestTickCoolsAnnealing:
    def test_tick_cools_annealing(self) -> None:
        """Each tick cools the annealing temperature."""
        bc, _crystallizer, _tempo, annealing = _make_components(
            t_init=1.0,
            cooling_rate=0.9,
        )

        initial_temp = annealing.temperature
        now = time.time()

        bc.tick(now)

        assert annealing.temperature < initial_temp

    def test_multiple_ticks_decrease_temperature(self) -> None:
        """Multiple ticks should progressively decrease temperature."""
        bc, _crystallizer, _tempo, annealing = _make_components(
            t_init=1.0,
            cooling_rate=0.9,
        )

        temps: list[float] = [annealing.temperature]
        now = time.time()

        for i in range(5):
            bc.tick(now + i)
            temps.append(annealing.temperature)

        # Each temperature should be less than or equal to the previous
        for i in range(1, len(temps)):
            assert temps[i] <= temps[i - 1]


class TestMultipleTicksBreathingRhythm:
    def test_multiple_ticks_breathing_rhythm(self) -> None:
        """Run 10 ticks and verify we get a coherent pattern of events."""
        bc, crystallizer, _tempo, annealing = _make_components(
            t_init=1.0,
            cooling_rate=0.9,
            default_ttl=5.0,
        )

        # Create a crystallized combinator that will expire during the run
        validation = _make_validation()
        vc = crystallizer.crystallize("_Split", lambda x: x, validation)

        now = time.time()
        all_events: list[BreathingEvent] = []

        for i in range(10):
            tick_time = vc.expires_at + i  # Start past TTL
            events = bc.tick(tick_time)
            all_events.extend(events)

        # Should have at least one dissolved event (from the expired combinator)
        event_types = [e.event_type for e in all_events]
        assert "dissolved" in event_types

        # Should have at least one noop (once expired combinator is processed)
        assert "noop" in event_types

        # All events should have valid timestamps
        for event in all_events:
            assert event.timestamp >= vc.expires_at


class TestBreathingEventImmutability:
    def test_breathing_event_is_frozen(self) -> None:
        """BreathingEvent should be immutable (frozen dataclass)."""
        event = BreathingEvent(
            timestamp=1.0,
            event_type="noop",
        )
        with pytest.raises(AttributeError):
            event.event_type = "reheated"  # type: ignore[misc]
