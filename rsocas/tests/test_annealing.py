"""Additional tests for AnnealingSchedule — deep cooling, phase boundaries, entropy."""

from __future__ import annotations

import pytest

from rsocas.breathing.annealing import AnnealingSchedule, AnnealingState


class TestAnnealingDeepCooling:
    """Tests for deep cooling behavior."""

    def test_100_cooling_steps_approaches_t_min(self) -> None:
        """After 100 cooling steps, temperature should approach t_min."""
        schedule = AnnealingSchedule(
            t_init=1.0, t_min=0.01, cooling_rate=0.95,
        )

        for _ in range(100):
            schedule.cool()

        # 0.95^100 ≈ 0.0059, so temperature should be near t_min
        assert schedule.temperature <= 0.02
        assert schedule.temperature >= 0.01  # never below t_min

    def test_100_cooling_steps_step_count(self) -> None:
        """Step count should track correctly over 100 steps."""
        schedule = AnnealingSchedule(t_init=1.0, t_min=0.01, cooling_rate=0.95)

        for _ in range(100):
            schedule.cool()

        state = schedule.get_state()
        assert state.step_count == 100

    def test_temperature_never_below_t_min(self) -> None:
        """Temperature should never go below t_min even after many steps."""
        schedule = AnnealingSchedule(
            t_init=1.0, t_min=0.05, cooling_rate=0.9,
        )

        for _ in range(200):
            schedule.cool()

        assert schedule.temperature >= 0.05


class TestAnnealingPhaseBoundary:
    """Tests for phase boundary detection with synthetic inflections."""

    def test_phase_boundary_with_synthetic_inflection(self) -> None:
        """Phase boundary detected when reheat creates an inflection point."""
        schedule = AnnealingSchedule(
            t_init=1.0, t_min=0.01, cooling_rate=0.95,
        )

        # Cool down to build monotonic decreasing history
        for _ in range(4):
            schedule.cool()

        # Now reheat to create an inflection point
        schedule.reheat(0.2)

        # Cool again to complete the inflection
        schedule.cool()
        schedule.cool()

        # The reheat followed by cooling creates a sign change
        # in second differences
        # History: [1.0, 0.95, 0.9025, 0.857, 0.814, ~1.014, ~0.963, ~0.915]
        # This should produce a detectable phase boundary
        state = schedule.get_state()
        assert state.at_phase_boundary is True

    def test_no_phase_boundary_monotonic_cooling(self) -> None:
        """No phase boundary during pure monotonic cooling."""
        schedule = AnnealingSchedule(
            t_init=1.0, t_min=0.01, cooling_rate=0.95,
        )

        # Pure geometric cooling: no inflection
        for _ in range(10):
            schedule.cool()

        assert not schedule.at_phase_boundary()

    def test_phase_boundary_requires_minimum_history(self) -> None:
        """Phase boundary detection requires at least 5 history points."""
        schedule = AnnealingSchedule(t_init=1.0, t_min=0.01, cooling_rate=0.95)

        # 1 point (initial)
        assert not schedule.at_phase_boundary()

        schedule.cool()  # 2 points
        assert not schedule.at_phase_boundary()

        schedule.cool()  # 3 points
        assert not schedule.at_phase_boundary()

        schedule.cool()  # 4 points
        assert not schedule.at_phase_boundary()


class TestAnnealingEntropyBudget:
    """Tests for entropy budget at various (temperature, staleness) combinations."""

    @pytest.mark.parametrize(
        "temperature,staleness,expected_budget",
        [
            (1.0, 0.0, 1.0),
            (1.0, 0.5, 1.5),
            (1.0, 1.0, 2.0),
            (0.5, 0.0, 0.5),
            (0.5, 1.0, 1.0),
            (0.1, 0.0, 0.1),
            (0.1, 2.0, 0.3),
        ],
    )
    def test_entropy_budget_combinations(
        self,
        temperature: float,
        staleness: float,
        expected_budget: float,
    ) -> None:
        """Verify entropy_budget = temperature * (1.0 + staleness)."""
        schedule = AnnealingSchedule(t_init=temperature, t_min=0.001)
        budget = schedule.entropy_budget(staleness=staleness)
        assert abs(budget - expected_budget) < 0.001

    def test_entropy_budget_zero_staleness_equals_temperature(self) -> None:
        """With zero staleness, budget equals temperature."""
        schedule = AnnealingSchedule(t_init=0.75)
        budget = schedule.entropy_budget(staleness=0.0)
        assert abs(budget - 0.75) < 0.001

    def test_entropy_budget_after_cooling(self) -> None:
        """Entropy budget decreases as temperature drops."""
        schedule = AnnealingSchedule(t_init=1.0, t_min=0.01, cooling_rate=0.9)
        staleness = 0.5

        budget_initial = schedule.entropy_budget(staleness)

        for _ in range(10):
            schedule.cool()

        budget_cooled = schedule.entropy_budget(staleness)

        assert budget_cooled < budget_initial

    def test_entropy_budget_after_reheat(self) -> None:
        """Entropy budget increases after reheat."""
        schedule = AnnealingSchedule(t_init=1.0, t_min=0.01, cooling_rate=0.9)
        staleness = 0.5

        for _ in range(10):
            schedule.cool()

        budget_before_reheat = schedule.entropy_budget(staleness)
        schedule.reheat(0.3)
        budget_after_reheat = schedule.entropy_budget(staleness)

        assert budget_after_reheat > budget_before_reheat


class TestAnnealingReheat:
    """Tests for reheat behavior."""

    def test_reheat_records_step(self) -> None:
        """Reheat should record the step at which it occurred."""
        schedule = AnnealingSchedule(t_init=1.0)

        for _ in range(5):
            schedule.cool()

        schedule.reheat(0.2)
        state = schedule.get_state()

        assert state.last_reheat_step == 5

    def test_no_reheat_step_initially(self) -> None:
        """Initially, last_reheat_step should be None."""
        schedule = AnnealingSchedule()
        state = schedule.get_state()
        assert state.last_reheat_step is None

    def test_multiple_reheats(self) -> None:
        """Multiple reheats should track latest step."""
        schedule = AnnealingSchedule(t_init=1.0, t_min=0.01, cooling_rate=0.9)

        for _ in range(5):
            schedule.cool()
        schedule.reheat(0.1)

        for _ in range(3):
            schedule.cool()
        schedule.reheat(0.1)

        state = schedule.get_state()
        assert state.last_reheat_step == 8
