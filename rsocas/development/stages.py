"""Developmental stages and transition controller.

The system progresses through irreversible developmental stages,
each enabling additional capabilities. Transitions are triggered
by measurable metrics -- the system grows up as it accumulates
evidence of readiness.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class DevelopmentalStage(IntEnum):
    """Progressive developmental stages of the RSOCAS system.

    Each stage enables additional features. Transitions are irreversible.
    """

    EMBRYONIC = 0       # Only Lambda-RLM execution, no evaluation
    FETAL = 1           # + contrapuntal evaluation
    BORN = 2            # + breathing cycle
    CHILDHOOD = 3       # + penumbra variants + archive
    ADOLESCENCE = 4     # + GEPA optimization (future -- stub for now)
    ADULT = 5           # + full autonomy with human anchoring (future -- stub)


_FEATURES_BY_STAGE: dict[DevelopmentalStage, set[str]] = {
    DevelopmentalStage.EMBRYONIC: {"execution"},
    DevelopmentalStage.FETAL: {"execution", "evaluation"},
    DevelopmentalStage.BORN: {"execution", "evaluation", "breathing"},
    DevelopmentalStage.CHILDHOOD: {
        "execution", "evaluation", "breathing", "archive", "penumbra",
    },
    DevelopmentalStage.ADOLESCENCE: {
        "execution", "evaluation", "breathing", "archive", "penumbra",
        "optimization",
    },
    DevelopmentalStage.ADULT: {
        "execution", "evaluation", "breathing", "archive", "penumbra",
        "optimization", "autonomy",
    },
}


@dataclass(frozen=True)
class DevelopmentalMetrics:
    """Snapshot of metrics used to evaluate stage transitions."""

    total_traces: int
    consecutive_traces_with_eval: int
    successful_dissolutions: int
    archive_size: int
    disagreement_correlation: float | None  # Spearman rho, None if not yet computed


class DevelopmentalController:
    """Controls progressive developmental transitions.

    Transitions are irreversible -- the system only moves forward.
    Each transition requires specific metric thresholds to be met.
    """

    def __init__(
        self,
        initial_stage: DevelopmentalStage = DevelopmentalStage.EMBRYONIC,
    ) -> None:
        self._stage = initial_stage
        self._transition_history: list[
            tuple[float, DevelopmentalStage, DevelopmentalStage]
        ] = []

    @property
    def current_stage(self) -> DevelopmentalStage:
        """Current developmental stage."""
        return self._stage

    def check_transition(
        self,
        metrics: DevelopmentalMetrics,
        timestamp: float,
    ) -> DevelopmentalStage | None:
        """Check if ready for next stage. Transitions are IRREVERSIBLE.

        Returns new stage if transition occurred, None otherwise.

        Transition conditions:
            EMBRYONIC -> FETAL: always allowed (contrapuntal eval is ready)
            FETAL -> BORN: consecutive_traces_with_eval >= 100
            BORN -> CHILDHOOD: successful_dissolutions >= 1
            CHILDHOOD -> ADOLESCENCE: archive_size >= 500
            ADOLESCENCE -> ADULT: disagreement_correlation is not None and >= 0.6
        """
        next_stage = self._next_stage()
        if next_stage is None:
            return None

        if not self._meets_threshold(next_stage, metrics):
            return None

        old_stage = self._stage
        self._stage = next_stage
        self._transition_history.append((timestamp, old_stage, next_stage))
        return next_stage

    def force_transition(
        self,
        stage: DevelopmentalStage,
        timestamp: float,
    ) -> None:
        """Force transition to a specific stage. For testing only."""
        old_stage = self._stage
        self._stage = stage
        self._transition_history.append((timestamp, old_stage, stage))

    def get_enabled_features(self) -> set[str]:
        """Which features are enabled at current stage."""
        return set(_FEATURES_BY_STAGE[self._stage])

    @property
    def transition_history(
        self,
    ) -> list[tuple[float, DevelopmentalStage, DevelopmentalStage]]:
        """History of all transitions as (timestamp, from_stage, to_stage)."""
        return list(self._transition_history)

    def _next_stage(self) -> DevelopmentalStage | None:
        """Return the next stage, or None if already at ADULT."""
        value = self._stage.value + 1
        try:
            return DevelopmentalStage(value)
        except ValueError:
            return None

    @staticmethod
    def _meets_threshold(
        target: DevelopmentalStage,
        metrics: DevelopmentalMetrics,
    ) -> bool:
        """Check if metrics meet the threshold for transitioning to target."""
        if target == DevelopmentalStage.FETAL:
            return True
        if target == DevelopmentalStage.BORN:
            return metrics.consecutive_traces_with_eval >= 100
        if target == DevelopmentalStage.CHILDHOOD:
            return metrics.successful_dissolutions >= 1
        if target == DevelopmentalStage.ADOLESCENCE:
            return metrics.archive_size >= 500
        if target == DevelopmentalStage.ADULT:
            return (
                metrics.disagreement_correlation is not None
                and metrics.disagreement_correlation >= 0.6
            )
        return False
