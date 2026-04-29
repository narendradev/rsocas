"""ContinualLearningSystem -- top-level orchestrator.

Wires together all RSOCAS modules. Features are enabled progressively
based on the developmental stage. The orchestrator does NOT import
Lambda-RLM -- it receives TreeTrace objects from the caller.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from rsocas.contracts.evaluation import (
    DisagreementSignal,
    EvalResult,
    Evaluator,
)
from rsocas.contracts.traces import TreeTrace
from rsocas.development.stages import (
    DevelopmentalController,
    DevelopmentalMetrics,
    DevelopmentalStage,
)
from rsocas.evaluation.disagreement import compute_disagreement


@dataclass(frozen=True)
class RunResult:
    """Immutable result of one continual learning cycle."""

    output: str
    trace: TreeTrace | None
    evaluations: tuple[EvalResult, ...] | None
    disagreement: DisagreementSignal | None
    stage: DevelopmentalStage
    surfaced_for_human: bool


@dataclass(frozen=True)
class SystemStatus:
    """Immutable snapshot of the entire system state."""

    stage: DevelopmentalStage
    enabled_features: frozenset[str]
    total_runs: int
    archive_size: int
    active_combinators: int
    breathing_rate: float | None
    temperature: float | None


class ContinualLearningSystem:
    """The full RSOCAS continual learning system.

    Wires together all modules. Features are enabled progressively
    based on the developmental stage. All components are injected
    via the constructor -- the orchestrator creates nothing.
    """

    def __init__(
        self,
        evaluators: tuple[Evaluator, ...] = (),
        tempo: object | None = None,
        annealing: object | None = None,
        crystallizer: object | None = None,
        archive: object | None = None,
        interference: object | None = None,
        feedback_anchor: object | None = None,
        development: DevelopmentalController | None = None,
        disagreement_threshold: float = 0.3,
    ) -> None:
        self._evaluators = evaluators
        self._tempo = tempo
        self._annealing = annealing
        self._crystallizer = crystallizer
        self._archive = archive
        self._interference = interference
        self._feedback_anchor = feedback_anchor
        self._dev = development or DevelopmentalController()
        self._disagreement_threshold = disagreement_threshold
        self._total_runs = 0
        self._consecutive_eval_traces = 0
        self._successful_dissolutions = 0
        self._last_surface_time = 0.0

    def run(self, trace: TreeTrace) -> RunResult:
        """Execute one continual learning cycle on an already-computed trace.

        NOTE: This does NOT call Lambda-RLM. The caller provides the trace
        (from Lambda-RLM execution with tracing enabled). This keeps the
        orchestrator decoupled from Lambda-RLM.

        Steps based on developmental stage:
            1. ALWAYS: increment run counter
            2. FETAL+: evaluate with contrapuntal evaluators, compute disagreement
            3. BORN+: tick breathing cycle (tempo, annealing), check interference
            4. CHILDHOOD+: store in archive
            5. Check for developmental transition
        """
        self._total_runs += 1
        features = self._dev.get_enabled_features()

        evaluations: tuple[EvalResult, ...] | None = None
        disagreement: DisagreementSignal | None = None
        surfaced = False

        # Step 2: Evaluation (FETAL+)
        if "evaluation" in features and self._evaluators:
            eval_results = tuple(
                ev.evaluate(trace) for ev in self._evaluators
            )
            evaluations = eval_results
            disagreement = compute_disagreement(
                eval_results,
                threshold=self._disagreement_threshold,
                timestamp=trace.timestamp,
            )
            self._consecutive_eval_traces += 1

        # Step 3: Breathing cycle (BORN+)
        if "breathing" in features:
            surfaced = self._tick_breathing(disagreement, trace.timestamp)

        # Step 4: Archive (CHILDHOOD+)
        if "archive" in features and self._archive is not None:
            self._archive.store(
                trace,
                evaluations=evaluations or (),
                disagreement=disagreement,
            )

        # Step 5: Check developmental transition
        metrics = self._compute_metrics()
        self._dev.check_transition(metrics, trace.timestamp)

        return RunResult(
            output=trace.final_output,
            trace=trace,
            evaluations=evaluations,
            disagreement=disagreement,
            stage=self._dev.current_stage,
            surfaced_for_human=surfaced,
        )

    def receive_human_feedback(
        self,
        timestamp: float,
        feedback_type: str = "general",
    ) -> None:
        """Process human feedback event.

        Updates feedback anchor and tempo controller.
        If annealing active, trigger controlled reheat.
        """
        if self._feedback_anchor is not None:
            self._feedback_anchor.record(timestamp, feedback_type)

        if self._tempo is not None:
            self._tempo.record_human_feedback(timestamp)

        if self._annealing is not None:
            self._annealing.reheat()

    def status(self) -> SystemStatus:
        """Current system status snapshot."""
        breathing_rate: float | None = None
        if self._tempo is not None:
            breathing_rate = self._tempo.breathing_rate()

        temperature: float | None = None
        if self._annealing is not None:
            temperature = self._annealing.temperature

        archive_size = 0
        if self._archive is not None:
            archive_size = self._archive.count()

        active_combinators = 0
        if self._crystallizer is not None:
            active_combinators = len(
                self._crystallizer._db.list_by_status("crystallized")
            ) + len(
                self._crystallizer._db.list_by_status("fluid")
            )

        return SystemStatus(
            stage=self._dev.current_stage,
            enabled_features=frozenset(self._dev.get_enabled_features()),
            total_runs=self._total_runs,
            archive_size=archive_size,
            active_combinators=active_combinators,
            breathing_rate=breathing_rate,
            temperature=temperature,
        )

    def _compute_metrics(self) -> DevelopmentalMetrics:
        """Compute current developmental metrics from internal state."""
        archive_size = 0
        if self._archive is not None:
            archive_size = self._archive.count()

        return DevelopmentalMetrics(
            total_traces=self._total_runs,
            consecutive_traces_with_eval=self._consecutive_eval_traces,
            successful_dissolutions=self._successful_dissolutions,
            archive_size=archive_size,
            disagreement_correlation=None,
        )

    def _tick_breathing(
        self,
        disagreement: DisagreementSignal | None,
        timestamp: float,
    ) -> bool:
        """Tick the breathing cycle and check for surfacing.

        Returns True if the system should surface for human feedback.
        """
        if self._tempo is not None:
            self._tempo.record_system_event(timestamp, "run")

        if self._annealing is not None:
            self._annealing.cool()

        if (
            self._interference is not None
            and disagreement is not None
            and self._tempo is not None
        ):
            state = self._tempo.get_state()
            surfaced = self._interference.should_surface(
                disagreement=disagreement,
                human_freq=state.human_freq,
                system_freq=state.system_freq,
                last_surface_time=self._last_surface_time,
                now=timestamp,
            )
            if surfaced:
                self._last_surface_time = timestamp
            return surfaced

        return False
