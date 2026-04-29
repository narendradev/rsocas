"""MLflow logger for RSOCAS — logs each run() as an MLflow experiment run.

All mlflow calls are wrapped in try/except so tracking failures never
crash the system. The mlflow dependency is optional; when absent the
logger silently disables itself.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rsocas.contracts.traces import TreeTrace
    from rsocas.development.orchestrator import RunResult, SystemStatus

mlflow: object | None = None
_MLFLOW_AVAILABLE = False
try:
    import mlflow  # type: ignore[no-redef]

    _MLFLOW_AVAILABLE = True
except ImportError:
    pass


class MlflowLogger:
    """ObservabilityExporter that ships metrics/params/artifacts to MLflow."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_prefix: str = "rsocas",
    ) -> None:
        self._enabled = False
        if not _MLFLOW_AVAILABLE:
            return
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self._prefix = experiment_prefix
        self._enabled = True

    def on_run(
        self,
        trace: TreeTrace,
        result: RunResult,
        status: SystemStatus,
    ) -> None:
        if not self._enabled:
            return
        try:
            self._log(trace, result, status)
        except Exception:
            pass

    def shutdown(self) -> None:
        pass  # mlflow doesn't need explicit shutdown

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log(
        self,
        trace: TreeTrace,
        result: RunResult,
        status: SystemStatus,
    ) -> None:
        experiment_name = f"{self._prefix}/{result.stage.name}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            mlflow.log_params(
                {
                    "task_type": trace.task_type,
                    "k": trace.k,
                    "depth": trace.depth,
                    "tau": trace.tau,
                    "stage": result.stage.name,
                }
            )

            metrics: dict[str, float] = {
                "execution_time_ms": trace.execution_time_ms,
                "total_llm_calls": trace.total_llm_calls,
                "total_tokens": trace.total_tokens,
                "cost_estimate": trace.cost_estimate,
                "archive_size": status.archive_size,
            }

            if trace.final_score is not None:
                metrics["final_score"] = trace.final_score

            if result.evaluations:
                for ev in result.evaluations:
                    metrics[f"eval_{ev.signal_type}_score"] = ev.score

            if result.disagreement is not None:
                metrics["disagreement_magnitude"] = result.disagreement.magnitude
                metrics["surfaced"] = 1.0 if result.surfaced_for_human else 0.0

            if status.breathing_rate is not None:
                metrics["breathing_rate"] = status.breathing_rate

            if status.temperature is not None:
                metrics["temperature"] = status.temperature

            mlflow.log_metrics(metrics)

            trace_dict = dataclasses.asdict(trace)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(trace_dict, f, default=str)
                tmp_path = f.name
            mlflow.log_artifact(tmp_path, "traces")
            os.unlink(tmp_path)
