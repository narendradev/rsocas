"""Tests for MlflowLogger with mocked mlflow."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, call

import pytest

from rsocas.contracts.evaluation import DisagreementSignal, EvalResult
from rsocas.contracts.traces import LeafTrace, NodeTrace, TreeTrace
from rsocas.development.orchestrator import RunResult, SystemStatus
from rsocas.development.stages import DevelopmentalStage

import rsocas.observability.mlflow_logger as mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace(final_score: float | None = 0.85) -> TreeTrace:
    node = NodeTrace(
        id="n0", depth=0, position=0, combinator="SPLIT",
        input_size=10, output="out",
    )
    leaf = LeafTrace(node_id="n0", prompt="p", response="r")
    return TreeTrace(
        trace_id="t1",
        task_type="qa",
        k=3,
        depth=1,
        tau=1,
        cost_estimate=0.02,
        nodes=(node,),
        leaf_traces=(leaf,),
        final_output="answer",
        timestamp=time.time(),
        execution_time_ms=150.0,
        total_llm_calls=5,
        total_tokens=1200,
        final_score=final_score,
    )


def _make_result(
    evaluations: tuple[EvalResult, ...] | None = None,
    disagreement: DisagreementSignal | None = None,
    surfaced: bool = False,
    stage: DevelopmentalStage = DevelopmentalStage.FETAL,
) -> RunResult:
    return RunResult(
        output="answer",
        trace=None,
        evaluations=evaluations,
        disagreement=disagreement,
        stage=stage,
        surfaced_for_human=surfaced,
    )


def _make_status(
    breathing_rate: float | None = None,
    temperature: float | None = None,
    archive_size: int = 42,
) -> SystemStatus:
    return SystemStatus(
        stage=DevelopmentalStage.FETAL,
        enabled_features=frozenset({"execution", "evaluation"}),
        total_runs=10,
        archive_size=archive_size,
        active_combinators=3,
        breathing_rate=breathing_rate,
        temperature=temperature,
    )


@pytest.fixture()
def mock_mlflow(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a mock mlflow module and enable the logger."""
    fake = MagicMock()
    fake.start_run.return_value.__enter__ = MagicMock()
    fake.start_run.return_value.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(mod, "mlflow", fake)
    monkeypatch.setattr(mod, "_MLFLOW_AVAILABLE", True)
    return fake


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_disabled_when_mlflow_not_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "_MLFLOW_AVAILABLE", False)
    logger = mod.MlflowLogger()
    # on_run should be a no-op — no crash, no mlflow calls
    logger.on_run(_make_trace(), _make_result(), _make_status())


def test_params_logged_correctly(mock_mlflow: MagicMock) -> None:
    logger = mod.MlflowLogger()
    trace = _make_trace()
    result = _make_result(stage=DevelopmentalStage.BORN)
    logger.on_run(trace, result, _make_status())

    params = mock_mlflow.log_params.call_args[0][0]
    assert params["task_type"] == "qa"
    assert params["k"] == 3
    assert params["depth"] == 1
    assert params["tau"] == 1
    assert params["stage"] == "BORN"


def test_metrics_logged_correctly(mock_mlflow: MagicMock) -> None:
    logger = mod.MlflowLogger()
    trace = _make_trace(final_score=0.9)
    logger.on_run(trace, _make_result(), _make_status(archive_size=100))

    metrics = mock_mlflow.log_metrics.call_args[0][0]
    assert metrics["execution_time_ms"] == 150.0
    assert metrics["total_llm_calls"] == 5
    assert metrics["total_tokens"] == 1200
    assert metrics["cost_estimate"] == 0.02
    assert metrics["final_score"] == 0.9
    assert metrics["archive_size"] == 100


def test_evaluator_scores_logged(mock_mlflow: MagicMock) -> None:
    evals = (
        EvalResult(score=0.8, confidence=0.9, signal_type="correctness"),
        EvalResult(score=0.7, confidence=0.85, signal_type="coherence"),
    )
    result = _make_result(evaluations=evals)
    logger = mod.MlflowLogger()
    logger.on_run(_make_trace(), result, _make_status())

    metrics = mock_mlflow.log_metrics.call_args[0][0]
    assert metrics["eval_correctness_score"] == 0.8
    assert metrics["eval_coherence_score"] == 0.7


def test_disagreement_metrics_logged(mock_mlflow: MagicMock) -> None:
    disagreement = DisagreementSignal(magnitude=0.45)
    result = _make_result(disagreement=disagreement, surfaced=True)
    logger = mod.MlflowLogger()
    logger.on_run(_make_trace(), result, _make_status())

    metrics = mock_mlflow.log_metrics.call_args[0][0]
    assert metrics["disagreement_magnitude"] == 0.45
    assert metrics["surfaced"] == 1.0


def test_null_evaluations_handled(mock_mlflow: MagicMock) -> None:
    result = _make_result(evaluations=None)
    logger = mod.MlflowLogger()
    logger.on_run(_make_trace(), result, _make_status())

    metrics = mock_mlflow.log_metrics.call_args[0][0]
    assert not any(k.startswith("eval_") for k in metrics)


def test_null_final_score_handled(mock_mlflow: MagicMock) -> None:
    trace = _make_trace(final_score=None)
    logger = mod.MlflowLogger()
    logger.on_run(trace, _make_result(), _make_status())

    metrics = mock_mlflow.log_metrics.call_args[0][0]
    assert "final_score" not in metrics


def test_experiment_name_matches_stage(mock_mlflow: MagicMock) -> None:
    result = _make_result(stage=DevelopmentalStage.CHILDHOOD)
    logger = mod.MlflowLogger()
    logger.on_run(_make_trace(), result, _make_status())

    mock_mlflow.set_experiment.assert_called_once_with("rsocas/CHILDHOOD")


def test_artifact_logged(mock_mlflow: MagicMock) -> None:
    logger = mod.MlflowLogger()
    logger.on_run(_make_trace(), _make_result(), _make_status())

    mock_mlflow.log_artifact.assert_called_once()
    call_args = mock_mlflow.log_artifact.call_args
    assert call_args[0][1] == "traces"
    assert call_args[0][0].endswith(".json")


def test_on_run_never_crashes(mock_mlflow: MagicMock) -> None:
    mock_mlflow.set_experiment.side_effect = RuntimeError("boom")
    logger = mod.MlflowLogger()
    # Must not raise
    logger.on_run(_make_trace(), _make_result(), _make_status())


def test_shutdown_is_noop() -> None:
    logger = mod.MlflowLogger.__new__(mod.MlflowLogger)
    logger._enabled = False
    logger.shutdown()  # must not crash


def test_breathing_rate_and_temperature_logged(
    mock_mlflow: MagicMock,
) -> None:
    logger = mod.MlflowLogger()
    status = _make_status(breathing_rate=0.5, temperature=0.8)
    logger.on_run(_make_trace(), _make_result(), status)

    metrics = mock_mlflow.log_metrics.call_args[0][0]
    assert metrics["breathing_rate"] == 0.5
    assert metrics["temperature"] == 0.8
