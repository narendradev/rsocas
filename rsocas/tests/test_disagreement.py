"""Tests for compute_disagreement function."""

from __future__ import annotations

import pytest

from rsocas.contracts.evaluation import DisagreementSignal, EvalResult
from rsocas.evaluation.disagreement import compute_disagreement


def _make_result(
    score: float,
    signal_type: str,
    per_node_scores: dict[str, float] | None = None,
) -> EvalResult:
    """Helper to create an EvalResult for testing."""
    return EvalResult(
        score=score,
        confidence=0.9,
        signal_type=signal_type,
        per_node_scores=per_node_scores or {},
        explanation=f"Test result for {signal_type}",
    )


class TestComputeDisagreement:
    """Tests for the compute_disagreement pure function."""

    def test_three_agreeing_evaluators(self) -> None:
        """Three evaluators with same score -> magnitude ~ 0, should_surface=False."""
        results = (
            _make_result(0.9, "info"),
            _make_result(0.9, "boundary"),
            _make_result(0.9, "goodhart"),
        )
        signal = compute_disagreement(results, threshold=0.3)

        assert signal.magnitude == pytest.approx(0.0, abs=0.01)
        assert signal.should_surface is False
        # All pairwise differences should be ~0
        for key, diff in signal.pairwise.items():
            assert diff == pytest.approx(0.0, abs=0.01)

    def test_one_outlier(self) -> None:
        """One outlier (0.9, 0.9, 0.1) -> magnitude ~ 0.8, outlier identified."""
        results = (
            _make_result(0.9, "info"),
            _make_result(0.9, "boundary"),
            _make_result(0.1, "goodhart"),
        )
        signal = compute_disagreement(results, threshold=0.3)

        assert signal.magnitude == pytest.approx(0.8, abs=0.01)
        assert signal.should_surface is True
        assert signal.outlier_voice == "goodhart"
        # Check pairwise
        assert "info_vs_goodhart" in signal.pairwise
        assert signal.pairwise["info_vs_goodhart"] == pytest.approx(0.8, abs=0.01)

    def test_balanced_disagreement(self) -> None:
        """Balanced disagreement (0.3, 0.5, 0.7) -> magnitude ~ 0.4."""
        results = (
            _make_result(0.3, "info"),
            _make_result(0.5, "boundary"),
            _make_result(0.7, "goodhart"),
        )
        signal = compute_disagreement(results, threshold=0.3)

        assert signal.magnitude == pytest.approx(0.4, abs=0.01)
        assert signal.should_surface is True
        # Outlier should be one of the extremes (0.3 or 0.7 -- both deviate 0.2 from mean 0.5)
        assert signal.outlier_voice in ("info", "goodhart")

    def test_per_node_disagreement(self) -> None:
        """Per-node disagreement computed as variance across evaluators."""
        results = (
            _make_result(
                0.8, "info",
                per_node_scores={"node_a": 0.9, "node_b": 0.8},
            ),
            _make_result(
                0.7, "boundary",
                per_node_scores={"node_a": 0.1, "node_b": 0.8},
            ),
            _make_result(
                0.6, "goodhart",
                per_node_scores={"node_a": 0.5, "node_b": 0.8},
            ),
        )
        signal = compute_disagreement(results, threshold=0.3)

        # node_a: scores [0.9, 0.1, 0.5], mean=0.5, variance = ((0.16+0.16+0.0)/3) ~ 0.1067
        assert "node_a" in signal.per_node
        node_a_scores = [0.9, 0.1, 0.5]
        node_a_mean = sum(node_a_scores) / 3
        node_a_var = sum((s - node_a_mean) ** 2 for s in node_a_scores) / 3
        assert signal.per_node["node_a"] == pytest.approx(node_a_var, abs=0.001)

        # node_b: scores [0.8, 0.8, 0.8], mean=0.8, variance=0.0
        assert "node_b" in signal.per_node
        assert signal.per_node["node_b"] == pytest.approx(0.0, abs=0.001)

    def test_empty_results(self) -> None:
        """Empty results should return neutral signal."""
        signal = compute_disagreement((), threshold=0.3)

        assert signal.magnitude == 0.0
        assert signal.should_surface is False
        assert signal.outlier_voice is None

    def test_single_result(self) -> None:
        """Single result should return zero disagreement."""
        results = (_make_result(0.5, "info"),)
        signal = compute_disagreement(results, threshold=0.3)

        assert signal.magnitude == 0.0
        assert signal.should_surface is False

    def test_threshold_boundary_below(self) -> None:
        """Magnitude below threshold should NOT surface."""
        results = (
            _make_result(0.5, "info"),
            _make_result(0.7, "boundary"),
        )
        # magnitude = 0.2, threshold = 0.3 -> NOT surfaced
        signal = compute_disagreement(results, threshold=0.3)

        assert signal.magnitude == pytest.approx(0.2, abs=0.01)
        assert signal.should_surface is False

    def test_threshold_boundary_at(self) -> None:
        """Magnitude at threshold should surface (>=)."""
        results = (
            _make_result(0.5, "info"),
            _make_result(0.8, "boundary"),
        )
        # magnitude = 0.3, threshold = 0.3 -> surfaced (>=)
        signal = compute_disagreement(results, threshold=0.3)

        assert signal.magnitude == pytest.approx(0.3, abs=0.01)
        assert signal.should_surface is True

    def test_timestamp_propagated(self) -> None:
        """Timestamp should be propagated to the signal."""
        results = (
            _make_result(0.5, "info"),
            _make_result(0.8, "boundary"),
        )
        signal = compute_disagreement(results, threshold=0.3, timestamp=42.0)

        assert signal.timestamp == 42.0

    def test_deterministic(self) -> None:
        """Same inputs should produce same outputs (deterministic)."""
        results = (
            _make_result(0.3, "info", {"n1": 0.5}),
            _make_result(0.7, "boundary", {"n1": 0.9}),
            _make_result(0.5, "goodhart", {"n1": 0.1}),
        )
        signal_a = compute_disagreement(results, threshold=0.3, timestamp=1.0)
        signal_b = compute_disagreement(results, threshold=0.3, timestamp=1.0)

        assert signal_a.magnitude == signal_b.magnitude
        assert signal_a.pairwise == signal_b.pairwise
        assert signal_a.per_node == signal_b.per_node
        assert signal_a.outlier_voice == signal_b.outlier_voice
        assert signal_a.should_surface == signal_b.should_surface
