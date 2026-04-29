"""Contract 2: Evaluator Protocol and DisagreementSignal.

Three evaluators produce EvalResults. compute_disagreement() is a pure function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rsocas.contracts.traces import TreeTrace


@dataclass(frozen=True)
class EvalResult:
    score: float
    confidence: float
    signal_type: str
    per_node_scores: dict[str, float] = field(default_factory=dict)
    explanation: str = ""


@dataclass(frozen=True)
class DisagreementSignal:
    magnitude: float
    pairwise: dict[str, float] = field(default_factory=dict)
    per_node: dict[str, float] = field(default_factory=dict)
    outlier_voice: str | None = None
    should_surface: bool = False
    timestamp: float = 0.0


class Evaluator(Protocol):
    @property
    def signal_type(self) -> str: ...

    def evaluate(self, trace: TreeTrace, ground_truth: str | None = None) -> EvalResult: ...
