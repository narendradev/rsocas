"""Contrapuntal evaluation module: three evaluators + disagreement metric."""

from rsocas.evaluation.boundary_detection import BoundaryDetectionEval
from rsocas.evaluation.disagreement import compute_disagreement
from rsocas.evaluation.goodhart_resistant import GoodhartResistantEval
from rsocas.evaluation.info_theoretic import InformationTheoreticEval

__all__ = [
    "BoundaryDetectionEval",
    "GoodhartResistantEval",
    "InformationTheoreticEval",
    "compute_disagreement",
]
