"""Breathing cycle module — tempo control, annealing, and feedback anchoring.

The breathing metaphor: the system oscillates between crystallization
(systole) and dissolution (diastole), with tempo controlled by human
feedback frequency.
"""

from rsocas.breathing.annealing import AnnealingSchedule, AnnealingState
from rsocas.breathing.feedback_anchor import FeedbackAnchor
from rsocas.breathing.interference import InterferencePattern
from rsocas.breathing.tempo import PIDTempoController, TempoState

__all__ = [
    "AnnealingSchedule",
    "AnnealingState",
    "FeedbackAnchor",
    "InterferencePattern",
    "PIDTempoController",
    "TempoState",
]
