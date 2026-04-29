"""Combinator lifecycle: versioning, penumbra, crystallization."""

from rsocas.combinators.crystallizer import Crystallizer
from rsocas.combinators.lambda_rlm_integration import patch_lambda_rlm_full
from rsocas.combinators.penumbra import PenumbraStore
from rsocas.combinators.registry import CombinatorRegistry
from rsocas.combinators.versioned import CombinatorDB

__all__ = [
    "CombinatorDB",
    "CombinatorRegistry",
    "Crystallizer",
    "PenumbraStore",
    "patch_lambda_rlm_full",
]
