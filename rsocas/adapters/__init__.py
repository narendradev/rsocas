"""Framework adapters for RSOCAS — GEPA, DSPy, and Meta-Harness integration."""

from rsocas.adapters.dspy_leaf_registry import DSPyLeafRegistry, LeafModuleSpec
from rsocas.adapters.gepa_tree_adapter import TreeTraceGEPAAdapter
from rsocas.adapters.metaharness_bridge import CombinatorCandidate, MetaHarnessBridge

__all__ = [
    "CombinatorCandidate",
    "DSPyLeafRegistry",
    "LeafModuleSpec",
    "MetaHarnessBridge",
    "TreeTraceGEPAAdapter",
]
