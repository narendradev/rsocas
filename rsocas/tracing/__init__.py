"""Tracing module — patches Lambda-RLM to emit TreeTrace objects."""

from rsocas.tracing.collector import CallEvent, TraceCollector
from rsocas.tracing.builder import TreeTraceBuilder
from rsocas.tracing.patch import patch_for_tracing

__all__ = [
    "CallEvent",
    "TraceCollector",
    "TreeTraceBuilder",
    "patch_for_tracing",
]
