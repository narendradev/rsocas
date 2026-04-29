"""Contract 1: TreeTrace — the universal data bus.

Every component reads or writes TreeTrace. This contract is frozen.
Changes require explicit team approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NodeTrace:
    id: str
    depth: int
    position: int
    combinator: str
    input_size: int
    output: str
    children: tuple[str, ...] = ()
    llm_calls: int = 0
    latency_ms: float = 0.0
    score: float | None = None


@dataclass(frozen=True)
class LeafTrace:
    node_id: str
    prompt: str
    response: str
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""
    confidence: float | None = None


@dataclass(frozen=True)
class TreeTrace:
    trace_id: str
    task_type: str
    k: int
    depth: int
    tau: int
    cost_estimate: float
    nodes: tuple[NodeTrace, ...]
    leaf_traces: tuple[LeafTrace, ...]
    final_output: str
    timestamp: float
    execution_time_ms: float
    total_llm_calls: int = 0
    total_tokens: int = 0
    final_score: float | None = None
    combinator_versions: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
