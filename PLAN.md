# RSOCAS Implementation Plan v2 — Executable, Modular, Agent-Team Ready

**Date**: 2026-04-28
**Status**: AWAITING CONFIRMATION

---

## Pre-Agreed Contracts

Every module boundary is defined by a Protocol or frozen dataclass. Agent teams implement against interfaces, not against each other. Changes to contracts require explicit approval.

---

## Contract 1: TreeTrace (the universal data bus)

Every component reads or writes TreeTrace. This is agreed first, changed never.

```python
# rsocas/contracts/traces.py

from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class NodeTrace:
    id: str
    depth: int
    position: int                    # position among siblings (0-indexed)
    combinator: str                  # "SPLIT" | "MAP" | "REDUCE" | "FILTER" | "CROSS" | "LEAF"
    input_size: int                  # chars
    output: str
    children: tuple[str, ...]        # child node IDs (empty for leaves)
    llm_calls: int
    latency_ms: float
    score: float | None = None       # populated by evaluators, not by execution

@dataclass(frozen=True)
class LeafTrace:
    node_id: str
    prompt: str
    response: str
    tokens_in: int
    tokens_out: int
    model: str
    confidence: float | None = None

@dataclass(frozen=True)
class TreeTrace:
    trace_id: str                    # UUID
    task_type: str
    k: int
    depth: int
    tau: int
    cost_estimate: float
    nodes: tuple[NodeTrace, ...]
    leaf_traces: tuple[LeafTrace, ...]
    final_output: str
    final_score: float | None        # populated after evaluation
    timestamp: float
    execution_time_ms: float
    total_llm_calls: int
    total_tokens: int
    combinator_versions: dict[str, str]  # name -> version_id (empty until Phase 1)
    metadata: dict                   # extensible
```

---

## Contract 2: Evaluator Protocol

```python
# rsocas/contracts/evaluation.py

@dataclass(frozen=True)
class EvalResult:
    score: float                     # 0.0 to 1.0
    confidence: float                # 0.0 to 1.0
    signal_type: str                 # "information_theoretic" | "boundary" | "goodhart_resistant"
    per_node_scores: dict[str, float]  # node_id -> score
    explanation: str                 # short text for debugging

class Evaluator(Protocol):
    signal_type: str
    def evaluate(self, trace: TreeTrace, ground_truth: str | None = None) -> EvalResult: ...

@dataclass(frozen=True)
class DisagreementSignal:
    magnitude: float                 # 0.0 to 1.0 (max pairwise disagreement)
    pairwise: dict[str, float]       # "info_vs_boundary" -> delta, etc.
    per_node: dict[str, float]       # node_id -> max variance across evaluators
    outlier_voice: str | None        # which evaluator is the outlier, or None
    should_surface: bool             # True if magnitude > threshold
    timestamp: float

def compute_disagreement(results: tuple[EvalResult, ...], threshold: float = 0.3) -> DisagreementSignal:
    """Pure function. No side effects. Deterministic."""
    ...
```

---

## Contract 3: GEPA Adapter Interface

Based on actual GEPA API — uses `GEPAAdapter` protocol with `evaluate()` and `make_reflective_dataset()`.

```python
# rsocas/contracts/gepa_adapter.py

from gepa.core.adapter import GEPAAdapter, EvaluationBatch

class TreeTraceGEPAAdapter(GEPAAdapter):
    """Adapts Lambda-RLM tree traces into GEPA's reflective dataset format."""

    def evaluate(
        self,
        batch: list,
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Run Lambda-RLM with candidate prompts, return EvaluationBatch."""
        ...

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, str]]]:
        """Convert tree traces to GEPA reflective dataset format.

        Key innovation: per-node surgical context instead of flat trace.
        Each entry targets a SPECIFIC failing node:
        {
            "leaf_qa_prompt": [
                {
                    "Inputs": {"context": "chunk text...", "node_position": "1.2 at depth 2"},
                    "Generated Outputs": {"answer": "wrong answer"},
                    "Feedback": "Node 1.2 received chunk 2/8. Siblings produced [X,Y,Z].
                                 The answer was in this chunk at paragraph 3.
                                 The prompt missed the temporal reference.",
                },
            ]
        }
        """
        ...
```

---

## Contract 4: DSPy Leaf Module Interface

Based on actual DSPy API — extends `dspy.Module`, implements `forward()`.

```python
# rsocas/contracts/dspy_leaf.py

import dspy

class LeafModule(dspy.Module):
    """Wraps a Lambda-RLM leaf node as a DSPy Module for optimization."""

    def __init__(self, task_type: str, template: str, lm: dspy.LM | None = None):
        super().__init__()
        self.predict = dspy.ChainOfThought(f"context, question -> answer")
        if lm:
            self.predict.lm = lm  # per-predictor model override

    def forward(self, context: str, question: str = "") -> dspy.Prediction:
        return self.predict(context=context, question=question)

class LeafRegistry:
    """Maps task_type -> LeafModule. Provides the callable for Lambda-RLM REPL injection."""

    def get_leaf_fn(self, task_type: str) -> callable:
        """Returns a plain function(prompt: str) -> str for REPL injection."""
        module = self.modules[task_type]
        def leaf_fn(prompt: str) -> str:
            return module(context=prompt).answer
        return leaf_fn

    def optimize(self, task_type: str, optimizer, trainset, metric) -> None:
        """Optimize a leaf module. Optimizer can be GEPA, MIPRO, BootstrapFewShot, etc."""
        self.modules[task_type] = optimizer.compile(
            self.modules[task_type], trainset=trainset
        )
```

---

## Contract 5: Combinator Lifecycle

```python
# rsocas/contracts/combinators.py

@dataclass(frozen=True)
class ValidationSnapshot:
    task_types: tuple[str, ...]
    input_size_range: tuple[int, int]
    n_samples: int
    mean_score: float
    score_std: float
    timestamp: float

@dataclass(frozen=True)
class RepairRecord:
    timestamp: float
    trigger: str                     # "distribution_shift" | "disagreement" | "human" | "expiry"
    from_version: str
    change_summary: str
    score_delta: float

@dataclass(frozen=True)
class VersionedCombinator:
    name: str
    version_id: str                  # UUID
    code_hash: str
    status: str                      # "fluid" | "crystallized" | "dissolving" | "expired"
    created_at: float
    expires_at: float
    validation: ValidationSnapshot
    repairs: tuple[RepairRecord, ...]
    cost_constant: float
    type_signature: str

class CombinatorStore(Protocol):
    def crystallize(self, name: str, fn: callable, validation: ValidationSnapshot) -> VersionedCombinator: ...
    def dissolve(self, version_id: str, reason: str) -> VersionedCombinator: ...
    def get_active(self, name: str) -> VersionedCombinator | None: ...
    def get_penumbra(self, name: str, limit: int = 5) -> list[VersionedCombinator]: ...
    def check_staleness(self, version_id: str, current: ValidationSnapshot) -> float: ...

class TempoController(Protocol):
    def record_human_feedback(self, timestamp: float) -> None: ...
    def record_system_event(self, timestamp: float, event: str) -> None: ...
    def breathing_rate(self) -> float: ...
    def should_crystallize(self) -> bool: ...
    def should_dissolve(self) -> bool: ...
```

---

## Contract 6: Meta-Harness Integration

Based on actual Meta-Harness API — candidates implement `MemorySystem` abstract class.

```python
# rsocas/contracts/metaharness.py

from abc import ABC, abstractmethod

class CombinatorCandidate(ABC):
    """Implements Meta-Harness's MemorySystem interface for combinator discovery."""

    @abstractmethod
    def predict(self, input_data: dict) -> tuple[str, dict]:
        """Process input, return (answer, metadata)."""
        ...

    @abstractmethod
    def learn_from_batch(self, results: list[dict]) -> None:
        """Update internal state from evaluation results."""
        ...

def validate_combinator_candidate(candidate_path: str, type_checker) -> bool:
    """Extended validation: import check + Lambda-RLM type check.
    This is the Verified Innovation Pipeline from the dream."""
    # Step 1: Standard Meta-Harness validation (import check)
    # Step 2: Lambda-RLM type signature validation
    # Step 3: Termination proof (bounded loops, no recursion without depth limit)
    # Step 4: Cost constant computation
    ...
```

---

## Module Map (Agent Team Assignments)

```
rsocas/
├── contracts/                  # SHARED — frozen, change requires team approval
│   ├── __init__.py
│   ├── traces.py               # Contract 1: TreeTrace
│   ├── evaluation.py           # Contract 2: Evaluator Protocol
│   ├── gepa_adapter.py         # Contract 3: GEPA Adapter
│   ├── dspy_leaf.py            # Contract 4: DSPy Leaf Module
│   ├── combinators.py          # Contract 5: Combinator Lifecycle
│   └── metaharness.py          # Contract 6: Meta-Harness Integration
│
├── tracing/                    # AGENT A: Trace instrumentation
│   ├── __init__.py
│   ├── collector.py            # TraceCollector — wraps LocalREPL._llm_query
│   ├── builder.py              # TreeTraceBuilder — constructs TreeTrace from raw events
│   └── lambda_rlm_patch.py     # Monkey-patch LambdaRLM for trace emission
│
├── evaluation/                 # AGENT B: Contrapuntal evaluation
│   ├── __init__.py
│   ├── info_theoretic.py       # Evaluator 1: compression ratio per node
│   ├── boundary_detection.py   # Evaluator 2: self-referential loop detection
│   ├── goodhart_resistant.py   # Evaluator 3: perturbation robustness
│   ├── disagreement.py         # compute_disagreement() pure function
│   └── benchmark.py            # Correlation benchmark (THE load-bearing test)
│
├── combinators/                # AGENT C: Combinator lifecycle
│   ├── __init__.py
│   ├── versioned.py            # VersionedCombinator data + SQLite store
│   ├── penumbra.py             # PenumbraStore — cold storage variants
│   ├── crystallizer.py         # Crystallizer — lifecycle state machine
│   └── registry.py             # CombinatorRegistry — wraps Lambda-RLM _register_library
│
├── breathing/                  # AGENT D: Tempo and annealing
│   ├── __init__.py
│   ├── tempo.py                # TempoController — PID anchored to human feedback
│   ├── annealing.py            # AnnealingSchedule — cooling with phase boundary detection
│   ├── feedback_anchor.py      # FeedbackAnchor — tracks human feedback frequency
│   └── interference.py         # InterferencePattern — standing wave, surfacing logic
│
├── adapters/                   # AGENT E: Framework connectors
│   ├── __init__.py
│   ├── gepa_tree_adapter.py    # TreeTraceGEPAAdapter (Contract 3 impl)
│   ├── dspy_leaf_registry.py   # LeafRegistry + REPL injection (Contract 4 impl)
│   └── metaharness_bridge.py   # Meta-Harness <-> Lambda-RLM bridge (Contract 6 impl)
│
├── archive/                    # AGENT F: Persistent storage
│   ├── __init__.py
│   ├── trace_archive.py        # SQLite + FTS5 trace storage
│   ├── repair_index.py         # Growth plate / kintsugi episode indexing
│   └── distribution_tracker.py # Running distribution statistics for staleness
│
├── development/                # AGENT G: Developmental stages + orchestrator
│   ├── __init__.py
│   ├── stages.py               # DevelopmentalStage enum + transition logic
│   └── orchestrator.py         # ContinualLearningSystem — top-level wiring
│
├── tests/                      # Each agent owns their module's tests
│   ├── test_traces.py          # Agent A
│   ├── test_evaluators.py      # Agent B
│   ├── test_disagreement.py    # Agent B
│   ├── test_correlation.py     # Agent B (integration)
│   ├── test_combinators.py     # Agent C
│   ├── test_penumbra.py        # Agent C
│   ├── test_crystallizer.py    # Agent C
│   ├── test_tempo.py           # Agent D
│   ├── test_annealing.py       # Agent D
│   ├── test_gepa_adapter.py    # Agent E
│   ├── test_dspy_leaf.py       # Agent E
│   ├── test_archive.py         # Agent F
│   └── test_orchestrator.py    # Agent G (integration)
│
└── pyproject.toml
```

---

## Agent Team Assignments

| Agent | Module | Dependencies | Can Start After |
|-------|--------|-------------|-----------------|
| **A** (Trace Instrumentation) | `tracing/` | Lambda-RLM internals, Contract 1 | Contracts agreed |
| **B** (Contrapuntal Evaluation) | `evaluation/` | Contract 1, Contract 2 | Contracts agreed |
| **C** (Combinator Lifecycle) | `combinators/` | Contract 5 | Contracts agreed |
| **D** (Breathing/Tempo) | `breathing/` | Contract 5 (TempoController) | Contracts agreed |
| **E** (Framework Adapters) | `adapters/` | Contracts 3,4,6 + GEPA/DSPy APIs | Phase 0 validated |
| **F** (Archive) | `archive/` | Contract 1, Contract 2 | Contracts agreed |
| **G** (Orchestrator) | `development/` | ALL contracts | Phase 0 validated |

**Parallelism**: Agents A, B, C, D, F can ALL start simultaneously after contracts are agreed. They depend only on contracts, not on each other. Agent E starts after Phase 0 validates. Agent G starts last.

---

## Phase 0: The Existential Test

**Goal**: Answer the Critic's question — does disagreement correlate with failure?

### Step 0.0: Create project scaffold + install deps

```bash
cd ~/projects/rsocas
# pyproject.toml with deps: lambda-rlm (local), gepa, dspy, numpy, scipy, sqlite3
# Create all __init__.py files
# Create contracts/ with all Protocol definitions (frozen — no changes after this)
```

**Owner**: Any agent. **Duration**: 30 min. **Risk**: None.

### Step 0.1: TreeTrace emission (Agent A)

**What**: Patch `LocalREPL._llm_query` and `_llm_query_batched` to emit trace events. Build `TreeTraceBuilder` that reconstructs the tree from flat events.

**Critical detail from codebase analysis**: The `_Reduce` and `_FilterRelevant` closures capture `repl._llm_query` directly at `lambda_rlm.py:466`, NOT via `repl.globals["llm_query"]`. Therefore:

```python
# rsocas/tracing/lambda_rlm_patch.py

def patch_lambda_rlm_for_tracing(lrlm: LambdaRLM) -> LambdaRLM:
    """Monkey-patch a LambdaRLM instance to emit TreeTrace.

    Strategy: Wrap the _register_library method so that when it creates
    the _llm closure (line 466), it wraps repl._llm_query BEFORE the
    closure captures it. This ensures ALL paths (leaf, reduce, filter)
    are traced.
    """
    original_register = lrlm._register_library

    def traced_register(repl, plan, query=""):
        # Wrap repl._llm_query BEFORE _register_library captures it
        collector = TraceCollector()
        original_llm_query = repl._llm_query
        original_llm_query_batched = repl._llm_query_batched

        def traced_llm_query(prompt, model=None):
            call_id = collector.start_call(prompt, model)
            response = original_llm_query(prompt, model)
            collector.end_call(call_id, response)
            return response

        def traced_llm_query_batched(prompts, model=None):
            call_ids = [collector.start_call(p, model) for p in prompts]
            responses = original_llm_query_batched(prompts, model)
            for cid, resp in zip(call_ids, responses):
                collector.end_call(cid, resp)
            return responses

        # Replace BEFORE _register_library captures _llm
        repl._llm_query = traced_llm_query
        repl._llm_query_batched = traced_llm_query_batched
        # Also update the globals reference
        repl.globals["llm_query"] = traced_llm_query
        repl.globals["llm_query_batched"] = traced_llm_query_batched

        original_register(repl, plan, query)

        # Attach collector to repl for later retrieval
        repl._trace_collector = collector

    lrlm._register_library = traced_register
    return lrlm
```

**Test**: Run Lambda-RLM on one SNIAH sample with tracing. Verify TreeTrace has correct tree structure (k nodes at depth 1, k^2 at depth 2, etc.). Verify all leaf calls and reduce calls are captured.

**Duration**: 1 day. **Risk**: Medium (REPL internals).

### Step 0.2: Three evaluators (Agent B)

**What**: Implement three evaluators against Contract 2. Each must be independently testable with synthetic TreeTraces.

**Evaluator 1: InformationTheoreticEval** (`evaluation/info_theoretic.py`)
- For each REDUCE node: `info_ratio = compressed_size(output) / sum(compressed_size(input_i))`
- Use `zlib.compress()` — zero LLM calls, pure computation
- Score: mean info_ratio across all REDUCE nodes (higher = more info preserved)
- Per-node: each node gets its own info_ratio

**Evaluator 2: BoundaryDetectionEval** (`evaluation/boundary_detection.py`)
- For each REDUCE node: compute cosine similarity between output and each input
- If max_similarity > 0.95: the system is echoing, not synthesizing (boundary violation)
- Use `sentence-transformers` with `all-MiniLM-L6-v2` (local, ~10ms per embedding)
- Fallback: Jaccard similarity on word sets (if no GPU/embedding model)
- Score: 1.0 - max_echo_ratio across nodes
- Per-node: each node gets its echo score

**Evaluator 3: GoodhartResistantEval** (`evaluation/goodhart_resistant.py`)
- Perturb the input: insert one irrelevant sentence into a random leaf's chunk
- Re-run that leaf call only (not the whole tree)
- Compare original output with perturbed output
- If output changes significantly on irrelevant perturbation: fragile (Goodhart'd)
- Score: stability ratio (fraction of perturbations that don't change the answer)
- Per-node: only leaf nodes get scores
- Cost: 1-3 additional LLM calls. **Optional** — skip if no LLM client available

**Test**: Each evaluator tested with:
- A "perfect" synthetic trace (high info preservation, no echoing, stable)
- A "failing" synthetic trace (info loss, echoing, fragile)
- Edge cases (empty output, single node tree, very deep tree)

**Duration**: 2 days. **Risk**: Medium (embedding model dependency — mitigated by Jaccard fallback).

### Step 0.3: Disagreement metric (Agent B)

**What**: Implement `compute_disagreement()` as a pure function.

```python
def compute_disagreement(
    results: tuple[EvalResult, ...],
    threshold: float = 0.3,
) -> DisagreementSignal:
    # Pairwise: max |score_a - score_b| across all pairs
    # Per-node: for each node, variance of per_node_scores across evaluators
    # Outlier: which evaluator's score differs most from the other two
    # should_surface: magnitude > threshold
    ...
```

**Test**: Hand-constructed triples:
- (0.9, 0.9, 0.9) → magnitude ~0.0, should_surface=False
- (0.9, 0.9, 0.1) → magnitude ~0.8, outlier="goodhart_resistant", should_surface=True
- (0.5, 0.3, 0.7) → magnitude ~0.4, no clear outlier, should_surface=True

**Duration**: 0.5 day. **Risk**: Low.

### Step 0.4: Correlation benchmark (Agent B, depends on A)

**What**: THE existential test.

```python
# evaluation/benchmark.py

def run_correlation_benchmark(
    lambda_rlm: LambdaRLM,
    evaluators: tuple[Evaluator, ...],
    dataset: list[Sample],          # from Lambda-RLM benchmark harness
    shifted_dataset: list[Sample],  # distribution-shifted variants
) -> CorrelationResult:
    """
    1. Run Lambda-RLM with tracing on both in-dist and shifted samples
    2. For each sample: TreeTrace -> evaluators -> disagreement -> F1
    3. Compute Spearman(disagreement.magnitude, 1 - F1)
    4. Compute precision@k
    """
    ...

@dataclass(frozen=True)
class CorrelationResult:
    spearman_rho: float
    spearman_p: float
    precision_at_5: float
    precision_at_10: float
    n_samples: int
    per_evaluator_correlation: dict[str, float]  # each evaluator vs failure
    per_pair_correlation: dict[str, float]        # each pair vs failure
```

**Distribution shift strategies** (using Lambda-RLM's existing SNIAH/OolongBench):
- Shift needle position to extreme beginning/end
- Replace haystack domain (swap all context to a different topic)
- Truncate context mid-sentence (simulate real-world messy data)
- Double context length beyond training range

**Success gate**: `spearman_rho >= 0.4` AND `spearman_p < 0.01` AND `precision_at_10 >= 0.7`

**If gate FAILS**:
1. Try alternative disagreement formulas (geometric mean, KL divergence)
2. Try weighted disagreement (weight by evaluator confidence)
3. Try different evaluator combinations (drop the weakest, try 4 evaluators)
4. If nothing works after 3 redesign attempts: pivot to single-evaluator + human anchor

**Duration**: 2 days. **Risk**: CRITICAL — this determines whether we proceed.

---

## Phase 1: Combinator Lifecycle (Agent C, starts parallel with Phase 0)

### Step 1.1: VersionedCombinator + SQLite store

```python
# combinators/versioned.py

class CombinatorDB:
    """SQLite store for versioned combinators and their penumbra."""

    def __init__(self, db_path: str = "~/.rsocas/combinators.db"):
        ...
        # Tables:
        # combinators(version_id, name, code_hash, status, created_at, expires_at,
        #             validation_json, repairs_json, cost_constant, type_signature)
        # penumbra(variant_id, parent_name, version_id, fitness_delta, created_at)
```

**Test**: CRUD operations, status transitions, expiry detection, penumbra query.

### Step 1.2: Crystallizer state machine

```
fluid ──[type_check_pass]──> crystallized ──[staleness > threshold]──> dissolving ──[dissolved]──> expired
  ^                                                                        |
  └───────────────────── [penumbra re-selection] ──────────────────────────┘
```

**Test**: Full lifecycle with synthetic distributions. Verify:
- Crystallization stores 3-5 near-miss variants in penumbra
- Staleness detection triggers at correct threshold
- Dissolution releases sub-components
- Re-selection from penumbra recovers performance

### Step 1.3: CombinatorRegistry for Lambda-RLM

Wraps `_register_library()` — version-stamps each combinator without changing behavior.

```python
# combinators/registry.py

class CombinatorRegistry:
    def __init__(self, crystallizer: Crystallizer):
        self.crystallizer = crystallizer

    def wrap_register_library(self, lrlm: LambdaRLM) -> None:
        """Extend _register_library to version-stamp combinators."""
        original = lrlm._register_library
        def versioned_register(repl, plan, query=""):
            original(repl, plan, query)
            # After registration, version-stamp each combinator
            for name in ["_Split", "_Peek", "_Reduce", "_FilterRelevant"]:
                if name in repl.globals:
                    vc = self.crystallizer.get_or_create(name, repl.globals[name])
                    # Store version info for trace metadata
                    ...
        lrlm._register_library = versioned_register
```

**Test**: Lambda-RLM runs identically with registry attached. Version IDs appear in TreeTrace.combinator_versions.

**Phase 1 Duration**: 3 days. **Risk**: Low (self-contained module).

---

## Phase 2: Breathing Cycle (Agent D, starts parallel with Phase 0)

### Step 2.1: TempoController (PID)

```python
# breathing/tempo.py

class TempoController:
    def __init__(self, kp=0.5, ki=0.1, kd=0.05, target_ratio=2.0):
        self._human_events: list[float] = []
        self._system_events: list[float] = []
        self._integral = 0.0
        self._prev_error = 0.0

    def record_human_feedback(self, timestamp: float) -> None:
        self._human_events.append(timestamp)

    def breathing_rate(self) -> float:
        human_freq = self._compute_frequency(self._human_events)
        system_freq = self._compute_frequency(self._system_events)
        error = (human_freq * self.target_ratio) - system_freq
        # PID
        self._integral += error
        derivative = error - self._prev_error
        self._prev_error = error
        adjustment = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(0.01, system_freq + adjustment)  # never negative

    def _compute_frequency(self, events: list[float], window: float = 3600.0) -> float:
        """Exponentially-weighted event frequency over window."""
        ...
```

**Test**: Simulate human feedback at 1/hour → system breathes at 2/hour. Increase human feedback to 5/hour → system slows to 10/hour. Remove human feedback → system speeds up crystallization.

### Step 2.2: AnnealingSchedule

```python
# breathing/annealing.py

class AnnealingSchedule:
    def __init__(self, t_init=1.0, t_min=0.01, cooling_rate=0.95):
        self.temperature = t_init
        self._history: list[float] = []

    def cool(self) -> float:
        self._history.append(self.temperature)
        self.temperature = max(self.t_min, self.temperature * self.cooling_rate)
        if self._at_phase_boundary():
            self.temperature *= 1.1  # slow down at boundaries
        return self.temperature

    def reheat(self, amount: float = 0.3) -> float:
        """Controlled reheat. Triggered by high disagreement or human feedback."""
        self.temperature = min(self.t_init, self.temperature + amount)
        return self.temperature

    def _at_phase_boundary(self) -> bool:
        """Detect phase boundary via second derivative of temperature trajectory."""
        if len(self._history) < 5:
            return False
        recent = self._history[-5:]
        # Phase boundary = inflection point in cooling curve
        ...

    def entropy_budget(self, combinator_staleness: float) -> float:
        """Higher staleness → higher entropy budget → more flexibility."""
        return self.temperature * (1.0 + combinator_staleness)
```

**Test**: Cooling curve is monotonically decreasing except at phase boundaries and reheats. Entropy budget increases with staleness.

### Step 2.3: InterferencePattern

```python
# breathing/interference.py

class InterferencePattern:
    def compute(self, human_freq: float, system_freq: float) -> float:
        """Constructive interference amplitude. Uses beat frequency model."""
        if human_freq < 1e-6:
            return 0.0  # no human signal
        beat_freq = abs(human_freq - system_freq)
        # Constructive when beat frequency is low (frequencies are close)
        # Destructive when beat frequency is high
        return 1.0 / (1.0 + beat_freq)

    def should_surface(self, disagreement: DisagreementSignal, interference: float) -> bool:
        """Surface when disagreement is high AND interference is constructive."""
        return disagreement.should_surface and interference > 0.5
```

**Phase 2 Duration**: 2 days. **Risk**: Low (standard control theory).

---

## Phase 3: Archive + Adapters (Agent E + F, starts after Phase 0 gate)

### Step 3.1: TraceArchive (Agent F)

SQLite + FTS5. Tables: `traces`, `evaluations`, `disagreements`, `repairs`.

Key queries:
- `query_by_failure(failure_type, limit)` — traces where specific evaluator flagged issues
- `query_repairs(combinator_name)` — (before, after) trace pairs (growth plates)
- `distribution_stats(task_type, window)` — running ValidationSnapshot for staleness

### Step 3.2: GEPA Tree Adapter (Agent E)

Implements Contract 3 using actual GEPA API:

```python
# adapters/gepa_tree_adapter.py

from gepa.api import optimize

class TreeTraceGEPAAdapter:
    def __init__(self, lambda_rlm: LambdaRLM, leaf_registry: LeafRegistry):
        ...

    def optimize_leaves(self, trainset, valset=None, max_metric_calls=200):
        """Optimize leaf prompts using GEPA with tree-structured credit assignment."""
        result = optimize(
            seed_candidate=self._extract_leaf_prompts(),
            trainset=trainset,
            valset=valset,
            adapter=self,  # self implements GEPAAdapter protocol
            reflection_lm="openai/gpt-5.1",
            candidate_selection_strategy="pareto",
            max_metric_calls=max_metric_calls,
            callbacks=[self._trace_callback],
        )
        self._apply_optimized_prompts(result.best_candidate)
        return result
```

### Step 3.3: DSPy Leaf Registry (Agent E)

Implements Contract 4. Key challenge: REPL injection.

```python
# adapters/dspy_leaf_registry.py

class DSPyLeafRegistry(LeafRegistry):
    def inject_into_repl(self, repl, task_type: str) -> None:
        """Replace repl's llm_query with a DSPy module call."""
        leaf_fn = self.get_leaf_fn(task_type)
        # Inject as the llm_query function in REPL globals
        # This way, Lambda-RLM's generated Phi code calls our DSPy module
        # at leaf nodes without any code generation changes
        repl.globals["llm_query"] = leaf_fn
        # Also need to update the _llm closure reference
        repl._llm_query = leaf_fn
```

**Phase 3 Duration**: 3 days. **Risk**: Medium (GEPA API integration).

---

## Phase 4: Developmental Stages + Orchestrator (Agent G, starts after Phases 0-2)

### Step 4.1: DevelopmentalStage

```python
class DevelopmentalStage(str, Enum):
    EMBRYONIC = "embryonic"       # Only Lambda-RLM, no evaluation
    FETAL = "fetal"               # + contrapuntal evaluation
    BORN = "born"                 # + breathing cycle
    CHILDHOOD = "childhood"       # + penumbra + archive
    ADOLESCENCE = "adolescence"   # + GEPA optimization + self-eval
    ADULT = "adult"               # + full autonomy with human anchoring
```

Feature enablement per stage — each stage enables exactly one new subsystem:
| Stage | New Feature | Transition Condition |
|-------|------------|---------------------|
| EMBRYONIC → FETAL | Contrapuntal eval | Phase 0 gate passes |
| FETAL → BORN | Breathing cycle | 100 consecutive traces with disagreement data |
| BORN → CHILDHOOD | Penumbra + archive | First successful dissolution-regermination cycle |
| CHILDHOOD → ADOLESCENCE | GEPA optimization | Archive has 500+ traces with repair episodes |
| ADOLESCENCE → ADULT | Full autonomy | Contrapuntal disagreement predicts failure with rho > 0.6 on live data |

### Step 4.2: ContinualLearningSystem orchestrator

```python
class ContinualLearningSystem:
    def run(self, prompt: str) -> Result:
        stage = self.development.current_stage

        # 1. Always: execute Lambda-RLM with tracing
        trace = self.execute_traced(prompt)

        # 2. FETAL+: evaluate contrapuntally
        if stage >= DevelopmentalStage.FETAL:
            evals = tuple(e.evaluate(trace) for e in self.evaluators)
            disagreement = compute_disagreement(evals)
        else:
            disagreement = None

        # 3. BORN+: tick breathing cycle
        if stage >= DevelopmentalStage.BORN:
            self.tempo.record_system_event(time.time(), "execution")
            events = self.crystallizer.tick(time.time(), disagreement)
            if disagreement and self.interference.should_surface(disagreement, ...):
                self.request_human_feedback(trace, disagreement)

        # 4. CHILDHOOD+: archive trace
        if stage >= DevelopmentalStage.CHILDHOOD:
            self.archive.store(trace, evals, disagreement)

        # 5. ADOLESCENCE+: periodic GEPA optimization
        if stage >= DevelopmentalStage.ADOLESCENCE:
            if self.should_optimize():
                self.gepa_adapter.optimize_leaves(self.archive.recent_traces())

        # 6. Check for developmental transition
        self.development.check_transition(self.compute_metrics())

        return Result(output=trace.final_output, trace=trace, disagreement=disagreement)
```

**Phase 4 Duration**: 3 days. **Risk**: Medium (integration complexity).

---

## Recurring Development Support

### Adding a New Evaluator

1. Implement `Evaluator` protocol in `evaluation/new_eval.py`
2. Add test in `tests/test_new_eval.py` with synthetic traces
3. Register in orchestrator's evaluator tuple
4. Disagreement metric automatically includes it (works with any number of evaluators)
5. No other module changes needed

### Adding a New Combinator

1. Write Python function with standard signature
2. Submit through `Crystallizer.crystallize()` — it type-checks, version-stamps, stores penumbra
3. Or let Meta-Harness discover it via `CombinatorCandidate` interface
4. Registry automatically wraps it for Lambda-RLM injection
5. No other module changes needed

### Adding a New DSPy Leaf Module

1. Subclass `LeafModule` in `adapters/dspy_leaf_registry.py`
2. Register with `LeafRegistry.register(task_type, module)`
3. GEPA adapter automatically picks it up for optimization
4. No Lambda-RLM code changes needed

### Adding a New Framework

1. Define contract in `contracts/new_framework.py`
2. Implement adapter in `adapters/new_adapter.py`
3. Wire into orchestrator at the appropriate developmental stage
4. Existing modules unchanged

---

## Dependency Graph (What Can Run In Parallel)

```
                    ┌─── Agent A (tracing) ──────────┐
                    │                                 │
Contracts ──────────┼─── Agent B (evaluation) ───────┤──── Phase 0 Gate
(30 min)            │                                 │      │
                    ├─── Agent C (combinators) ──────┤      │
                    │                                 │      │
                    ├─── Agent D (breathing) ─────────┤      │
                    │                                 │      │
                    └─── Agent F (archive) ───────────┘      │
                                                             │
                         Agent E (adapters) ─────────────────┤
                                                             │
                         Agent G (orchestrator) ──────────────┘
```

**Total estimated timeline**: 
- Day 1: Contracts + scaffold + Agents A-D,F start in parallel
- Day 3: Phase 0 gate (correlation benchmark)
- Day 4-6: Agents E,G start; Phases 1-2 complete
- Day 7-8: Phase 3-4 complete; integration testing
- Day 9: End-to-end test on Nemotron

---

## Non-Negotiable Rules

1. **Contracts are frozen after Step 0.0.** Any change requires all agents to acknowledge.
2. **Every module has tests.** No merging without passing tests.
3. **Lambda-RLM backward compatibility.** All existing benchmarks must produce identical results with `trace_enabled=False`.
4. **No mutation.** All dataclasses are `frozen=True`. Functions return new objects.
5. **Phase 0 gate is real.** If `spearman_rho < 0.3` after three redesign attempts, we pivot. No sunk cost.
6. **One module, one responsibility.** If a file exceeds 400 lines, split it.
7. **Adapters own framework dependencies.** Only `adapters/` imports GEPA, DSPy, or Meta-Harness. Everything else depends only on contracts.

---

**WAITING FOR CONFIRMATION**: Proceed with this plan?
