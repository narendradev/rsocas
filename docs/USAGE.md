# RSOCAS Usage Guide

## Quick Start: Add Contrapuntal Evaluation to Any LLM Pipeline

You do not need Lambda-RLM to use RSOCAS. The minimum viable integration: construct a `TreeTrace` manually, run three evaluators, check disagreement.

```python
import time
import uuid

from rsocas.contracts.traces import TreeTrace, NodeTrace, LeafTrace
from rsocas.evaluation.info_theoretic import InformationTheoreticEval
from rsocas.evaluation.boundary_detection import BoundaryDetectionEval
from rsocas.evaluation.goodhart_resistant import GoodhartResistantEval
from rsocas.evaluation.disagreement import compute_disagreement


# Step 1: Build a TreeTrace from your pipeline's execution
trace = TreeTrace(
    trace_id=uuid.uuid4().hex,
    task_type="QA",
    k=1,                    # branching factor (1 = single LLM call)
    depth=0,                # tree depth (0 = no decomposition)
    tau=5000,               # input size in characters
    cost_estimate=0.01,
    nodes=(
        NodeTrace(
            id="leaf_0",
            depth=0,
            position=0,
            combinator="leaf",
            input_size=5000,
            output="Paris is the capital of France.",
            children=(),
            llm_calls=1,
            latency_ms=350.0,
        ),
    ),
    leaf_traces=(
        LeafTrace(
            node_id="leaf_0",
            prompt="What is the capital of France? Context: ...",
            response="Paris is the capital of France.",
            tokens_in=200,
            tokens_out=15,
            model="gpt-4o",
        ),
    ),
    final_output="Paris is the capital of France.",
    timestamp=time.time(),
    execution_time_ms=350.0,
    total_llm_calls=1,
    total_tokens=215,
)

# Step 2: Evaluate with three voices
eval_info = InformationTheoreticEval().evaluate(trace)
eval_boundary = BoundaryDetectionEval().evaluate(trace)
eval_goodhart = GoodhartResistantEval().evaluate(trace)  # neutral without rerun_fn

# Step 3: Compute disagreement
signal = compute_disagreement(
    (eval_info, eval_boundary, eval_goodhart),
    threshold=0.3,
)

# Step 4: Act on the result
if signal.should_surface:
    print(f"HIGH DISAGREEMENT ({signal.magnitude:.2f}) -- flag for review")
    print(f"  Outlier voice: {signal.outlier_voice}")
else:
    print(f"Low disagreement ({signal.magnitude:.2f}) -- auto-accept")
```

---

## Integration with Lambda-RLM

### Step 1: Patch Lambda-RLM for tracing

```python
from rlm import LambdaRLM
from rsocas.tracing.patch import patch_for_tracing
from rsocas.tracing.builder import TreeTraceBuilder

# Create Lambda-RLM instance
lrlm = LambdaRLM(
    backend="openai",
    backend_kwargs={
        "model_name": "nemotron-3-super",
        "base_url": "http://localhost:8000/v1",
        "api_key": "dummy",
        "temperature": 0.6,
        "max_tokens": 4096,
    },
    context_window_chars=100_000,
)

# Patch for tracing -- this wraps _llm_query BEFORE _register_library
# captures it into closures, so both leaf and reduce calls are traced.
lrlm, collector = patch_for_tracing(lrlm)
builder = TreeTraceBuilder()
```

### Step 2: Run and collect trace

```python
import time

prompt = "Context:\n{long_document}\n\nQuestion: What is X?\n\nAnswer:"

collector.clear()
t0 = time.monotonic()
completion = lrlm.completion(prompt)
t1 = time.monotonic()

events = collector.get_events()

# Build a plan-like object from Lambda-RLM's execution
plan = type("Plan", (), {
    "k_star": max(2, len(events) // 2),
    "tau_star": 50_000,
    "depth": 1 if len(events) > 1 else 0,
    "cost_estimate": 0.0,
})()

trace = builder.build(events, plan, "QA", completion.response.strip(), t0, t1)
```

### Step 3: Evaluate

```python
from rsocas.evaluation.info_theoretic import InformationTheoreticEval
from rsocas.evaluation.boundary_detection import BoundaryDetectionEval
from rsocas.evaluation.goodhart_resistant import GoodhartResistantEval
from rsocas.evaluation.disagreement import compute_disagreement

# For Goodhart evaluation with actual re-run capability:
def rerun_leaf(prompt: str) -> str:
    from rlm.clients import get_client
    client = get_client("openai", backend_kwargs)
    return client.completion(prompt)

evaluators = (
    InformationTheoreticEval(),
    BoundaryDetectionEval(),
    GoodhartResistantEval(rerun_fn=rerun_leaf),
)

results = tuple(e.evaluate(trace) for e in evaluators)
disagreement = compute_disagreement(results, threshold=0.3)
```

### Full integration with versioning

For both tracing AND combinator version tracking:

```python
from rsocas.combinators.lambda_rlm_integration import patch_lambda_rlm_full

lrlm, collector, registry = patch_lambda_rlm_full(lrlm)

# After execution, the registry tracks which combinator versions were used
versions = registry.get_active_versions()
# e.g., {"_Split": "abc123", "_Reduce": "def456", ...}
```

---

## Integration with Any LLM Pipeline

If you are not using Lambda-RLM, construct `TreeTrace` objects manually. The key fields:

### Single-call pipeline (depth=0)

```python
trace = TreeTrace(
    trace_id=uuid.uuid4().hex,
    task_type="summarization",
    k=1, depth=0, tau=len(input_text),
    cost_estimate=0.0,
    nodes=(
        NodeTrace(
            id="node_0", depth=0, position=0, combinator="leaf",
            input_size=len(input_text), output=llm_response,
            children=(), llm_calls=1, latency_ms=latency,
        ),
    ),
    leaf_traces=(
        LeafTrace(
            node_id="node_0", prompt=full_prompt, response=llm_response,
            tokens_in=input_tokens, tokens_out=output_tokens, model=model_name,
        ),
    ),
    final_output=llm_response,
    timestamp=time.time(),
    execution_time_ms=latency,
    total_llm_calls=1,
    total_tokens=input_tokens + output_tokens,
)
```

### Multi-call pipeline (depth=1)

If your pipeline splits work across multiple LLM calls and then combines results:

```python
# Build leaf nodes (one per LLM call)
leaf_nodes = []
leaf_traces = []
for i, (prompt, response) in enumerate(call_results):
    nid = f"leaf_{i}"
    leaf_nodes.append(NodeTrace(
        id=nid, depth=1, position=i, combinator="leaf",
        input_size=len(prompt), output=response,
        children=(), llm_calls=1, latency_ms=call_latencies[i],
    ))
    leaf_traces.append(LeafTrace(
        node_id=nid, prompt=prompt, response=response,
        tokens_in=tokens_in[i], tokens_out=tokens_out[i], model=model_name,
    ))

# Build root node (the reduction/combination step)
root = NodeTrace(
    id="root", depth=0, position=0, combinator="reduce",
    input_size=sum(len(r) for _, r in call_results),
    output=final_combined_output,
    children=tuple(f"leaf_{i}" for i in range(len(call_results))),
    llm_calls=1, latency_ms=combine_latency,
)

trace = TreeTrace(
    trace_id=uuid.uuid4().hex,
    task_type="QA",
    k=len(call_results), depth=1, tau=max_chunk_size,
    cost_estimate=0.0,
    nodes=tuple([root] + leaf_nodes),
    leaf_traces=tuple(leaf_traces),
    final_output=final_combined_output,
    timestamp=time.time(),
    execution_time_ms=total_latency,
    total_llm_calls=len(call_results) + 1,
    total_tokens=sum(tokens_in) + sum(tokens_out),
)
```

---

## The Quality Gate Pattern

Use disagreement threshold as an automated quality gate:

```python
def quality_gate(trace: TreeTrace, evaluators, threshold=0.3):
    """Auto-accept or flag for human review based on evaluator disagreement."""
    results = tuple(e.evaluate(trace) for e in evaluators)
    signal = compute_disagreement(results, threshold=threshold)

    if not signal.should_surface:
        return {"action": "accept", "output": trace.final_output}

    # Find which nodes are problematic
    problem_nodes = sorted(
        signal.per_node.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:3]

    return {
        "action": "review",
        "output": trace.final_output,
        "disagreement": signal.magnitude,
        "outlier_voice": signal.outlier_voice,
        "problem_nodes": problem_nodes,
    }
```

---

## The Self-Improving Service Pattern

Wire all modules together for a system that improves over time:

```python
from rsocas.evaluation.info_theoretic import InformationTheoreticEval
from rsocas.evaluation.boundary_detection import BoundaryDetectionEval
from rsocas.evaluation.goodhart_resistant import GoodhartResistantEval
from rsocas.breathing.tempo import PIDTempoController
from rsocas.breathing.annealing import AnnealingSchedule
from rsocas.breathing.feedback_anchor import FeedbackAnchor
from rsocas.breathing.interference import InterferencePattern
from rsocas.archive.trace_archive import TraceArchive
from rsocas.development.stages import DevelopmentalController, DevelopmentalStage
from rsocas.development.orchestrator import ContinualLearningSystem

# Build the full system
evaluators = (
    InformationTheoreticEval(),
    BoundaryDetectionEval(),
    GoodhartResistantEval(rerun_fn=your_rerun_fn),  # or None
)

system = ContinualLearningSystem(
    evaluators=evaluators,
    tempo=PIDTempoController(),
    annealing=AnnealingSchedule(),
    archive=TraceArchive("./rsocas_traces.db"),      # persistent storage
    interference=InterferencePattern(),
    feedback_anchor=FeedbackAnchor(),
    development=DevelopmentalController(DevelopmentalStage.FETAL),
    disagreement_threshold=0.3,
)

# Process each trace through the continual learning cycle
result = system.run(trace)

# Check result
print(f"Stage: {result.stage.name}")
print(f"Disagreement: {result.disagreement.magnitude if result.disagreement else 'N/A'}")
print(f"Should surface for human: {result.surfaced_for_human}")

# When a human provides feedback
system.receive_human_feedback(time.time(), "correction")

# Check system status
status = system.status()
print(f"Total runs: {status.total_runs}")
print(f"Archive size: {status.archive_size}")
print(f"Breathing rate: {status.breathing_rate}")
print(f"Temperature: {status.temperature}")
```

---

## The Batch Processing Pattern

Process many documents, auto-flagging failures:

```python
flagged = []
accepted = []

for doc in documents:
    trace = build_trace_from_your_pipeline(doc)
    result = system.run(trace)

    if result.surfaced_for_human or (
        result.disagreement and result.disagreement.should_surface
    ):
        flagged.append({
            "doc": doc,
            "output": result.output,
            "disagreement": result.disagreement.magnitude,
            "outlier": result.disagreement.outlier_voice,
        })
    else:
        accepted.append(result.output)

print(f"Auto-accepted: {len(accepted)}, Flagged for review: {len(flagged)}")
```

---

## GEPA Optimization Loop

Feed tree-structured traces to GEPA for per-node credit assignment:

```python
from rsocas.adapters.gepa_tree_adapter import TreeTraceGEPAAdapter
from rsocas.archive.trace_archive import TraceArchive

adapter = TreeTraceGEPAAdapter()
archive = TraceArchive("./rsocas_traces.db")

# Get recent failures from the archive
failures = archive.query_by_failure(min_disagreement=0.3, limit=20)

# Convert to GEPA reflective dataset format
traces_with_disagreement = [
    (trace, disagreement, None)  # (trace, disagreement, ground_truth)
    for trace, disagreement in failures
]
dataset = adapter.adapt_for_gepa_optimize(traces_with_disagreement)

# The dataset is in GEPA's expected format:
# {
#     "leaf_prompt": [
#         {
#             "Inputs": {"context": "...", "task_type": "QA"},
#             "Generated Outputs": {"answer": "..."},
#             "Feedback": "Node [leaf_2] at depth [1], position [2] of [3]...",
#         },
#         ...
#     ]
# }

# Feed to GEPA for optimization (requires GEPA installation)
# gepa_optimizer.optimize(dataset)
```

The adapter's key innovation: the `Feedback` field contains surgical per-node context (the node's position in the tree, its siblings' outputs, its parent's combinator type) rather than a flat trace. This enables GEPA to do targeted optimization on specific failing nodes.

---

## Meta-Harness Combinator Discovery

Use the bridge to discover and validate new operations:

```python
from rsocas.adapters.metaharness_bridge import MetaHarnessBridge, CombinatorCandidate
from rsocas.contracts.combinators import ValidationSnapshot
from rsocas.combinators.registry import CombinatorRegistry

bridge = MetaHarnessBridge(
    candidates_dir="./candidates",
    archive_dir="./candidate_archive",
)

# 1. Generate a skill prompt for Meta-Harness's proposer agent
skill_md = bridge.generate_skill_md(
    "Discover new reduction combinators for multi-document QA "
    "that preserve cross-reference relationships."
)

# 2. Validate a candidate discovered by Meta-Harness
candidate = CombinatorCandidate(
    name="cross_ref_reduce",
    code='''
def cross_ref_reduce(chunks: list[str]) -> str:
    """Reduce chunks preserving cross-references.

    Hypothesis: Sorting chunks by shared entity count before
    concatenation preserves cross-reference context.
    """
    # Extract simple entity overlap counts
    entity_sets = []
    for chunk in chunks:
        words = set(chunk.lower().split())
        proper = {w for w in words if w[0:1].isupper()} if words else set()
        entity_sets.append(proper)

    # Sort by overlap with other chunks (most connected first)
    overlap_scores = []
    for i, es in enumerate(entity_sets):
        score = sum(len(es & other) for j, other in enumerate(entity_sets) if i != j)
        overlap_scores.append((score, i))
    overlap_scores.sort(reverse=True)

    ordered = [chunks[idx] for _, idx in overlap_scores]
    return "\\n---\\n".join(ordered)
''',
    type_signature="(list[str]) -> str",
    hypothesis="Sorting by entity overlap preserves cross-references",
)

valid, msg = bridge.validate_candidate(candidate)
print(f"Valid: {valid}, Message: {msg}")
# Checks: syntax OK, no unbounded loops, no recursion, callable found

if valid:
    # 3. Write candidate to disk for further testing
    path = bridge.write_candidate(candidate)

    # 4. After testing, archive with validation results
    validation = ValidationSnapshot(
        task_types=("QA",),
        input_size_range=(5000, 50000),
        n_samples=20,
        mean_score=0.85,
        score_std=0.08,
        timestamp=time.time(),
    )
    bridge.archive_candidate(candidate, validation, accepted=True)
```

The bridge's validation checks:
1. **Syntax**: code parses without error.
2. **Bounded execution**: no `while True` loops, no recursive self-calls (AST analysis).
3. **Exec safety**: code can be executed without side effects.
4. **Callable**: a function with the candidate's name (or any callable) exists in the namespace.

---

## Configuration Guide

### Disagreement threshold

**Parameter**: `threshold` in `compute_disagreement()` and `disagreement_threshold` in `ContinualLearningSystem`
**Default**: 0.3
**Effect**: Controls when `should_surface` is True. Higher = fewer surfacing events, lower = more human review.
**Recommended**: 0.3 for production (balances false positives/negatives). 0.2 for safety-critical applications. 0.5 for high-throughput batch processing.

### PID gains

**Parameters**: `kp=0.5, ki=0.1, kd=0.05` in `PIDTempoController`
**Effect**: Controls how aggressively breathing rate responds to the gap between human feedback frequency and system frequency.
- Higher `kp`: faster response, risk of oscillation.
- Higher `ki`: eliminates steady-state error, risk of windup.
- Higher `kd`: anticipates changes, amplifies noise.
**Recommended**: Start with defaults. Tune `kp` first if the system feels too sluggish (increase) or jittery (decrease).

### Annealing schedule

**Parameters**: `t_init=1.0, t_min=0.01, cooling_rate=0.95` in `AnnealingSchedule`
**Effect**:
- `t_init`: Starting temperature. Higher = longer before crystallization begins.
- `t_min`: Floor temperature. The system never fully freezes.
- `cooling_rate`: Multiplicative factor per step. 0.95 = 5% cooling per step.
**Recommended**: Defaults work for most cases. Lower `cooling_rate` (e.g., 0.99) if combinators are crystallizing too quickly. Increase `t_init` for exploratory phases.

### Combinator TTL

**Parameter**: `default_ttl` in `Crystallizer`
**Default**: 86400.0 seconds (24 hours)
**Effect**: How long a crystallized combinator lives before expiring. Expired combinators transition to "dissolving".
**Recommended**: 86400 for daily-cycle systems. Shorter (3600 = 1 hour) for rapidly evolving domains. Longer (604800 = 1 week) for stable production systems.

### Reheat threshold

**Parameter**: `reheat_threshold` in `BreathingCrystallizer`
**Default**: 0.5
**Effect**: When disagreement magnitude exceeds this, the annealing schedule reheats. This prevents premature crystallization when evaluators detect problems.
**Recommended**: 0.5 (reheat only on significant disagreement). Lower to 0.3 if you want more aggressive adaptation.

### Feedback anchor half-life

**Parameter**: `half_life` in `FeedbackAnchor` and `PIDTempoController`
**Default**: 1800 seconds (30 minutes) for `FeedbackAnchor`, 900 seconds (15 minutes) for `PIDTempoController`
**Effect**: Controls exponential weighting of human feedback events. Shorter half-life = recent feedback counts more. Longer = smoother frequency estimate.
**Recommended**: 1800 for typical human review cadences. Shorter (600) for real-time human-in-the-loop systems.

### Penumbra variant limit

**Parameter**: `max_variants` in `PenumbraStore.prune()` and `CombinatorDB.prune_penumbra()`
**Default**: 10
**Effect**: Maximum near-miss variants kept per parent combinator.
**Recommended**: 10 is sufficient for most cases. Increase to 50 if distribution shifts are frequent and diverse.

### Developmental stage transitions

Thresholds are hardcoded in `DevelopmentalController._meets_threshold()`:

| Transition | Threshold |
|-----------|-----------|
| EMBRYONIC -> FETAL | Always allowed |
| FETAL -> BORN | 100 consecutive traces with evaluation |
| BORN -> CHILDHOOD | 1 successful dissolution |
| CHILDHOOD -> ADOLESCENCE | 500 archived traces |
| ADOLESCENCE -> ADULT | disagreement_correlation >= 0.6 |

These are not configurable at runtime. To change them, subclass `DevelopmentalController`.

---

## API Reference

### Contracts (`rsocas.contracts`)

#### `traces.py`

- **`NodeTrace`** -- Frozen dataclass. One node in the execution tree.
  - `id: str`, `depth: int`, `position: int`, `combinator: str`, `input_size: int`, `output: str`, `children: tuple[str, ...]`, `llm_calls: int`, `latency_ms: float`, `score: float | None`

- **`LeafTrace`** -- Frozen dataclass. One LLM call at a leaf.
  - `node_id: str`, `prompt: str`, `response: str`, `tokens_in: int`, `tokens_out: int`, `model: str`, `confidence: float | None`

- **`TreeTrace`** -- Frozen dataclass. Complete execution record.
  - `trace_id: str`, `task_type: str`, `k: int`, `depth: int`, `tau: int`, `cost_estimate: float`, `nodes: tuple[NodeTrace, ...]`, `leaf_traces: tuple[LeafTrace, ...]`, `final_output: str`, `timestamp: float`, `execution_time_ms: float`, `total_llm_calls: int`, `total_tokens: int`, `final_score: float | None`, `combinator_versions: dict[str, str]`, `metadata: dict`

#### `evaluation.py`

- **`EvalResult`** -- Frozen dataclass.
  - `score: float`, `confidence: float`, `signal_type: str`, `per_node_scores: dict[str, float]`, `explanation: str`

- **`DisagreementSignal`** -- Frozen dataclass.
  - `magnitude: float`, `pairwise: dict[str, float]`, `per_node: dict[str, float]`, `outlier_voice: str | None`, `should_surface: bool`, `timestamp: float`

- **`Evaluator`** -- Protocol.
  - `signal_type: str` (property), `evaluate(trace: TreeTrace, ground_truth: str | None = None) -> EvalResult`

#### `combinators.py`

- **`ValidationSnapshot`** -- Frozen dataclass. Distribution metadata.
  - `task_types: tuple[str, ...]`, `input_size_range: tuple[int, int]`, `n_samples: int`, `mean_score: float`, `score_std: float`, `timestamp: float`

- **`RepairRecord`** -- Frozen dataclass.
  - `timestamp: float`, `trigger: str`, `from_version: str`, `change_summary: str`, `score_delta: float`

- **`VersionedCombinator`** -- Frozen dataclass.
  - `name: str`, `version_id: str`, `code_hash: str`, `status: str`, `created_at: float`, `expires_at: float`, `validation: ValidationSnapshot`, `repairs: tuple[RepairRecord, ...]`, `cost_constant: float`, `type_signature: str`

### Tracing (`rsocas.tracing`)

- **`TraceCollector`** -- Thread-safe event accumulator.
  - `start_call(prompt, model, call_context) -> str` (returns call_id)
  - `end_call(call_id, response, tokens_in, tokens_out)`
  - `get_events() -> list[CallEvent]`
  - `clear()`

- **`TreeTraceBuilder`** -- Reconstructs tree from flat events.
  - `build(events, plan, task_type, final_output, start_time, end_time) -> TreeTrace`

- **`patch_for_tracing(lrlm) -> tuple[lrlm, TraceCollector]`** -- Monkey-patches Lambda-RLM for tracing.

### Evaluation (`rsocas.evaluation`)

- **`InformationTheoreticEval`** -- Measures information preservation. `signal_type = "information_theoretic"`.
- **`BoundaryDetectionEval`** -- Detects echo/copying. `signal_type = "boundary"`. Optional `similarity_fn` constructor arg.
- **`GoodhartResistantEval`** -- Perturbation robustness. `signal_type = "goodhart_resistant"`. Optional `perturb_fn` and `rerun_fn` constructor args. Returns neutral result (score=0.5, confidence=0.0) if `rerun_fn` is None.
- **`compute_disagreement(results, threshold=0.3, timestamp=0.0) -> DisagreementSignal`** -- Pure function.

### Combinators (`rsocas.combinators`)

- **`CombinatorDB(db_path=":memory:")`** -- SQLite store. `store()`, `load()`, `load_active()`, `update_status()`, `list_by_status()`, `store_penumbra()`, `load_penumbra()`, `prune_penumbra()`.
- **`PenumbraStore(db)`** -- Near-miss variant manager. `store_variant()`, `retrieve_candidates()`, `prune()`.
- **`Crystallizer(db, penumbra, default_ttl=86400.0)`** -- Lifecycle state machine. `crystallize()`, `dissolve()`, `expire()`, `tick()`, `check_staleness()`, `get_or_create()`.
- **`CombinatorRegistry(crystallizer)`** -- Top-level facade. `register()`, `get_active_versions()`, `get_active()`.
- **`patch_lambda_rlm_full(lrlm, registry=None) -> tuple[lrlm, TraceCollector, CombinatorRegistry]`** -- Full integration patch.

### Breathing (`rsocas.breathing`)

- **`PIDTempoController(kp, ki, kd, target_ratio, window, half_life)`** -- PID tempo control. `record_human_feedback()`, `record_system_event()`, `breathing_rate()`, `should_crystallize()`, `should_dissolve()`, `get_state() -> TempoState`.
- **`AnnealingSchedule(t_init, t_min, cooling_rate)`** -- Simulated annealing. `cool()`, `reheat(amount)`, `at_phase_boundary()`, `entropy_budget(staleness)`, `get_state() -> AnnealingState`. Property: `temperature`.
- **`FeedbackAnchor(half_life=1800.0)`** -- Human feedback tracker. `record()`, `frequency()`, `time_since_last()`, `feedback_density()`.
- **`InterferencePattern`** -- Standing wave model. `compute(human_freq, system_freq) -> float`, `should_surface(disagreement, human_freq, system_freq, min_interval, last_surface_time) -> bool`.
- **`BreathingCrystallizer(crystallizer, tempo, annealing, reheat_threshold=0.5)`** -- Orchestrates breathing. `tick(current_time, disagreement) -> list[BreathingEvent]`, `receive_human_feedback(timestamp) -> BreathingEvent`.

### Archive (`rsocas.archive`)

- **`TraceArchive(db_path=":memory:")`** -- SQLite + FTS5 storage. `store()`, `load()`, `query_by_task_type()`, `query_by_failure()`, `query_by_combinator_version()`, `search_output()`, `count()`.
- **`RepairIndex(archive)`** -- Repair episode tracker. `record_repair()`, `query_repairs()`, `query_recent_repairs()`.
- **`DistributionTracker(archive)`** -- Distribution snapshot computer. `compute_snapshot(task_type, window_seconds) -> ValidationSnapshot`.

### Adapters (`rsocas.adapters`)

- **`TreeTraceGEPAAdapter`** -- Tree trace to GEPA format. `identify_failing_nodes()`, `extract_subtree_context()`, `make_reflective_dataset()`, `adapt_for_gepa_optimize()`.
- **`DSPyLeafRegistry`** -- DSPy module registry. `register()`, `get_leaf_fn()`, `create_dspy_module()`, `optimize()`, `inject_into_repl()`, `list_registered()`.
- **`MetaHarnessBridge(candidates_dir, archive_dir)`** -- Combinator discovery bridge. `validate_candidate()`, `write_candidate()`, `load_candidates()`, `archive_candidate()`, `generate_skill_md()`.

### Development (`rsocas.development`)

- **`DevelopmentalStage`** -- IntEnum: EMBRYONIC(0), FETAL(1), BORN(2), CHILDHOOD(3), ADOLESCENCE(4), ADULT(5).
- **`DevelopmentalController(initial_stage)`** -- Stage controller. `check_transition()`, `force_transition()`, `get_enabled_features()`. Property: `current_stage`, `transition_history`.
- **`ContinualLearningSystem(...)`** -- Top-level orchestrator. `run(trace) -> RunResult`, `receive_human_feedback()`, `status() -> SystemStatus`.
- **`RunResult`** -- Frozen dataclass: `output`, `trace`, `evaluations`, `disagreement`, `stage`, `surfaced_for_human`.
- **`SystemStatus`** -- Frozen dataclass: `stage`, `enabled_features`, `total_runs`, `archive_size`, `active_combinators`, `breathing_rate`, `temperature`.
