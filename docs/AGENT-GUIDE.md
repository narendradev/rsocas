# RSOCAS Agent Guide

This guide is for AI agents that will use RSOCAS programmatically to evaluate their own outputs, improve their own prompts, and know when to ask a human for help.

---

## When to Use RSOCAS

Use this decision tree:

```
Are you making multiple LLM calls to answer one question?
  YES --> RSOCAS gives strong signal (multi-node traces have rich structure)
  NO  --> Are you processing long documents (>10K tokens)?
            YES --> Lambda-RLM + RSOCAS is the full solution
            NO  --> Is the cost of a wrong answer high?
                      YES --> RSOCAS evaluation adds a cheap quality gate
                      NO  --> RSOCAS adds overhead with limited benefit. Skip it.
```

More specifically, RSOCAS is valuable when:

1. **You decompose work across multiple LLM calls** -- the evaluators compare parent vs child outputs, so multi-node traces give them real signal.
2. **You need to know when you are wrong** -- disagreement between three evaluators detects failure modes that no single metric catches.
3. **You want to improve over time** -- the archive stores every execution, enabling GEPA optimization and distribution shift detection.
4. **You need to decide whether to ask a human** -- the interference pattern logic tells you when human feedback will have maximum impact.

RSOCAS is NOT valuable when:
- You make a single short LLM call with a clear right/wrong answer.
- Latency is your binding constraint (evaluation adds overhead).
- You have constant human oversight already.

---

## Minimum Viable Integration

The smallest useful integration is four lines after any LLM execution:

```python
from rsocas.contracts.traces import TreeTrace, NodeTrace, LeafTrace
from rsocas.evaluation.info_theoretic import InformationTheoreticEval
from rsocas.evaluation.boundary_detection import BoundaryDetectionEval
from rsocas.evaluation.goodhart_resistant import GoodhartResistantEval
from rsocas.evaluation.disagreement import compute_disagreement

# You already have: prompt, response, model_name, latency_ms
# Build the minimal trace:
trace = TreeTrace(
    trace_id="unique-id",
    task_type="your_task",
    k=1, depth=0, tau=len(prompt),
    cost_estimate=0.0,
    nodes=(NodeTrace(
        id="n0", depth=0, position=0, combinator="leaf",
        input_size=len(prompt), output=response, children=(),
    ),),
    leaf_traces=(LeafTrace(
        node_id="n0", prompt=prompt, response=response,
    ),),
    final_output=response,
    timestamp=0.0, execution_time_ms=latency_ms,
)

# Evaluate:
results = tuple(e.evaluate(trace) for e in (
    InformationTheoreticEval(),
    BoundaryDetectionEval(),
    GoodhartResistantEval(),
))
signal = compute_disagreement(results)

# Act:
if signal.should_surface:
    # Flag for review or re-try with different approach
    pass
```

This adds zero LLM calls (the `GoodhartResistantEval` returns a neutral score without a `rerun_fn`). The information-theoretic and boundary evaluators use only compression and word overlap -- no external dependencies.

---

## Reading Disagreement Signals

### The DisagreementSignal object

```python
signal = compute_disagreement(results, threshold=0.3)
```

- **`signal.magnitude`** -- A float in [0, 1]. The maximum pairwise score difference across all evaluator pairs. This is your primary quality indicator.
  - magnitude < 0.15: evaluators agree. High confidence the output is acceptable.
  - magnitude 0.15-0.30: mild disagreement. Probably fine, but monitor.
  - magnitude 0.30-0.50: significant disagreement. Consider re-trying or reviewing.
  - magnitude > 0.50: strong disagreement. At least one evaluator detects a problem.

- **`signal.should_surface`** -- Boolean. True when magnitude >= threshold. This is the binary "should I flag this?" signal.

- **`signal.outlier_voice`** -- Which evaluator disagrees most from the mean. Tells you the type of problem:
  - `"information_theoretic"` -- information was lost during composition. The reduction step discarded important content.
  - `"boundary"` -- the system echoed one of its inputs instead of synthesizing. A self-referential loop.
  - `"goodhart_resistant"` -- the output is fragile to irrelevant perturbations. The model is latching onto surface features.

- **`signal.pairwise`** -- Dict of `"typeA_vs_typeB" -> float`. Shows which evaluator pairs disagree and by how much.

- **`signal.per_node`** -- Dict of `"node_id" -> float` (variance). Shows which specific nodes in the tree have the most evaluator disagreement. High-variance nodes are where the problem lives.

### Interpreting per-evaluator scores

Each `EvalResult` has a `score` in [0, 1] and `per_node_scores` dict:

| Evaluator | High score means | Low score means |
|-----------|-----------------|-----------------|
| `information_theoretic` | Information preserved through the tree | Reduction steps lost content |
| `boundary` | Outputs are distinct from inputs | Echoing/copying detected |
| `goodhart_resistant` | Stable under perturbation | Fragile to irrelevant changes |

### Example: diagnosing a flagged trace

```python
if signal.should_surface:
    # Which evaluator is the outlier?
    if signal.outlier_voice == "boundary":
        # The system is echoing. Look at which node is copying.
        worst_node = max(signal.per_node.items(), key=lambda x: x[1])
        print(f"Node {worst_node[0]} may be echoing (variance: {worst_node[1]:.3f})")

    elif signal.outlier_voice == "information_theoretic":
        # Information loss. The reduce step is discarding content.
        print("Reduction step may be losing information")

    elif signal.outlier_voice == "goodhart_resistant":
        # Fragile output. The model is not robust.
        print("Output is sensitive to irrelevant perturbations")
```

---

## When to Surface for Human Feedback

The `InterferencePattern` module models constructive/destructive interference between your execution frequency and human feedback frequency.

### The logic

The system surfaces when ALL three conditions hold:

1. **Disagreement says to surface** -- `signal.should_surface == True` (magnitude >= threshold).
2. **Constructive interference** -- human and system frequencies are close (amplitude > 0.5). This means the human is actively engaged. Surfacing during constructive interference maximizes the impact of human feedback.
3. **Rate limiting** -- enough time has passed since the last surfacing event.

### In agent terms

- If you are processing a batch and the human has not given feedback recently, the interference is destructive. Even if disagreement is high, do NOT surface -- queue the flagged items for later review.
- If the human just gave feedback (constructive interference) and disagreement is high, surface immediately -- this is the moment of maximum reorganization.
- If disagreement is low, never surface regardless of interference.

### Using it

```python
from rsocas.breathing.interference import InterferencePattern
from rsocas.breathing.feedback_anchor import FeedbackAnchor

ip = InterferencePattern()
anchor = FeedbackAnchor()

# Record human interactions
anchor.record(time.time(), "correction")

# Check whether to surface
should_ask = ip.should_surface(
    disagreement=signal,
    human_freq=anchor.frequency(),
    system_freq=traces_per_hour,  # your execution rate
    min_interval=300.0,           # don't ask more than once per 5 minutes
    last_surface_time=last_time_you_asked,
)
```

---

## Self-Improvement Protocol

Step-by-step for an agent to improve its own prompts using the GEPA adapter.

### Step 1: Accumulate failure data

Run your pipeline through the `ContinualLearningSystem`. Traces with high disagreement are automatically archived.

```python
# After many runs:
archive = system._archive  # or your TraceArchive instance
failures = archive.query_by_failure(min_disagreement=0.3, limit=50)
```

### Step 2: Build GEPA reflective dataset

```python
from rsocas.adapters.gepa_tree_adapter import TreeTraceGEPAAdapter

adapter = TreeTraceGEPAAdapter()
traces_data = [(trace, disagreement, None) for trace, disagreement in failures]
dataset = adapter.adapt_for_gepa_optimize(traces_data, component_name="leaf_prompt")
```

The dataset contains, for each failing node:
- **Inputs**: the chunk text and task type.
- **Generated Outputs**: what the node actually produced.
- **Feedback**: surgical context -- the node's position, its siblings' outputs, its parent's combinator type. This is not a flat summary. It is per-node credit assignment.

### Step 3: Feed to GEPA

If GEPA is installed:
```python
# gepa_optimizer = GEPA(...)
# gepa_optimizer.optimize(program, trainset=dataset["leaf_prompt"])
```

If GEPA is not installed, the dataset is still useful as structured training data for any optimization approach (manual prompt tuning, DSPy optimizers, etc.).

### Step 4: Validate improvement

After optimization, run the same inputs through the improved pipeline and compare disagreement magnitudes:

```python
# Before optimization: mean disagreement on failure set
before = [d.magnitude for _, d in failures]

# After optimization: re-run the same inputs
after = [compute_disagreement(tuple(e.evaluate(new_trace) for e in evaluators)).magnitude
         for new_trace in new_traces]

improvement = sum(before) / len(before) - sum(after) / len(after)
```

---

## Combinator Discovery Protocol

How an agent proposes, validates, and crystallizes new operations.

### Step 1: Identify the need

Look at archived failures. If many failures share a pattern (e.g., cross-reference information is always lost in reduction), propose a new combinator.

### Step 2: Generate candidate

Use the `MetaHarnessBridge` to generate a skill prompt for a proposer agent:

```python
from rsocas.adapters.metaharness_bridge import MetaHarnessBridge, CombinatorCandidate

bridge = MetaHarnessBridge()
skill_md = bridge.generate_skill_md(
    "Discover a reduction combinator that preserves cross-reference "
    "relationships by sorting chunks by entity overlap before merging."
)
# Feed skill_md to an LLM agent to generate candidate code
```

### Step 3: Validate

```python
candidate = CombinatorCandidate(
    name="entity_aware_reduce",
    code=proposed_code,
    type_signature="(list[str]) -> str",
    hypothesis="Entity-sorted reduction preserves cross-references",
)

valid, msg = bridge.validate_candidate(candidate)
if not valid:
    print(f"Rejected: {msg}")
    # Fix and re-try
```

Validation checks syntax, bounded execution (no infinite loops, no recursion), and callability.

### Step 4: Test

Run the new combinator on archived traces and compare with the existing combinator:

```python
# Manually integrate into your pipeline and compare disagreement
```

### Step 5: Crystallize

If the combinator performs well:

```python
from rsocas.contracts.combinators import ValidationSnapshot

validation = ValidationSnapshot(
    task_types=("QA",),
    input_size_range=(5000, 50000),
    n_samples=50,
    mean_score=0.88,
    score_std=0.06,
    timestamp=time.time(),
)

registry.register("entity_aware_reduce", actual_function, validation)
bridge.archive_candidate(candidate, validation, accepted=True)
```

### Step 6: Monitor for staleness

The `DistributionTracker` computes current distribution snapshots. Compare against the combinator's stored validation:

```python
from rsocas.archive.distribution_tracker import DistributionTracker

tracker = DistributionTracker(archive)
current = tracker.compute_snapshot("QA", window_seconds=3600)
staleness = crystallizer.check_staleness(combinator.version_id, current)

if staleness > 1.0:
    # Distribution has drifted -- consider dissolution
    crystallizer.dissolve(combinator.version_id, "distribution drift")
```

---

## Anti-Patterns

### 1. Ignoring disagreement

**Wrong**: Running evaluation but not acting on `should_surface == True`.

The whole point of contrapuntal evaluation is that disagreement is the signal. If you ignore it, you are running three evaluators for nothing. At minimum, log flagged traces for later review.

### 2. Crystallizing too fast

**Wrong**: Immediately crystallizing a combinator after one good run.

Crystallization should happen only after the combinator has been validated on a meaningful sample (`n_samples >= 10`) across the relevant task types. The annealing schedule exists to prevent premature crystallization -- respect the temperature.

### 3. Bypassing the type checker

**Wrong**: Constructing `TreeTrace` objects with inconsistent data (e.g., `children` referencing node IDs that do not exist in `nodes`, or `depth` not matching actual tree structure).

The evaluators rely on the tree structure being correct. `InformationTheoreticEval` traverses children to compute information ratios. `BoundaryDetectionEval` compares parent/child outputs. Bad structure = meaningless evaluations.

### 4. Running without the archive

**Wrong**: Using evaluation but never storing traces.

The archive is what makes the system self-improving. Without it:
- No GEPA optimization (no failure data to learn from).
- No distribution tracking (no staleness detection).
- No repair index (no learning from past fixes).
- No developmental progression (archive_size is a transition condition).

### 5. Setting the GoodhartResistantEval rerun_fn to the same model with temperature=0

**Wrong**: Re-running with deterministic settings for perturbation testing.

The point is to test whether the response changes on irrelevant perturbation. If the model is deterministic, perturbation testing is testing something different (whether the model's deterministic behavior is robust to token boundary shifts). Use temperature > 0 for the rerun to get a meaningful perturbation test.

### 6. Treating developmental stages as configuration

**Wrong**: Starting at ADULT stage to "unlock all features."

```python
# DO NOT DO THIS:
dev = DevelopmentalController(DevelopmentalStage.ADULT)
```

Developmental stages exist to ensure the system has demonstrated competence at each level before gaining more autonomy. Skipping stages means the system has not proven that its evaluation correlates with actual quality (`ADULT` requires `disagreement_correlation >= 0.6`).

Use `force_transition()` only in tests, never in production.

### 7. Assuming more evaluators = better

**Wrong**: Adding ten evaluators to get "more signal."

The contrapuntal architecture was designed for three voices because:
- Three is the minimum for meaningful disagreement (two evaluators either agree or disagree -- no nuance).
- Three independent perspectives cover each other's blind spots.
- Adding more evaluators increases the chance that at least two will agree by chance, diluting the disagreement signal.

If you need a fourth evaluator, it should replace one of the existing three, not supplement them.

### 8. Compacting the archive

**Wrong**: Deleting old traces to save disk space.

The archive currently has no compaction strategy (this is a known limitation). Deleting traces removes training data for GEPA, breaks repair index links, and invalidates distribution tracking. If disk space is a concern, move old traces to cold storage rather than deleting them.
