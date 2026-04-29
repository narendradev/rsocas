# RSOCAS Architecture

## System Overview

RSOCAS (Recursive Self-Optimizing Compound AI System) is a framework for adding self-evaluation, self-improvement, and temporal lifecycle management to compound AI systems that make multiple LLM calls. It synthesizes ideas from five research systems:

| Framework | What it contributes to RSOCAS | Role in the synthesis |
|-----------|-------------------------------|----------------------|
| **Lambda-RLM** | Typed tree decomposition with formal cost/accuracy bounds | Execution engine producing structured traces |
| **DSPy** | Programmatic LLM module compilation | Leaf node optimization via modular signatures |
| **GEPA** | Prompt optimization via full execution trace reflection | Per-node credit assignment from tree traces |
| **Meta-Harness** | Agentic code search over scaffold structure | Combinator discovery and validation |
| **LangProBe** | Cost-quality Pareto evaluation | Evaluation philosophy (three cheap signals > one expensive one) |

The central thesis: these five frameworks are different views of the same underlying system. Lambda-RLM sees a mathematical tree. DSPy sees a program. GEPA sees optimizable text parameters. Meta-Harness sees mutable code. LangProBe sees a point in cost-quality space. RSOCAS provides the connective tissue that lets each view inform the others.

The practical result: every LLM execution produces a `TreeTrace` that three evaluators analyze independently. Their disagreement tells you whether the output is trustworthy. Over time, the system uses these traces to improve its own prompts (via GEPA), discover new operations (via Meta-Harness), and know when to ask a human for help.

---

## Layer Architecture

RSOCAS has seven layers. Information flows upward (traces), downward (configuration), and laterally (cross-layer adaptation).

```
                          ┌─────────────────────────────────────┐
                          │  Layer 7: TEMPORAL EVOLUTION         │
                          │  DevelopmentalController, stages     │
                          │  EMBRYONIC → FETAL → BORN →          │
                          │  CHILDHOOD → ADOLESCENCE → ADULT     │
                          └──────────────┬──────────────────────┘
                                         │ governs which layers are active
                          ┌──────────────▼──────────────────────┐
                          │  Layer 6: BREATHING CYCLE            │
                          │  PIDTempoController, AnnealingSchedule│
                          │  BreathingCrystallizer               │
                          │  InterferencePattern, FeedbackAnchor │
                          └──────────────┬──────────────────────┘
                                         │ controls crystallization timing
                          ┌──────────────▼──────────────────────┐
                          │  Layer 5: COMBINATOR LIFECYCLE       │
                          │  Crystallizer, PenumbraStore          │
                          │  CombinatorRegistry, CombinatorDB   │
                          │  MetaHarnessBridge                   │
                          └──────────────┬──────────────────────┘
                                         │ version-stamps operations
                          ┌──────────────▼──────────────────────┐
                          │  Layer 4: ARCHIVE & MEMORY           │
                          │  TraceArchive (SQLite + FTS5)         │
                          │  RepairIndex, DistributionTracker    │
                          └──────────────┬──────────────────────┘
                                         │ stores traces for learning
                          ┌──────────────▼──────────────────────┐
                          │  Layer 3: CONTRAPUNTAL EVALUATION    │
                          │  InformationTheoreticEval            │
                          │  BoundaryDetectionEval               │
                          │  GoodhartResistantEval               │
                          │  compute_disagreement()              │
                          └──────────────┬──────────────────────┘
                                         │ evaluates every trace
                          ┌──────────────▼──────────────────────┐
                          │  Layer 2: FRAMEWORK ADAPTERS          │
                          │  TreeTraceGEPAAdapter                │
                          │  DSPyLeafRegistry                    │
                          │  MetaHarnessBridge                   │
                          └──────────────┬──────────────────────┘
                                         │ translates between systems
                          ┌──────────────▼──────────────────────┐
                          │  Layer 1: EXECUTION & TRACING         │
                          │  TraceCollector, TreeTraceBuilder    │
                          │  patch_for_tracing()                 │
                          │  patch_lambda_rlm_full()             │
                          └─────────────────────────────────────┘
```

### Information flow

**Upward (traces):** Layer 1 produces `TreeTrace` objects. Layer 3 evaluates them. Layer 4 archives them. Layer 5 uses evaluation results to decide combinator staleness. Layer 6 uses disagreement magnitude to adjust breathing tempo. Layer 7 uses accumulated metrics to trigger developmental transitions.

**Downward (configuration):** Layer 7 controls which layers are active (e.g., evaluation is disabled during the EMBRYONIC stage). Layer 6 controls crystallization timing. Layer 5 version-stamps operations that Layer 1 executes.

**Lateral (adaptation):** Layer 2 adapters translate Layer 1 traces into formats that external systems (GEPA, DSPy, Meta-Harness) consume. Layer 4 provides historical traces to Layer 2 for optimization.

---

## The Embryogenesis Model

RSOCAS does not iterate toward an optimal configuration. It develops through irreversible stages, each enabling additional capabilities. This is a deliberate design choice from the dream session (see `dreams/2026-04-28-continual-learning-dream-trace.md`): optimization loops revisit the same space (circular, self-referential, vulnerable to Goodhart's Law). Developmental cascades are directional -- each commitment narrows the future but enriches the system's surface area.

The stages are defined in `rsocas/development/stages.py`:

```
Stage           Value   Enabled Features                          Transition Condition
──────────────  ─────   ──────────────────────────────────────    ──────────────────────────────────
EMBRYONIC       0       execution                                 Always (immediate)
FETAL           1       + evaluation                              Always allowed
BORN            2       + breathing                               100 consecutive traces with eval
CHILDHOOD       3       + archive, penumbra                       1 successful dissolution
ADOLESCENCE     4       + optimization (stub)                     500 archived traces
ADULT           5       + autonomy (stub)                         disagreement_correlation >= 0.6
```

Transitions are irreversible. The system cannot regress from CHILDHOOD to BORN. This prevents the pathology where a system oscillates between autonomous and supervised modes.

The transition conditions are intentionally simple thresholds. They exist to enforce a minimum evidence bar: the system must demonstrate that evaluation works (FETAL), that the breathing cycle can dissolve stale combinators (CHILDHOOD), and that its self-evaluation correlates with actual quality (ADULT). More sophisticated transition detectors are a research direction.

The `DevelopmentalController` checks transitions via `check_transition(metrics, timestamp)`. The `ContinualLearningSystem` orchestrator calls this after every run, passing current `DevelopmentalMetrics`.

---

## Contrapuntal Evaluation

The evaluation architecture uses three independent evaluators ("voices") that assess each trace from orthogonal perspectives. The key insight from the dream session: no single evaluator can detect its own blind spots (Godel's incompleteness). Three individually-incomplete detectors can catch failures that any single one misses, because each voice's blind spot is covered by the other two.

### The three voices

| Evaluator | What it measures | Signal type | How it works |
|-----------|-----------------|-------------|--------------|
| `InformationTheoreticEval` | Information preservation through the tree | `information_theoretic` | Compares zlib-compressed size of parent output vs sum of children outputs. Ratio close to 1.0 = information preserved. Low ratio = information lost during reduction. |
| `BoundaryDetectionEval` | Self-referential loops (echoing) | `boundary` | Computes Jaccard word similarity between parent output and each child output. High similarity (>0.95) = the parent is just copying a child, not synthesizing. Score = 1 - max_echo_ratio. |
| `GoodhartResistantEval` | Perturbation robustness | `goodhart_resistant` | Inserts irrelevant distractor sentences into leaf prompts, re-runs the LLM call, and compares responses. If the response changes significantly on irrelevant perturbation, the leaf is fragile. Requires a `rerun_fn` (adds LLM calls). |

### Disagreement as signal

The `compute_disagreement()` function in `rsocas/evaluation/disagreement.py` is a pure function. Given a tuple of `EvalResult` objects, it computes:

- **magnitude**: max pairwise absolute difference in scores across all evaluator pairs.
- **pairwise**: dict of `"typeA_vs_typeB"` keys mapping to the pairwise score difference.
- **per_node**: for each node_id, the variance of that node's score across evaluators.
- **outlier_voice**: which evaluator's score differs most from the mean.
- **should_surface**: `True` if magnitude >= threshold (default 0.3).

The per-node disagreement is important: it tells you exactly which node in the tree the evaluators disagree about, enabling surgical diagnosis rather than whole-pipeline guessing.

### Phase 0 validation results

The Phase 0 benchmark (`rsocas/evaluation/benchmark.py`) tested whether evaluator disagreement correlates with actual quality failure (1 - F1 score) on a Sequential Needle-in-a-Haystack (SNIAH) dataset with distribution shifts.

```
Metric                   Value       Gate Threshold
─────────────────────    ─────────   ──────────────
Spearman rho             0.7254      >= 0.4
Spearman p-value         0.007588    < 0.05
Precision@5              0.8         >= 0.7
Precision@10             0.6         --
Samples                  12          --
Failures (F1 < 0.5)      6           --
GATE                     PASSED      --

Per-evaluator correlation with failure:
  information_theoretic   0.265
  boundary               -0.7413
  goodhart_resistant      0.8569
```

The composite disagreement (rho=0.7254) far exceeds either information-theoretic (0.265) or boundary (-0.7413) alone. The goodhart_resistant evaluator has the highest individual correlation (0.8569). The boundary evaluator has a negative correlation, meaning high boundary scores are associated with failure -- this is because boundary detection measures echo/copying, and some failure modes involve the model copying the prompt rather than answering.

The negative boundary correlation is not a bug. The three voices disagree precisely because they measure different things. That disagreement IS the signal.

---

## The Breathing Cycle

The breathing cycle governs the tempo of combinator crystallization and dissolution. It has two phases:

- **Systole** (contraction): crystallize fluid patterns into versioned, type-checked combinators.
- **Diastole** (expansion): dissolve stale combinators, scatter penumbra variants, re-explore.

### Tempo control

The `PIDTempoController` (`rsocas/breathing/tempo.py`) uses a PID controller anchored to human feedback frequency:

```
setpoint = human_freq * target_ratio
error = setpoint - current_system_freq
output = kp * error + ki * integral + kd * derivative
breathing_rate = max(0.01, system_freq + output)
```

When humans intervene often, the setpoint rises, but system frequency is high too. When humans go quiet, the setpoint drops and the system speeds up crystallization. The human's own behavior IS the tempo controller -- an external signal that avoids Godelian self-reference.

Default PID gains: `kp=0.5, ki=0.1, kd=0.05`. Target ratio: 2.0 (system breathes 2x faster than human feedback). These are defaults, not empirically tuned.

### Annealing schedule

The `AnnealingSchedule` (`rsocas/breathing/annealing.py`) manages temperature for combinator crystallization:

- High temperature = keep things fluid, allow exploration.
- Low temperature = crystallize stable patterns.
- Phase boundary detection via second derivative of temperature trajectory.
- At phase boundaries, cooling rate is reduced (`cooling_rate^0.5` instead of `cooling_rate`), preserving interesting structures near phase transitions.
- Controlled reheat triggered by high disagreement or human feedback.

The entropy budget per combinator is `temperature * (1 + staleness)` -- stale combinators get more exploration budget.

### Interference pattern

The `InterferencePattern` (`rsocas/breathing/interference.py`) determines when to surface for human feedback by modeling constructive/destructive interference between human and system frequencies:

```
beat_frequency = |human_freq - system_freq|
amplitude = 1.0 / (1.0 + beat_frequency)
```

Surface for human feedback when ALL of:
1. Disagreement says to surface (`should_surface == True`)
2. Interference is constructive (amplitude > 0.5)
3. Enough time since last surface (> `min_interval`)

The system surfaces at moments of constructive interference -- when small human signals produce maximum reorganization.

### Breathing crystallizer

The `BreathingCrystallizer` (`rsocas/breathing/breathing_crystallizer.py`) wires tempo, annealing, and the combinator lifecycle together. One `tick()` represents one heartbeat:

1. Record system event in tempo controller.
2. Cool the annealing schedule.
3. If disagreement magnitude >= `reheat_threshold` (default 0.5): reheat by `magnitude * 0.5`.
4. Check for TTL-expired combinators. Dissolve any that have expired.
5. If tempo says crystallize AND temperature < 0.5: emit a "cooled" event (ready for crystallization).

---

## Combinator Lifecycle

Combinators (the operations that compose LLM outputs: SPLIT, MAP, REDUCE, FILTER) follow a lifecycle with four states:

```
    ┌──────────┐     crystallize()     ┌──────────────┐     dissolve()     ┌────────────┐     expire()     ┌─────────┐
    │  fluid   │ ──────────────────→   │ crystallized  │ ──────────────→   │ dissolving  │ ──────────────→ │ expired │
    └──────────┘                       └──────────────┘                    └────────────┘                  └─────────┘
```

State transitions are enforced by `Crystallizer._assert_transition()` (`rsocas/combinators/crystallizer.py`, line 194). Invalid transitions raise `ValueError`.

### VersionedCombinator

Each combinator carries (`rsocas/contracts/combinators.py`):

- **version_id**: UUID
- **code_hash**: SHA-256 of the callable's representation
- **status**: fluid | crystallized | dissolving | expired
- **created_at / expires_at**: timestamps (TTL-based expiry, default 86400s = 24h)
- **validation**: `ValidationSnapshot` -- the distribution it was validated on (task types, input size range, n_samples, mean_score, score_std)
- **repairs**: tuple of `RepairRecord` objects documenting the combinator's change history

### Penumbra variants

The `PenumbraStore` (`rsocas/combinators/penumbra.py`) maintains near-miss variants around each crystallized combinator -- the "immune memory" of the system. When the input distribution shifts, candidates from the penumbra can be rapidly re-selected without full re-optimization.

Variants are ranked by relevance to the current distribution: `relevance = 1.0 / (1.0 + |mean_score_diff| + |std_diff|)`.

Pruning keeps at most 10 variants per parent (configurable).

### Staleness detection

Staleness is computed as: `|stored.mean_score - current.mean_score| / max(stored.score_std, 0.01)`. Values > 1.0 indicate significant distribution drift. The `DistributionTracker` (`rsocas/archive/distribution_tracker.py`) computes current `ValidationSnapshot` from recent archived traces.

### Version stamps on traces

When `patch_lambda_rlm_full()` is used, every `TreeTrace` can carry `combinator_versions` -- a dict mapping combinator names to version_ids. This links each execution to the exact combinator versions that produced it, enabling bisection when performance degrades.

---

## Contract System

All inter-module communication goes through three frozen dataclass contracts in `rsocas/contracts/`:

### Contract 1: TreeTrace (`contracts/traces.py`)

The universal data bus. Every component reads or writes `TreeTrace`. The structure:

- `TreeTrace` -- the complete execution record (trace_id, task_type, k, depth, tau, cost, nodes, leaves, output, timestamps, token counts)
- `NodeTrace` -- one node in the execution tree (id, depth, position, combinator type, input_size, output, children, llm_calls, latency)
- `LeafTrace` -- one LLM call at a leaf node (prompt, response, tokens_in, tokens_out, model, confidence)

All three are `frozen=True` dataclasses. Immutability is enforced at the type level.

### Contract 2: Evaluation (`contracts/evaluation.py`)

- `EvalResult` -- one evaluator's assessment (score, confidence, signal_type, per_node_scores dict, explanation)
- `DisagreementSignal` -- composite disagreement across evaluators (magnitude, pairwise diffs, per_node variance, outlier_voice, should_surface)
- `Evaluator` -- Protocol defining the evaluator interface (`signal_type` property + `evaluate()` method)

### Contract 3: Combinator Lifecycle (`contracts/combinators.py`)

- `VersionedCombinator` -- a combinator with version metadata
- `ValidationSnapshot` -- the distribution a combinator was validated on
- `RepairRecord` -- one change event in a combinator's history
- `CombinatorStore` -- Protocol for combinator persistence
- `TempoController` -- Protocol for breathing tempo

### Why frozen protocols matter

Every contract dataclass is `frozen=True`. This means:
1. No mutation bugs -- you cannot accidentally modify a trace or evaluation result.
2. Thread safety -- frozen dataclasses are safe to pass between threads without locks (the `TraceCollector` is thread-safe with its own lock, but traces themselves need none).
3. Hash stability -- frozen dataclasses can be used as dict keys or set members.
4. Audit trail -- since objects cannot be modified after creation, every trace and evaluation result is a permanent record.

Modules communicate only through these contracts. The `ContinualLearningSystem` orchestrator (`rsocas/development/orchestrator.py`) receives `TreeTrace` objects and returns `RunResult` objects -- it never imports Lambda-RLM. The GEPA adapter receives `TreeTrace` and `DisagreementSignal` objects -- it never imports GEPA. This decoupling means each module can be tested, replaced, or extended independently.
