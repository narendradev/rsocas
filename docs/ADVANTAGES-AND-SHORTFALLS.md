# RSOCAS: Advantages and Shortfalls

An honest assessment of what RSOCAS does well, where it falls short, when not to use it, and what remains unresolved.

---

## Advantages

### Formal guarantees from Lambda-RLM

Lambda-RLM provides closed-form cost estimates and accuracy bounds BEFORE execution. No other agentic system can answer "how much will this cost?" and "how accurate will this be?" without spending tokens. The cost model:

```
k* = ceil(sqrt(n * c_in / c_compose))
C_hat = k*^d * C(tau*) + d * C_compose(k*) + C(500)
```

This means you can set hard budgets. If a query would cost more than $X, reject it before executing. The accuracy bound `A_total >= (a_leaf)^d * (a_compose)^d` gives a worst-case quality floor. RSOCAS preserves these guarantees by tracing execution without modifying the control flow.

### Contrapuntal evaluation validated at rho=0.7254

Three cheap evaluators (no LLM calls for two of them) detect failures better than any single expensive evaluator. The Phase 0 benchmark on SNIAH with Nemotron-3-Super:

```
Composite disagreement correlation with failure:  rho = 0.7254  (p = 0.007588)
Best individual evaluator (goodhart_resistant):   rho = 0.8569
Worst individual evaluator (boundary):            rho = -0.7413
Precision@5 (top-5 by disagreement are failures): 0.8
```

The composite signal outperforms two of three individual evaluators. The boundary evaluator has negative correlation (its high scores associate with failure), but this is exactly the point -- the three voices disagree in ways that capture different failure modes. The disagreement itself is more informative than any single score.

### Sample efficiency from GEPA (78x over RL)

GEPA achieves with 243-1179 rollouts what GRPO needs 24,000 for. Combined with RSOCAS's tree-structured traces (which provide per-node credit assignment instead of whole-pipeline guessing), this means the system can improve meaningfully from tens of executions, not thousands.

Cross-model transfer adds further efficiency: prompts optimized on Qwen3-8B gain +9% on GPT-4.1 Mini. Optimization work on one model transfers to others.

### Model-agnostic

RSOCAS evaluates `TreeTrace` objects, not model outputs. Any system that produces a tree of LLM calls can be evaluated. The DSPy adapter further decouples leaf nodes from specific models -- swap the model behind the DSPy module without changing the evaluation pipeline.

### Self-improving

Every execution produces a `TreeTrace`. Every trace is evaluated. Failed traces (high disagreement) are archived with per-node diagnostic data. That data feeds GEPA optimization. Improved prompts produce better traces. Better traces refine the optimization signal. This is a genuine improvement loop, not a static pipeline.

### The human is the thermostat, not the bottleneck

The interference pattern logic surfaces for human feedback at moments of constructive interference -- when human and system frequencies are in sync and small human signals produce maximum reorganization. Between these moments, the system operates autonomously. The human is not reviewing every output (bottleneck) and not absent entirely (unchecked drift). The human sets the tempo by their own behavior.

### Developmental stages prevent premature autonomy

The system cannot reach ADULT stage without demonstrating that its self-evaluation correlates with actual quality (rho >= 0.6). It cannot reach CHILDHOOD without successfully dissolving at least one stale combinator. Each capability is earned, not configured. This builds trust incrementally: you can deploy at FETAL stage (evaluation only, no autonomous action) and let the system prove itself before enabling breathing and optimization.

### Code-space search discovers innovations invisible to prompt optimization

Meta-Harness's proposer agent reads all prior candidates' source code, traces, and scores. It discovers structural innovations (new combinator types, new reduction strategies) that prompt optimization cannot reach. RSOCAS's `MetaHarnessBridge` validates these discoveries (syntax, bounded execution, callability) before promoting them to the combinator registry.

### Traces contain more signal than summaries

Both GEPA and Meta-Harness proved this empirically. GEPA: execution traces as "text gradients" enable diagnostic optimization rather than blind search. Meta-Harness: raw traces (50.0% median) beat LLM summaries (34.9% median). RSOCAS's tree-structured traces go further -- per-node credit assignment means the optimization can target the specific node that failed, not just the overall pipeline.

---

## Shortfalls

### Phase 0 validated on only 12 samples

The rho=0.7254 result comes from 12 valid samples (4 base SNIAH questions x 3 variants each). This is statistically significant (p=0.007588) but not robust. A larger-scale validation (100+ samples, multiple task types, multiple models) is needed before claiming the contrapuntal evaluation architecture is production-ready.

### GoodhartResistantEval requires re-running leaf calls

The perturbation robustness evaluator perturbs each leaf prompt and re-runs the LLM call. For a trace with N leaf nodes, this adds N additional LLM calls per evaluation. On a tree with depth=2 and k=3, that is 9 extra calls. Without a `rerun_fn`, this evaluator returns a neutral score (0.5, confidence=0.0), reducing the three-voice system to effectively two voices.

The per-evaluator correlation data shows this evaluator has the highest individual correlation with failure (rho=0.8569). Disabling it meaningfully weakens the system.

### Shallow trees give less signal

At depth=0 (single LLM call, no decomposition), the information-theoretic evaluator falls back to leaf response density heuristics, and the boundary evaluator compares response vs prompt similarity instead of parent vs child. Both fallbacks are weaker than the tree-based analysis. The system works best on genuinely long-context tasks that require decomposition (depth >= 1).

### GEPA and DSPy adapters are untested with real implementations

The `TreeTraceGEPAAdapter` produces GEPA's expected format (reflective datasets with Inputs/Generated Outputs/Feedback), but it has only been tested against the contract interface. No test passes data through an actual GEPA optimizer.

Similarly, the `DSPyLeafRegistry` creates `LeafModuleSpec` objects and can construct `dspy.Predict` instances, but `optimize()` has not been tested against a real DSPy optimizer because DSPy is not a dependency.

Both adapters will work with real implementations if those implementations consume the documented formats. But "will work" and "has been tested" are different things.

### Meta-Harness bridge validates syntax/structure but not semantics

The `validate_candidate()` method checks that candidate code compiles, has no unbounded loops, has no recursive self-calls, and defines a callable. It does NOT check:
- Whether the callable produces correct results.
- Whether the callable handles edge cases (empty input, very long input).
- Whether the callable's type signature matches its actual behavior.
- Whether the callable is efficient.

Semantic validation requires running the candidate on test data, which is the caller's responsibility.

### The breathing cycle's PID gains are defaults, not empirically tuned

`kp=0.5, ki=0.1, kd=0.05` were chosen as reasonable starting points. They have not been tuned on any real workload. The `target_ratio=2.0` (system breathes 2x faster than human feedback) is also an assumption.

In practice, these defaults produce stable behavior in tests, but optimal gains depend on:
- Human review cadence (how often feedback arrives).
- System throughput (how many traces per hour).
- Distribution volatility (how fast the input distribution changes).

### Developmental stage transitions use simple thresholds

The transition from BORN to CHILDHOOD requires only 1 successful dissolution. The transition to ADULT requires rho >= 0.6. These thresholds are hardcoded and do not account for:
- Whether the dissolution was meaningful or trivial.
- Whether the correlation was computed on a representative sample.
- Whether the system has been operating long enough for the metrics to be stable.

A more sophisticated transition detector would consider multiple metrics simultaneously, track trends, and require sustained performance rather than one-time threshold crossing.

### The archive grows unboundedly

`TraceArchive` stores every trace, evaluation, and disagreement signal in SQLite. There is no compaction strategy, no automatic pruning of old data, and no mechanism for moving cold data to cheaper storage. Over months of continuous operation, the database will grow large.

The FTS5 index on `final_output` adds further overhead. For a system processing 1000 traces per day with average output length of 500 characters, expect roughly 50-100 MB per day of growth.

### The sheaf-theoretic formulation is not implemented

The thinking documents (`000-META-THINKING.md`, `003-UNTHINKABLE-DIRECTIONS.md`) describe a sheaf-theoretic framework where different input clusters have locally-optimal configurations with consistency conditions on overlaps. This is not implemented. RSOCAS uses the same evaluation and breathing configuration for all inputs regardless of type. A router that selects per-input-cluster configurations remains theoretical.

### No production hardening

The current implementation has:
- No async support. All evaluations and database operations are synchronous.
- No batching. Each trace is evaluated individually.
- No rate limiting. Evaluators that use `rerun_fn` will make as many LLM calls as there are leaves.
- No error recovery. If an evaluator raises an exception, the entire `run()` call fails.
- No connection pooling on the SQLite database.
- No logging. The system uses no logging framework.

---

## When NOT to Use RSOCAS

### Single-shot short prompts

If you send a 100-token prompt and get a 50-token response with no decomposition, the evaluators have almost nothing to work with. The information-theoretic evaluator computes leaf density heuristics. The boundary evaluator checks prompt-response similarity. Both are weak signals on short texts. You would be better served by a simple metric (exact match, F1, semantic similarity with a reference).

### Latency-critical paths

Evaluation adds overhead:
- `InformationTheoreticEval`: zlib compression of all node outputs. Negligible for small traces, milliseconds for large ones.
- `BoundaryDetectionEval`: Jaccard similarity between all parent-child pairs. O(nodes * max_children).
- `GoodhartResistantEval`: N additional LLM calls (one per leaf). This dominates.
- `compute_disagreement()`: O(evaluators^2 * nodes). Negligible.
- Archive storage: one SQLite write per trace.

If your latency budget is under 100ms, skip RSOCAS. If you can tolerate 500ms+ overhead (excluding Goodhart re-runs), the first two evaluators are feasible. If you can tolerate multiple additional LLM calls, use all three.

### When you have abundant human reviewers

RSOCAS's value is in reducing human review load. If every output is reviewed by a human anyway, the evaluators are redundant. The breathing cycle, interference pattern, and developmental stages all assume that human attention is scarce and must be allocated wisely. If attention is abundant, use it directly.

### Tasks where accuracy is binary and easily measurable

If your task has a clear right/wrong answer that you can check programmatically (e.g., math problems with numeric answers, code that passes unit tests), a simple correctness check is cheaper and more reliable than contrapuntal evaluation. RSOCAS is designed for tasks where quality is nuanced and failure is subtle -- summarization, open-ended QA, document analysis, extraction from ambiguous sources.

---

## Research Gaps

Open questions from the thinking documents that the current implementation does not answer.

### Does the self-improving evaluation loop converge?

Direction 2 from `003-UNTHINKABLE-DIRECTIONS.md`: if GEPA optimizes prompts, Meta-Harness optimizes code, and the evaluation rubric is itself optimized, does this loop reach a fixed point? Under what conditions?

The Y combinator interpretation suggests convergence if the composition of optimizers is contractive (each iteration produces diminishing changes). The Nash equilibrium interpretation warns that the fixed point might be degenerate (each optimizer gaming the others). Neither has been tested.

### Can three individually-incomplete detectors compose into adequacy?

The Phase 0 result (rho=0.7254, N=12) partially answers this. The boundary evaluator individually has negative correlation but contributes to the composite signal. Larger-scale validation is needed to determine:
- Does the composite signal remain strong on N=100+?
- Does it generalize across task types (not just QA)?
- Does it generalize across models (not just Nemotron-3-Super)?

### What is the right combinator granularity for Meta-Harness discovery?

The `MetaHarnessBridge` validates individual functions. But Lambda-RLM's combinators operate at different granularities: `_Split` is a pure text operation, `_Reduce` involves an LLM call, `_FilterRelevant` involves an LLM call plus a decision. Should discovered combinators be at the pure-function level? The LLM-call level? The multi-step level?

Too fine-grained: uninterpretable micro-operations. Too coarse: no composability. The right granularity might emerge from the optimization process, but this has not been studied.

### Does the breathing cycle's tempo actually improve long-term performance?

The PID controller anchored to human feedback frequency is theoretically grounded (external signal avoids Godelian self-reference). But no experiment has compared:
- PID-controlled breathing vs. fixed crystallization schedule.
- PID-controlled breathing vs. no breathing (evaluate but never crystallize/dissolve).
- Different PID gains on the same workload.

Until these comparisons exist, the breathing cycle is a plausible mechanism, not a proven one.

### Can the GEPA adapter's per-node credit assignment work in practice?

GEPA was designed for flat traces. The `TreeTraceGEPAAdapter` provides per-node surgical context (sibling outputs, parent combinator type, position in tree). This is a richer signal than flat traces. But:
- Does GEPA's reflection mechanism handle tree-structured feedback?
- Does per-node targeting produce better prompt mutations than whole-pipeline targeting?
- Are there failure modes where tree structure misleads the optimizer?

These require experiments with a real GEPA installation.

### Does distribution shift detection work in practice?

The `DistributionTracker` computes `ValidationSnapshot` from recent traces and the `Crystallizer` computes staleness as a z-score. But:
- What staleness threshold should trigger dissolution? (Currently caller's choice.)
- How quickly does the tracker detect gradual drift vs. sudden shift?
- Does the 1-hour default window capture the right timescale for distribution change?

### What is the right compaction strategy for the archive?

The archive grows unboundedly. Possible strategies not yet implemented:
- Time-based: drop traces older than N days.
- Importance-based: keep high-disagreement traces, drop low-disagreement ones.
- Sampling-based: keep a representative sample from each time window.
- Summarization-based: replace old traces with statistical summaries.

Each strategy trades off differently against GEPA optimization quality (which benefits from diverse failure data), distribution tracking accuracy (which benefits from continuous coverage), and storage cost.
