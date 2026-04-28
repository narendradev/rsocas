# 005 — Architecture Skeleton: The System Shape

**Date**: 2026-04-28

---

## Not a Design Doc. A Thinking Scaffold.

This isn't the final architecture. It's the shape I see when I close my eyes and think about all five frameworks at once. It will change. But writing it down forces precision.

---

## The Stack (Bottom to Top)

```
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 6: TEMPORAL EVOLUTION                                     │
│ Online adaptation, personalization, difficulty frontier          │
│ State: grows with each invocation                               │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 5: META-OPTIMIZATION (Meta-Harness outer loop)            │
│ Code-space search over the entire system                        │
│ Input: archive of all prior system states + traces              │
│ Output: structural innovations (new combinators, routing, code) │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 4: EVALUATION (LangProBe + self-evaluation)               │
│ Multi-objective Pareto evaluation with anchor points            │
│ Dimensions: accuracy, latency, cost, robustness, diversity      │
│ Includes adversarial self-generated benchmarks                  │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 3: OPTIMIZATION CHAINING (BetterTogether)                 │
│ Strategy program (not string): "gepa -> finetune -> gepa"      │
│ The strategy itself is a DSPy program, optimizable              │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 2: PROMPT OPTIMIZATION (GEPA)                             │
│ Reflective evolution with tree-structured credit assignment     │
│ Pareto-based candidate selection (quality-diversity)            │
│ Input: tree-structured traces from Layer 1                      │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 1: PROGRAMMATIC COMPOSITION (DSPy)                        │
│ Typed modules with signatures                                   │
│ Model-agnostic compilation                                      │
│ Each leaf node in Lambda-RLM tree = DSPy Module                 │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 0: FORMAL EXECUTION ENGINE (Lambda-RLM)                   │
│ Typed combinators: SPLIT, MAP, FILTER, REDUCE, CROSS           │
│ Deterministic tree with cost/accuracy bounds                    │
│ Leaf nodes invoke DSPy Modules from Layer 1                     │
│ Guarantees: termination, cost bound, accuracy bound             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Information Flow

### Downward (Configuration)
```
Meta-Harness proposes system changes
  → BetterTogether chains optimization passes
    → GEPA optimizes prompts within DSPy modules
      → DSPy compiles optimized modules
        → Lambda-RLM plans and executes the tree
```

### Upward (Traces)
```
Lambda-RLM produces tree-structured execution traces
  → DSPy records module-level traces (signatures, demos used, outputs)
    → GEPA receives (tree_trace, score, text_feedback)
      → BetterTogether tracks per-stage candidate programs
        → Meta-Harness archives everything (code + traces + scores)
```

### Lateral (Cross-layer)
```
Meta-Harness ←→ Lambda-RLM: verified innovation pipeline
  (propose combinator → type-check → accept/reject with error message)

GEPA ←→ Lambda-RLM: tree-structured credit assignment
  (per-node diagnosis instead of whole-pipeline reflection)

Evaluation ←→ GEPA: optimizable evaluator
  (rubric programs are DSPy modules, optimized by GEPA)

Temporal ←→ All: online adaptation
  (each real invocation feeds all layers)
```

---

## The Connective Tissue (The Parts Nobody Has Built)

### Connector A: TreeTrace → GEPA Adapter

Lambda-RLM's tree trace has this structure:
```python
TreeTrace = {
    task_type: str,
    tree_params: {k: int, d: int, tau: int},
    nodes: [
        {
            id: str,
            depth: int,
            combinator: str,  # SPLIT, MAP, REDUCE, etc.
            input_size: int,
            output: str,
            children: [node_id],
            llm_calls: int,
            latency_ms: int,
            score: float | None  # if evaluable
        }
    ],
    leaf_traces: [
        {
            node_id: str,
            prompt: str,
            response: str,
            tokens_used: int,
            confidence: float | None
        }
    ],
    final_output: str,
    final_score: float,
    feedback: str
}
```

GEPA expects a flat trace. The adapter must:
1. Identify the failing node(s) — nodes where output score is low or confidence is low
2. Extract the relevant subtree (failing node + its ancestors + its siblings for context)
3. Format as GEPA's expected trace format but with tree context: "This is leaf node 1.2 at depth 2, processing chunk 2 of 8. Parent node 1 used SPLIT to divide a 65K document into 8 chunks. Sibling leaves 1.1, 1.3-1.8 produced [summaries]. This leaf received [chunk text] and produced [output]. Expected: [ground truth]. Score: 0.0."
4. GEPA's reflection LM now has surgical context: exactly which leaf failed, what it received, what its siblings produced, and how its output would have been composed.

### Connector B: Meta-Harness → Lambda-RLM Verifier

Meta-Harness proposes a new combinator as Python code:
```python
def VERIFY_AND_RETRY(extractor, verifier, max_retries=2):
    """Extract, verify, retry if verification fails."""
    for attempt in range(max_retries + 1):
        result = extractor(chunk)
        if verifier(result):
            return result
    return result  # last attempt even if unverified
```

Lambda-RLM's type checker must verify:
1. **Termination**: bounded loop (max_retries is finite) ✓
2. **Cost bound**: at most (max_retries + 1) * (C_extract + C_verify) ✓, computable
3. **Type signature**: (Chunk → Result, Result → Bool) → Chunk → Result ✓
4. **Composability**: Can be used anywhere EXTRACT is used (compatible type) ✓

If verification passes, the combinator enters the library with computed cost constants.
If verification fails, the error message guides Meta-Harness:
```
TYPE ERROR: Combinator uses unbounded external state via `verifier`.
HINT: verifier must be a pure function (no side effects, no network calls).
SUGGESTION: Accept verifier as a lambda with captured constants only.
```

### Connector C: DSPy Module ↔ Lambda-RLM Leaf

A Lambda-RLM leaf node currently calls the LLM directly:
```python
response = llm_query(prompt_template.format(chunk=chunk, question=question))
```

Replace with a DSPy Module:
```python
class LeafQA(dspy.Module):
    def __init__(self):
        self.answer = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, context, question):
        return self.answer(context=context, question=question)
```

This gives us:
- GEPA can optimize the leaf's instruction/demos via `dspy.GEPA`
- BetterTogether can fine-tune the leaf model via `BootstrapFinetune`
- The module carries its optimized state (instructions, demos, model)
- Different leaves can use different modules (heterogeneous tree)

### Connector D: Strategy Program

Replace BetterTogether's strategy string with a program:
```python
class OptimizationStrategy(dspy.Module):
    def __init__(self):
        self.should_continue = dspy.Predict("current_score, improvement_rate, budget_remaining -> continue: bool")
        self.select_optimizer = dspy.Predict("failure_analysis, available_optimizers -> optimizer_name, target_component")
    
    def forward(self, system, eval_set, budget):
        history = []
        while budget > 0:
            score = evaluate(system, eval_set)
            improvement = score - history[-1].score if history else 0
            
            if not self.should_continue(score, improvement, budget).continue:
                break
            
            analysis = analyze_failures(system, eval_set)
            choice = self.select_optimizer(analysis, ["gepa", "finetune", "meta_harness"])
            
            system = run_optimizer(choice.optimizer_name, system, eval_set, target=choice.target_component)
            budget -= 1
            history.append({"score": score, "optimizer": choice.optimizer_name, "target": choice.target_component})
        
        return system
```

This strategy program is itself a DSPy program — optimizable by GEPA. Meta-circularity achieved.

---

## The Invariants (What Must Always Be True)

1. **Every computation terminates.** Lambda-RLM's formal guarantees must hold at every layer. If a combinator is added that could loop, the type checker rejects it.

2. **Every execution produces a trace.** No silent failures, no unrecorded computation. The trace is the source of truth for all optimization.

3. **Every optimization step is reversible.** If a prompt change, code change, or structural change degrades performance, the previous version is restored. The archive provides the rollback points.

4. **The evaluation anchors don't move.** Human-judged anchor evaluations remain fixed across optimization iterations. The system can't game these because it can't modify them.

5. **Cost bounds are known before execution.** Lambda-RLM's cost model extends to the full tree. Before any inference call, the system can answer "how much will this cost?" (in tokens, latency, and dollars).

6. **Model substitution is transparent.** Swapping the underlying LLM (Haiku → Sonnet → Opus, or Nemotron → GPT → Gemini) requires no code changes, only re-optimization. DSPy's model-agnostic API ensures this.

---

## What's Missing From This Skeleton

1. **The sheaf structure** — I described it in 003 but haven't formalized the topology on the input space, the local sections (system configurations), or the gluing conditions. This is the mathematical work needed to turn "different inputs need different systems" into a rigorous framework.

2. **The contraction proof** — I claimed the optimization loop converges under contraction conditions but haven't specified those conditions. Need to define a metric on the space of system configurations and show that each optimization step is contractive in that metric.

3. **The adversarial task generator** — Described the anti-benchmark concept but haven't designed the generator. Key question: how do you generate tasks that are GENUINELY hard (require capability) vs. SPURIOUSLY hard (require knowledge of a specific fact or trick)?

4. **The temporal state management** — Online adaptation requires storing and efficiently querying a growing archive of traces. Need a specific data structure: probably a tree-indexed vector database where each trace is embedded and indexed by (task_type, tree_depth, node_position, failure_mode).

5. **The heterogeneous model assignment** — Described in 002 (heterogeneous compute trees) but haven't designed the assignment algorithm. The cost tensor indexed by (node_position, model, task_type) needs to be profiled for each model.

---

*This skeleton will grow. Each connector, invariant, and missing piece is a research direction. The system isn't something you build in one pass — it's something that grows, much like the system itself is designed to grow.*
