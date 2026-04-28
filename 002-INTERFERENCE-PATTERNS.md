# 002 — Interference Patterns: Where Frameworks Collide

**Date**: 2026-04-28

---

## What This Document Is

When two waves meet, they don't just add — they interfere. Constructive interference amplifies. Destructive interference cancels. The interesting physics is at the boundaries.

These frameworks have boundaries. This document maps them.

---

## Collision 1: GEPA Traces × Lambda-RLM Trees

### The Boundary

GEPA expects a flat execution trace: "Module A received X, produced Y, then Module B received Y, produced Z, final output Z, score 0.7."

Lambda-RLM produces a tree trace:
```
ROOT (task=QA, n=65000, k*=8, d=2)
├── NODE_1 (SPLIT, chunk 0-8125)
│   ├── LEAF_1.1 (QA prompt, input=8125 chars, output="Paris", score=1.0)
│   ├── LEAF_1.2 (QA prompt, input=8125 chars, output="unknown", score=0.0)
│   └── ...
│   └── REDUCE_1 (MAJORITY_VOTE, result="Paris")
├── NODE_2 (SPLIT, chunk 8125-16250)
│   └── ...
└── FINAL_REDUCE (SELECT_RELEVANT, inputs=8 answers, result="Paris", score=0.8)
```

### The Interference

**Constructive**: If GEPA can consume tree traces, it gains per-node credit assignment for free. It doesn't need to figure out that "the failure happened in the second chunk's third leaf" — the tree structure tells it explicitly. GEPA's reflection LM can be asked: "LEAF_1.2 received this text and said 'unknown'. The answer was in this chunk. Why did it miss it? Propose a better prompt."

This changes GEPA from a *system-level* optimizer to a *node-level* optimizer. The search space shrinks dramatically. Instead of searching over all possible prompt modifications for the whole pipeline, it searches over modifications to ONE leaf prompt, conditioned on a specific failure.

**Destructive**: Lambda-RLM's tree structure imposes constraints that GEPA might fight against. GEPA might discover that the best prompt for leaf nodes assumes context that the SPLIT combinator removed. The optimal prompt might be "Given this is part 3 of 8 of a larger document about X, answer..." But the combinator doesn't pass that context to leaves. GEPA would be optimizing within a box that the right answer lies outside of.

**Resolution**: Let GEPA optimize not just the leaf prompts but also the CONTEXT passed to each leaf. Lambda-RLM's SPLIT currently passes raw text chunks. GEPA could add a CONTEXT_HEADER combinator that prepends metadata: document title, chunk position, global topic. The combinator is typed and verified; the content of the header is GEPA-optimized text.

### What This Generates (New Idea)

**Typed Prompt Templates**: A hybrid where the STRUCTURE of the prompt (what sections exist, what type each section has) is formally specified (Lambda-RLM), but the CONTENT of each section is optimized text (GEPA). Like a typed template:

```
TypedPrompt<QA> = {
  system: String,           // GEPA-optimized
  context_header: String,   // GEPA-optimized, generated per-chunk
  chunk: RawText,           // from SPLIT combinator
  question: String,         // from user input
  output_format: Schema     // typed, verified
}
```

The type system ensures the prompt has the right structure. GEPA fills in the content. This is prompt engineering meets type engineering.

---

## Collision 2: Meta-Harness Code Search × Lambda-RLM Formal Guarantees

### The Boundary

Meta-Harness can propose ANY Python code. Lambda-RLM requires combinators to be typed, terminating, and formally verified.

### The Interference

**Constructive**: Meta-Harness discovers innovations that no human would design. But "innovation" sometimes means "clever hack that happens to work." Lambda-RLM's formal requirements act as a *regularizer* — the discovered code must respect structural constraints. This filters out overfitting-via-code (brittle if-chains, hardcoded patterns) while allowing genuine structural innovations.

Example: Meta-Harness might discover that for medical QA, a VERIFY step after each leaf call (check extracted answer against a known medical ontology) dramatically improves accuracy. This is a genuine innovation. It can be expressed as a new combinator: `VERIFY(extractor, ontology) -> verified_answer`. The combinator is typed (takes an extractor function and an ontology, returns a verified answer), terminating (one additional call), and verifiable. Lambda-RLM accepts it.

Counter-example: Meta-Harness might discover that adding `if "patient" in chunk: answer = "consult a doctor"` improves scores on the medical eval set. This is overfitting. Lambda-RLM rejects it — it's not a typed combinator, it has no formal properties, and it doesn't compose.

**Destructive**: Lambda-RLM's formal requirements might be too restrictive for some genuine innovations. Meta-Harness might discover that the best harness uses *stateful* processing (maintaining a running context across chunks, not just independent parallel processing). But Lambda-RLM's combinators are pure functions — no state. The formalism prevents a real improvement.

**Resolution**: Extend Lambda-RLM's type system to include *linear types* (from linear logic). A linear type ensures a value is used exactly once — no copying, no discarding. This enables controlled state: `STATEFUL_MAP(init_state, f: (State, Chunk) -> (State, Result), chunks) -> (FinalState, [Result])`. The linear type on State guarantees that state is threaded sequentially (no data races, no stale reads) while still being formally verified.

### What This Generates (New Idea)

**Verified Innovation Pipeline**: Meta-Harness proposes → Lambda-RLM type-checks → if verified, add to combinator library → if rejected, Meta-Harness receives type error as diagnostic feedback → proposes a compliant version.

This creates a *dialogue* between creative search and formal verification. The type checker isn't just a filter — its error messages guide the search. "Cannot use mutable state in combinator — consider using STATEFUL_MAP with linear types." The proposer agent learns what the type system accepts, and its proposals get closer to verifiable over time.

---

## Collision 3: DSPy Model Agnosticism × Lambda-RLM Cost Models

### The Boundary

DSPy treats models as interchangeable behind a uniform API. Lambda-RLM's cost model assumes specific cost constants (tokens per second, cost per token, accuracy per chunk size) that are model-dependent.

### The Interference

**Constructive**: DSPy's model-agnostic API lets you swap models without changing pipeline code. Lambda-RLM's cost model tells you WHAT model to use WHERE. Combine them: the cost model profiles each available model's cost constants, then the planner selects the optimal model for each position in the tree.

```
Leaf nodes on easy chunks → Haiku 4.5 (fast, cheap, sufficient accuracy)
Leaf nodes on hard chunks → Sonnet 4.6 (higher accuracy needed)
Compose nodes → Opus 4.6 (synthesis requires deep reasoning)
Task classification → Haiku 4.5 (simple digit menu)
```

This is **model routing within a formally optimized tree**. The tree structure gives you natural routing points (each node is a decision). The cost model gives you the optimization criterion. DSPy gives you the swappable model API.

**Destructive**: Lambda-RLM's cost model is calibrated for a specific model. Swapping models mid-tree invalidates the cost predictions. The accuracy bound `A_total >= (a_leaf)^d * (a_compose)^d` assumes uniform leaf accuracy. Mixed models mean non-uniform accuracy, and the bound becomes: `A_total >= (prod a_leaf_i) * (prod a_compose_j)` — a product over all nodes, which is harder to compute and optimize.

**Resolution**: Generalize the cost model to a *cost tensor* indexed by (node_position, model, task_type). The planner optimizes over model assignments jointly with tree structure. This is a discrete optimization problem, but for small trees (d=2-3, k=4-8) it's tractable via exhaustive search or integer programming.

### What This Generates (New Idea)

**Heterogeneous Compute Trees**: A single computation tree where different nodes use different models, different prompts, and different combinators — all jointly optimized. The tree isn't just a decomposition strategy; it's a *resource allocation plan*.

This is reminiscent of heterogeneous computing in hardware (CPU + GPU + TPU), where different compute units handle different parts of the workload based on their strengths. Nobody has applied this idea to LLM-based computation trees.

---

## Collision 4: GEPA Pareto Selection × LangProBe Evaluation Grid

### The Boundary

GEPA maintains a Pareto front of candidates based on per-example performance. LangProBe evaluates across a 2000+ configuration grid on multiple metrics.

### The Interference

**Constructive**: GEPA's Pareto front is over *training examples*. LangProBe's grid is over *system configurations*. What if we maintained a Pareto front over BOTH dimensions simultaneously? A 2D Pareto surface where one axis is "which examples this configuration excels on" and the other is "which cost-quality tradeoff this configuration achieves."

This gives you a *lookup table*: given a new input, classify it (easy/medium/hard, domain, length), then select the Pareto-optimal configuration for that input class AND your current cost budget. Different inputs get different systems.

**Destructive**: The combinatorial explosion. If GEPA maintains N candidates and LangProBe evaluates M configurations, the joint space is N×M. For N=50, M=100, that's 5000 configurations to evaluate. Each evaluation requires running the full pipeline. The cost model helps (you can predict performance without running), but the search space is still vast.

**Resolution**: Hierarchical evaluation. First, use Lambda-RLM's cost model to prune configurations that exceed budget. Second, use GEPA's reflection to predict which configurations are likely to improve (without running them). Third, evaluate the top-K candidates on a small held-out set. This reduces the effective search space from N×M to ~K (where K << N×M).

### What This Generates (New Idea)

**Adaptive System Selection**: At inference time, the system doesn't run a fixed pipeline. It:
1. Analyzes the input (length, complexity, domain)
2. Looks up the Pareto-optimal configuration for this input class
3. Instantiates the appropriate tree structure, model assignment, and prompt set
4. Executes

This is like a JIT compiler that selects different optimization strategies based on the code being compiled. The "compilation" here is: selecting the right AI system configuration for each specific input.

---

## Collision 5: Meta-Harness Non-Markovian Memory × GEPA Reflection

### The Boundary

Meta-Harness's proposer reads all prior iterations (~10M tokens of history). GEPA's reflection LM reads the current iteration's trace (~1-5K tokens).

### The Interference

**Constructive**: Give GEPA access to Meta-Harness's archive. Instead of reflecting on just the current trace, the reflection LM can ask: "In iteration 7, a similar prompt change was tried and it failed because X. Let me try something different." This turns GEPA from a memoryless optimizer into a *historically-aware* optimizer.

This is the difference between a chess engine that evaluates positions independently (GEPA today) and one that remembers its analysis of similar positions (GEPA with history). The latter avoids repeating failed strategies and builds on successful ones.

**Destructive**: 10M tokens of history is too much for most LLMs' context windows. Even with 1M context, you need to select the relevant history. But selection requires knowing what's relevant, which requires understanding the current failure, which is what the reflection is supposed to figure out.

**Resolution**: Lambda-RLM's tree structure provides a natural indexing scheme for the history. Index past iterations by: (task_type, tree_depth, node_position, combinator, failure_mode). When reflecting on a current failure at node (depth=2, position=3, combinator=REDUCE, failure=low_accuracy), retrieve past iterations where the same node configuration had similar failures. This turns 10M tokens into ~10K of relevant history.

### What This Generates (New Idea)

**Optimization Memory**: A persistent, indexed archive of optimization attempts — what was tried, what worked, what failed, and why. This archive transcends individual optimization runs. It's a *knowledge base of how to optimize*. New tasks start not from scratch but from the closest prior task's best configuration.

This is meta-learning for optimizers. Not learning to learn (meta-learning for models). Learning to *optimize* (meta-learning for the optimization process itself). The archive IS the learned meta-optimizer.

---

## The Pattern Across All Collisions

Every collision has the same shape:

1. **Framework A produces structured information**
2. **Framework B consumes information but expects a different structure**
3. **The translation between structures IS the innovation**

The translations:
- Tree traces → flat traces (Lambda-RLM → GEPA): tree-structured credit assignment
- Free-form code → typed combinators (Meta-Harness → Lambda-RLM): verified innovation pipeline
- Model-agnostic API → model-specific costs (DSPy → Lambda-RLM): heterogeneous compute trees
- Per-example Pareto → per-config grid (GEPA → LangProBe): adaptive system selection
- Historical archive → focused reflection (Meta-Harness → GEPA): optimization memory

**The meta-pattern**: Information wants to flow between frameworks, but the interfaces don't match. The system we need to build is not the frameworks themselves — it's the *connective tissue* between them.

---

*This document maps boundaries. The next document (003) will explore what happens when we stop thinking of these as separate frameworks and start thinking of them as a single system.*
