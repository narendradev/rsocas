# 006 — Questions That Matter: The Research Agenda

**Date**: 2026-04-28

---

## Organizing Principle

Not all questions are equal. Some questions, if answered, unlock everything downstream. Others are interesting but peripheral. This document tries to identify the load-bearing questions — the ones where the whole architecture depends on the answer.

---

## Tier 1: If We Don't Answer These, Nothing Else Matters

### Q1: Does tree-structured credit assignment improve GEPA's sample efficiency?

**Why load-bearing**: The entire architecture depends on Lambda-RLM trees providing better optimization signal than flat traces. If per-node credit assignment doesn't help — if GEPA's reflection LM is already good enough at diagnosing flat traces — then the Lambda-RLM → GEPA connector adds complexity without benefit.

**How to test**: Implement Connector A (TreeTrace → GEPA Adapter). Run GEPA on a standard DSPy benchmark in two modes: (a) flat traces (current GEPA), (b) tree-structured traces (new). Measure: convergence speed (rollouts to reach target score), final score, and diversity of discovered prompt variants.

**Prediction**: Tree-structured traces reduce rollouts by 3-10x for tasks with clear subtask decomposition (QA, extraction, cross-reference). No improvement for tasks that don't decompose (classification, short-form generation).

**If the answer is NO**: Remove the Lambda-RLM → GEPA connector. Lambda-RLM provides execution guarantees but not optimization advantages. The architecture simplifies but loses the "structure improves optimization" thesis.

### Q2: Can Meta-Harness discover combinators that pass Lambda-RLM's type checker?

**Why load-bearing**: The verified innovation pipeline (Connector B) is the mechanism by which the system GROWS — acquires new capabilities. If Meta-Harness can only propose code that Lambda-RLM rejects, the two frameworks are incompatible rather than complementary.

**How to test**: Set up Meta-Harness with Lambda-RLM's type checker as a constraint. Run on a task where the current combinator library is insufficient (e.g., a task requiring stateful processing). Measure: (a) how many proposals pass the type checker, (b) whether the type checker's error messages help Meta-Harness propose better candidates, (c) whether verified combinators generalize better than unverified code.

**Prediction**: Initial acceptance rate ~5-10%. After 20 iterations of the dialogue (propose → reject with error → re-propose), acceptance rate rises to ~30-40% as the proposer learns the type system's constraints. Verified combinators generalize 2-3x better than unverified code on held-out tasks.

**If the answer is NO**: The type system is too restrictive. Options: (a) relax the type system (allow more programs but lose some guarantees), (b) abandon formal verification and use empirical testing only, (c) add an intermediate verification level ("probably safe" vs "provably safe").

### Q3: Does the optimization loop converge?

**Why load-bearing**: If the loop (GEPA → Meta-Harness → Evaluate → GEPA → ...) doesn't converge, the system oscillates: GEPA optimizes prompts for one code version, then Meta-Harness changes the code, then GEPA's prompts are wrong for the new code, then GEPA re-optimizes, then Meta-Harness changes code again... forever.

**How to test**: Implement a simple version of the loop with GEPA and one other optimizer (BootstrapFewShot, not Meta-Harness — simpler to debug). Run for 50 iterations on a standard benchmark. Plot system performance over iterations. Look for: (a) convergence (performance plateaus), (b) oscillation (performance cycles), (c) divergence (performance degrades).

**Prediction**: Convergence in 10-20 iterations for simple pipelines (2-3 modules). Oscillation risk for complex pipelines (5+ modules) unless GEPA's learning rate is decayed (smaller prompt changes in later iterations).

**If the answer is "it oscillates"**: Add damping. Each optimizer makes smaller changes over time. The learning rate schedule is itself a parameter of the strategy program (Connector D), optimizable by the meta-level.

---

## Tier 2: Important but Not Blocking

### Q4: How should the input space be partitioned for the sheaf structure?

**Options**: 
- By input length (short/medium/long) — simple but misses semantic structure
- By task type (QA/extraction/summary) — task-dependent, might not generalize
- By learned embedding clusters — adaptive but requires training data
- By Lambda-RLM's cost model (which tree structure is optimal) — elegant because the partition emerges from formal analysis

**My intuition**: Use Lambda-RLM's cost model. Two inputs are "similar" if they lead to the same optimal tree structure (same k*, d, tau*). This creates natural clusters where the same system configuration is optimal. The sheaf's local sections are system configurations indexed by tree parameters.

### Q5: What's the right granularity for new combinators?

**Too fine**: VERIFY_SINGLE_FACT(fact, ontology) — too specific, won't compose with other tasks
**Too coarse**: SMART_PROCESS(input) — too vague, no formal properties
**Just right**: VERIFY(checker, item) — general enough to compose, specific enough to verify

**Hypothesis**: The right granularity is when the combinator's type signature has 2-4 type parameters. Fewer = too specific. More = too general (the type system can't constrain enough).

### Q6: Can the evaluator be self-improving without Goodharting?

**The tension**: An evaluator optimized to correlate with human judgment will eventually find shortcuts (patterns that correlate with human judgment on the training set but not in general). This is Goodhart's Law applied to meta-evaluation.

**Possible safeguards**:
- Anchor points (fixed human evaluations) that can't be optimized
- Red-team evaluation (adversarial examples designed to distinguish genuine quality from metric gaming)
- Diverse evaluator ensemble (multiple rubric programs, agreement required)
- Periodic human audit of evaluator behavior (is it still measuring what we think?)

### Q7: How large can the trace archive grow before retrieval degrades?

**Calculation**: At 10M tokens per Meta-Harness iteration, after 100 iterations: 1B tokens. After 1000 iterations (spanning multiple tasks, multiple optimization runs): 10B tokens. Even with tree-structured indexing, embedding-based retrieval has O(log n) scaling at best, and the constant matters at 10B scale.

**Possible solution**: Forgetting. Not all traces are equally valuable. Old traces from superseded system configurations can be compressed (store the summary, not the raw trace). Recent traces and traces from still-active Pareto candidates are stored in full. This mimics human memory: recent and important memories are detailed, old memories are gist-level.

---

## Tier 3: Interesting Research Questions

### Q8: Is there a natural correspondence between combinator operations and cognitive operations?

SPLIT ↔ problem decomposition, MAP ↔ applying a strategy to each part, FILTER ↔ attention/relevance, REDUCE ↔ synthesis, CROSS ↔ comparison/analogy.

If this correspondence is real (not just suggestive), then advances in cognitive science could inform new combinators, and new combinators could suggest cognitive primitives that psychologists haven't identified.

### Q9: Can the system discover entirely new optimization algorithms?

Meta-Harness discovers new harness code. Could it discover a new optimizer — something that isn't GEPA, isn't MIPRO, isn't BootstrapFewShot, but a genuinely novel approach to prompt optimization? If so, the system would be inventing its own optimization theory, not just applying existing theory.

### Q10: What's the information-theoretic capacity of a combinator tree?

How many bits of information can a tree of depth d, branching k, with leaf nodes of capacity C_leaf transmit from a document to a final answer? If C_total = k^d * C_leaf (leaves process chunks independently) minus d * C_loss (each REDUCE step loses some information), then there's an optimal tree structure that maximizes C_total - C_loss. This might differ from Lambda-RLM's current cost model, which minimizes compute cost rather than maximizing information transfer.

### Q11: Can we use this system to discover its own successors?

If Meta-Harness can search over system architectures (not just harness code), it could discover an architecture that is fundamentally different from the one we designed — one that we couldn't have imagined. The system would design its successor. This is AI-designed AI, but in a verifiable, interpretable way (the discovered architecture is code, not weights — you can read it).

---

## The Meta-Question

**Q0: Is the right approach to BUILD this system, or to PROVE that this system works?**

Building it tells you IF it works. Proving it tells you WHY and WHEN.

Building is faster. Proving is more durable.

The Lambda-RLM component of the architecture argues for proving (formal guarantees are its core contribution). The Meta-Harness component argues for building (you can't prove properties of arbitrary code, you have to run it and see).

**My answer**: Both, in layers. Build Layer 0-1 (Lambda-RLM + DSPy) with proofs. Build Layer 2-3 (GEPA + BetterTogether) with empirical validation and probabilistic bounds. Build Layer 4-5 (Evaluation + Meta-Harness) with empirical validation only. Build Layer 6 (Temporal) with monitoring and rollback.

The proof coverage decreases as you go up the stack, which mirrors the formal-to-empirical spectrum of the underlying frameworks. This is honest architecture: the guarantees are real where they can be real, and absent where they can't be.

---

*These questions are ordered by importance, not difficulty. Some easy questions (Q4) are lower tier because they don't block progress. Some hard questions (Q1-Q3) are top tier because everything depends on them. Start with Tier 1.*
