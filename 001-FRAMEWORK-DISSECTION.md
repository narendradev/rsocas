# 001 — Framework Dissection: What Each System Actually Does

**Date**: 2026-04-28

---

## Why This Document Exists

Before synthesizing, I need to know what I'm synthesizing. Not the marketing pitch. Not the abstract. The actual mechanism — what goes in, what comes out, what invariants hold, what breaks.

---

## GEPA (Genetic-Evolutionary Prompt Adaptation)

**What it actually does**: Takes a compound AI system (multiple LLM modules chained together), runs it on training examples, captures the full execution trace (not just final output), feeds the trace to a reflection LLM that proposes a modified prompt for one specific module, then evaluates whether the modification helped.

**The mechanism I didn't expect**: Pareto-based candidate selection. Most optimization keeps the single best candidate and mutates it. GEPA maintains a *population* where each candidate is preserved if it's the best on ANY training example. A candidate that scores 90% overall but aces the hardest example survives alongside a candidate that scores 95% overall but fails the hardest example. This prevents premature convergence.

Formally: for each training instance i, find the set of candidates P*[i] that achieve the best score. Union these. Remove dominated candidates (never best on anything). Sample proportionally to how many instances each candidate is best on.

This is borrowed from MAP-Elites / Quality-Diversity optimization — a field humans rarely connect to LLM optimization. The connection is: **prompt optimization is not hill-climbing. It's niche-filling.** Different prompts are good for different types of inputs. The optimal system might need to ROUTE between multiple prompt variants, not find one universal prompt.

**What I think is missing**: GEPA mutates one module per iteration (round-robin). But modules interact. A mutation to the retrieval module might only help if the synthesis module's prompt also changes to expect the new retrieval format. Coordinated multi-module mutations could unlock improvements that sequential single-module mutations can't reach. The search space is larger, but GEPA's sample efficiency might handle it.

**The real innovation**: Using execution traces as "text gradients." Scalar rewards tell you THAT something failed. Traces tell you WHY, WHERE, and HOW. The reflection LM doesn't need to explore blindly — it can diagnose. This is analogous to the difference between zeroth-order optimization (perturb and check) and first-order optimization (follow the gradient). Traces provide a discrete analog of gradient direction.

**Core numbers**:
- 78x more sample-efficient than GRPO
- 243-1179 rollouts to match GRPO's best (which needs 24K)
- Cross-model transfer: prompts optimized on Qwen3-8B gain +9% on GPT-4.1 Mini
- ICLR 2026 Oral (top ~1% of submissions)

---

## DSPy BetterTogether

**What it actually does**: A meta-optimizer that takes a strategy string like "p -> w -> p" and executes a sequence of optimization passes. "p" means prompt-optimize (default: BootstrapFewShotWithRandomSearch). "w" means weight-optimize (default: BootstrapFinetune). Each pass receives the program state from the previous pass. Between passes, it re-evaluates on a validation set and tracks all candidates.

**The mechanism I didn't expect**: The alternation isn't just "do both." Each pass changes the *landscape* for the next. Prompt optimization before fine-tuning creates better training data (the few-shot examples are optimized). Fine-tuning before the second prompt optimization changes the model's behavior, so prompts that were optimal for the base model may be suboptimal for the fine-tuned model. The second prompt pass adapts.

This is *exactly* like multi-pass compilation in compilers. Each pass exposes structure for the next. LLVM does: parse → AST → IR → optimize → lower → optimize → codegen. BetterTogether does: base model → optimize prompts → fine-tune → optimize prompts for fine-tuned model. The analogy is precise, not metaphorical.

**What I think is missing**: The strategy string is flat. It can't express:
- Conditionals: "if accuracy < 80%, do another gepa pass"
- Loops: "repeat gepa until convergence"
- Parallelism: "run mipro and gepa in parallel, keep the better result"
- Nesting: "within each gepa iteration, use meta-harness to search code"

Making the strategy string a strategy *program* would be a substantial advance. And that program could itself be optimized by Meta-Harness.

**The real innovation**: Making optimization passes composable via a uniform interface (`compile(student, trainset=...)`). This means new optimizers (GEPA, SIMBA, whatever comes next) plug in without framework changes. The BetterTogether meta-optimizer is optimizer-agnostic — it chains *any* optimizers in *any* order.

**Core numbers**:
- 5-78% improvement on HotPotQA over single-stage
- 2.5-10% on GSM8K
- 3.5-88% on Iris classification
- Works across Mistral-7B, Llama-2-7B, Llama-3-8B

---

## Lambda-RLM

**What it actually does**: Takes a long-context task (QA, summarization, translation, extraction over large documents), classifies the task type with ONE LLM call, computes a mathematically optimal decomposition tree (ZERO LLM calls — pure math), then executes the tree where only leaf nodes make LLM calls on bounded text chunks.

**The mechanism I didn't expect**: The cost model is closed-form and computed BEFORE execution.

```
k* = ceil(sqrt(n * c_in / c_compose))    // optimal branching factor
d  = ceil(log_k*(n/K))                    // tree depth
tau* = min(K, floor(n/k*))                // leaf chunk size
C_hat = k*^d * C(tau*) + d * C_compose(k*) + C(500)  // total cost
```

This means you can answer "how much will this cost?" before spending a single token. No other agentic system can do this. And the accuracy bound `A_total >= (a_leaf)^d * (a_compose)^d >= alpha` means you can answer "how accurate will this be?" too.

**What I think is missing**:

First: the task classification is a single LLM call with a digit menu. This is fragile. A misclassification means the wrong combinator chain, which means structurally wrong decomposition, which no amount of prompt optimization can fix. This should be a DSPy module optimized by GEPA — not a raw prompt.

Second: the combinators are fixed. SPLIT always splits by character count, not by semantic boundaries. REDUCE uses one of a few hardcoded strategies. These choices are good defaults but not adaptive. Let GEPA optimize the split heuristic. Let Meta-Harness discover new reduce strategies.

Third: the cost model assumes uniform leaf difficulty. In practice, some chunks are harder than others (contain ambiguous language, require world knowledge, etc.). A non-uniform cost model would allocate more compute to hard chunks (deeper sub-trees or more powerful models for specific leaves).

**The real innovation**: Separating control flow (combinators — deterministic, verified, typed) from neural computation (leaf calls — probabilistic, bounded). This separation means you can reason about the *structure* of the computation without reasoning about the *content* of neural calls. It's the AI equivalent of separating the CPU's instruction decoder from its ALU.

**Core numbers (our benchmarks)**:
- +11.7pp F1 over normal RLM on Nemotron-3-Super (nvfp4)
- 29/36 wins across model-task pairs
- Up to +21.9pp average accuracy
- Up to 4.1x lower latency on cloud APIs
- +28.6pp on cross-reference tasks (O(n^2) complexity)

---

## Meta-Harness

**What it actually does**: An agentic outer loop where a proposer agent (Claude Opus) has full filesystem access to an archive of all prior harness candidates — their source code, execution traces, and evaluation scores. Each iteration: read the archive (~82 files, ~10M tokens), diagnose failure modes, propose new harness variants (100-1000 LOC Python), evaluate, archive everything.

**The mechanism I didn't expect**: Raw traces outperform LLM-generated summaries. Ablation: 50.0% median with full traces vs 34.9% with summaries. This is counterintuitive — summaries should compress signal and remove noise. But they also remove *the subtle correlations that diagnosis depends on*. A summary might say "the model struggled with multi-hop questions." The raw trace shows that on multi-hop questions, the retrieval step returned relevant passages but the synthesis step ignored passage 3, which contained the bridging entity. The diagnosis requires the detail.

Implication: **any optimization system that summarizes intermediate results is leaving performance on the table.** This applies to GEPA too — it could benefit from even richer traces.

**What I think is missing**: Meta-Harness discovers harness code but doesn't verify it formally. A discovered harness might contain subtle bugs (race conditions, off-by-one in chunking, silent failures) that happen to improve scores on the eval set by accident. Lambda-RLM's formal properties could act as a verification layer — any discovered harness must respect termination bounds, cost bounds, and type signatures.

Also: Meta-Harness uses Claude Opus as the proposer, which is expensive. Could the proposer itself be a cheaper model with an optimized prompt (via GEPA)? This creates a beautiful recursion: use GEPA to optimize the prompt of the agent that runs Meta-Harness that discovers code improvements that feed back into GEPA.

**The real innovation**: Searching in *code space* rather than text space. Prompts are limited — they can't express routing logic, memory management, preprocessing pipelines, or architectural decisions. Code can express anything. And because the discovered harnesses are inspectable Python, you can audit them for overfitting (brittle if-chains targeting specific examples are visible in the code).

**Core numbers**:
- +7.7pp over SOTA (ACE) on text classification at 4x fewer context tokens
- 10x faster convergence than OpenEvolve/TTT-Discover
- +4.7pp pass@1 averaged across 5 held-out models (cross-architecture transfer)
- TerminalBench-2: 76.4% with Opus 4.6 (#2 overall), 37.6% with Haiku 4.5 (#1 among Haiku)

---

## LangProBe

**What it actually does**: Benchmarks 2000+ combinations of tasks × architectures × optimizers × language models to answer: "when does composing LLMs into programs beat calling a bigger model directly?"

**The core finding**: On average across diverse tasks, smaller models within optimized programs beat raw calls to larger models at a fraction of the cost.

**Why this matters for our system**: It means the effort of building RSOCAS isn't wasted on strong models. It's MOST valuable for weak-to-medium models (exactly the kind you'd deploy on edge, on device, or at scale). Lambda-RLM's +11.7pp on Nemotron-3-Super (a medium model with nvfp4 quantization) is consistent with LangProBe's finding: structured programs amplify weak models more than strong ones.

**The implicit research agenda**: If optimized programs beat bigger models, then the *optimizer is more important than the model*. Investing in better GEPA, better Meta-Harness, better BetterTogether chains has higher ROI than training bigger models. This inverts the current industry focus.

---

## Cross-Framework Observations

### Shared Assumptions (that might be wrong)

1. **All frameworks assume fixed models during optimization.** GEPA freezes weights. Lambda-RLM uses the model as-is. Meta-Harness holds the model constant. But BetterTogether's fine-tuning step changes the model. What if the model adapted *continuously* during GEPA optimization? Not full fine-tuning, but online learning — updating a small adapter based on each rollout's feedback.

2. **All frameworks optimize for a finite training/validation set.** This limits generalization. What if the training set itself were adversarially generated to probe weaknesses? GAN-style: one agent generates hard examples, another optimizes the system. The training distribution would shift to cover failure modes.

3. **All frameworks operate on static tasks.** Real-world tasks evolve. Today's QA questions are different from tomorrow's. A continually learning system needs to detect distribution shift and trigger re-optimization. This requires the evaluation layer to monitor production traces, not just benchmark results.

### Complementary Strengths (that compound)

| What GEPA lacks | What provides it |
|----------------|-----------------|
| Structural guarantees | Lambda-RLM's typed combinators |
| Code-level optimization | Meta-Harness |
| Multi-model orchestration | BetterTogether |
| Systematic evaluation | LangProBe |

| What Lambda-RLM lacks | What provides it |
|----------------------|-----------------|
| Self-optimization | GEPA + Meta-Harness |
| Prompt quality | GEPA |
| Model-agnostic compilation | DSPy |
| Adaptive combinators | Meta-Harness discovery |

| What Meta-Harness lacks | What provides it |
|------------------------|-----------------|
| Formal verification | Lambda-RLM |
| Sample efficiency | GEPA (78x less data) |
| Structured traces | Lambda-RLM tree |
| Prompt-level optimization | GEPA + DSPy |

**The compound effect**: Each framework's weakness is another framework's strength. This is not common. Usually, frameworks compete on the same axis. Here, they're on orthogonal axes. The multiplication is real.

---

*This document is diagnostic, not prescriptive. It tells you what each thing IS, not what to DO with it. See 000-META-THINKING.md for the synthesis.*
