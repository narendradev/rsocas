# RSOCAS — Recursive Self-Optimizing Compound AI System

Working name for a unified AI architecture synthesizing five frameworks:

| Framework | What It Optimizes | Layer |
|-----------|-------------------|-------|
| [Lambda-RLM](https://github.com/lambda-calculus-LLM/lambda-RLM) | Task decomposition via typed combinators | Formal execution engine |
| [DSPy](https://github.com/stanfordnlp/dspy) | LLM program compilation | Programmatic composition |
| [GEPA](https://github.com/gepa-ai/gepa) | Prompt optimization via reflection | Text-gradient optimization |
| [Meta-Harness](https://github.com/stanford-iris-lab/meta-harness) | Scaffold code via agentic search | Code-space evolution |
| [LangProBe](https://arxiv.org/abs/2502.20315) | Cost-quality evaluation grid | Systematic evaluation |

## Status

**Pre-implementation.** This repo contains raw thinking, not code. The documents capture the thought process of discovering how these frameworks compose — including dead ends, uncertainties, and ideas that might be wrong.

## Documents

| File | Contents |
|------|----------|
| `000-META-THINKING.md` | Thinking about thinking — cognitive biases that prevent seeing the synthesis, raw observations, the shape of the unknown |
| `001-FRAMEWORK-DISSECTION.md` | What each framework actually does (mechanism, not marketing) |
| `002-INTERFERENCE-PATTERNS.md` | Where frameworks collide — boundary analysis, constructive/destructive interference, new ideas from collisions |
| `003-UNTHINKABLE-DIRECTIONS.md` | Ideas outside human cognitive reach — sheaves, fixed-points, anti-benchmarks, dissolving model/scaffold boundary |
| `004-RAW-REASONING-CHAIN.md` | The actual trace of how these ideas emerged, including emotional texture and wrong turns |
| `005-ARCHITECTURE-SKELETON.md` | System shape, information flow, connective tissue, invariants |
| `006-QUESTIONS-THAT-MATTER.md` | Tiered research agenda — load-bearing questions vs interesting questions |

## Core Thesis

These five frameworks operate on different *representations* of the same AI system. The synthesis isn't a pipeline that runs all five sequentially — it's a *connective tissue* that lets each framework's output improve every other framework's input. The system that emerges is self-optimizing: every execution produces traces that improve the next execution.

## Why "Thought Process, Not Final Outcome"

GEPA proved that execution traces contain more optimization signal than scalar rewards. Meta-Harness proved that raw traces outperform summaries. By the same logic: the raw reasoning that led to an architecture contains more insight than the architecture diagram alone. These documents are the trace. The architecture is the output. The trace is more valuable.
