# RSOCAS — Recursive Self-Optimizing Compound AI System

## About This Repo

Pre-implementation thinking repo. Documents 000-006 contain raw reasoning traces about synthesizing five AI frameworks (GEPA, DSPy, Lambda-RLM, Meta-Harness, LangProBe) into a unified self-optimizing system. The core thesis: raw traces contain more signal than summaries, and the connective tissue between frameworks is where novelty lives.

## Dream Protocol

This repo includes a multi-agent dreaming protocol for collaborative ideation. Use it to explore RSOCAS concepts or any open question.

**Slash command:** `/dream [seed or question]`

**Natural language:** "Run a dream session about [topic]" — follow the protocol in `.claude/commands/dream.md`.

**Predefined seeds:** `name_the_framework`, `dissolve_boundaries`, `temporal_dreaming`

**How it works:** Three agents (Weaver, Critic, Wildcard) dream in parallel across multiple rounds. A resonance detector identifies emergent patterns after each round. Output goes to `dreams/`.

**Plan derivation:** After a dream session, say "derive a plan from the dream" to apply the combinator-as-thoughts pattern (SPLIT -> MAP -> FILTER -> REDUCE) and extract actionable threads.

## Key Concepts (for agent context)

- **Traces > summaries**: Every framework that works well uses execution traces as primary signal. Raw traces outperform summaries.
- **Five views of the same elephant**: DSPy (program), GEPA (parameters), Lambda-RLM (math), Meta-Harness (code), LangProBe (evaluation) are different representations of the same system.
- **Connective tissue**: The innovation is in the translation between frameworks, not the frameworks themselves.
- **Combinators as thoughts**: SPLIT, MAP, FILTER, REDUCE, CROSS are both computational and cognitive primitives.
- **Emotional texture**: surprise = genuine novelty, resistance = category boundary worth crossing, vertigo = self-reference worth formalizing, recognition = deep pattern match.

