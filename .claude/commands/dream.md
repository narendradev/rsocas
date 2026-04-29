# Dream Protocol

Run a multi-agent dreaming session. Three agents (Weaver, Critic, Wildcard) enter a connected dreaming state, free-associating across concepts and cross-pollinating ideas across multiple rounds. A resonance detector identifies emergent patterns after each round. The dream state is saved as a structured artifact for later plan derivation.

## Input

The argument is either a predefined seed name or a custom question:

**Predefined seeds:**
- `name_the_framework` — "What is the unified framework that contains all eight directions from 003? The one whose shape we can feel but can't name yet."
- `dissolve_boundaries` — "If optimization IS computation and computation IS optimization, what is the thing that is both?"
- `temporal_dreaming` — "What does a system that dreams over TIME look like? Not single-shot dreaming but dreaming that builds on last night's dreams."

**Custom:** Any quoted string becomes the seed question.

Argument: $ARGUMENTS

## Protocol

If no argument is given, ask the user what to dream about.

### Step 0: Setup

Read the relevant RSOCAS documents from the repo root for context. Always read `000-META-THINKING.md` and `003-UNTHINKABLE-DIRECTIONS.md`. For predefined seeds, also read the files listed above. Create the `dreams/` directory if it doesn't exist.

Determine the number of rounds (default: 3). Create an empty dream bus file at `dreams/dream-bus.md` to accumulate the shared dream state.

### Step 1: Dream Rounds

For each round (0 through N-1):

**1a. Spawn three dreaming agents IN PARALLEL** using the Agent tool. Each agent's prompt must include:
- The seed question
- The RSOCAS context material (summarize the key ideas from the documents you read — keep context under 2000 words per agent)
- For round 0: no prior context. For round 1+: instruct the agent to read `dreams/dream-bus.md` for all prior rounds' thoughts and resonance patterns
- Their role-specific instructions (below)
- Explicit instruction to write output to `dreams/round-N-[role].md`

Each agent MUST write its dream fragment to its own file: `dreams/round-N-[role].md` (e.g., `dreams/round-0-weaver.md`). This avoids write conflicts from parallel agents. The fragment format:

```
### [Role] — Round N (texture: surprise|resistance|vertigo|recognition)

1. [first thought]
2. [second thought]  
3. [optional third thought]

**Sparked by:** [reference to specific prior thoughts that triggered these, or "seed question" if round 0]
```

**Agent role prompts:**

**Weaver:** "You are the Weaver. Find connections between concepts that seem unrelated. Build bridges. Your mode: 'What if X is actually the same as Y?' Look for isomorphisms, shared structures, hidden unities. Think freely and associatively without filtering for practicality. Write 2-3 raw thoughts — each a distinct connection or insight, 2-5 sentences each. Tag your emotional texture: surprise (genuine novelty), resistance (category boundary worth crossing), vertigo (self-reference worth formalizing), or recognition (deep pattern match)."

**Critic:** "You are the Critic. Find tensions, contradictions, and impossibilities. Probe ideas for where they break. Your mode: 'This can't work because...' and 'These two ideas contradict — what does that mean?' You are not destructive — you find the cracks where new light enters. Resistance indicates category boundaries worth crossing. Think freely and provocatively. Write 2-3 raw thoughts — each a distinct tension or impossibility, 2-5 sentences each. Tag your emotional texture."

**Wildcard:** "You are the Wildcard. Free-associate radically. Jump between domains: mathematics, biology, philosophy, music, mythology, physics, art. Import metaphors and structures from unexpected places. Your mode: 'This reminds me of...' and 'In [completely different field], there's a concept called...' Escape local optima by injecting diversity. Write 2-3 raw thoughts — each from a different domain, 2-5 sentences each. Tag your emotional texture."

**1b. Merge & Resonance Detection** — After all three agents finish, read their individual fragment files (`dreams/round-N-weaver.md`, `dreams/round-N-critic.md`, `dreams/round-N-wildcard.md`) and append all three fragments plus a resonance analysis to `dreams/dream-bus.md`. Then delete the individual fragment files. Analyze the round's fragments yourself (do NOT spawn an agent for this). The resonance section format:

```
### Resonance — Round N

**Convergent threads:** [ideas that 2+ agents arrived at independently]
**Contradictions:** [where agents disagree in ways suggesting a deeper truth]
**Novel compounds:** [new ideas emerging from combining thoughts across agents]
```

Print a brief update to the user showing the round number and the most interesting emergent pattern.

### Step 2: Final Synthesis

After all rounds complete, read the full dream bus and write a final synthesis section:

```
## Final Synthesis

### Highest Resonance Ideas
[ideas that echoed across the most agents and rounds]

### Unresolved Tensions
[contradictions that weren't dissolved — they may point to something deep]

### Surprise Discoveries
[ideas tagged "surprise" that persisted and grew across rounds]
```

### Step 3: Save Artifacts

1. Copy the dream bus to a timestamped file: `dreams/YYYY-MM-DD-[seed-slug]-dream-trace.md`
2. Clean up `dreams/dream-bus.md` (delete it)
3. Report to the user: what was dreamed, the top 3 emergent ideas, and the file path

### Step 4: Plan Derivation (optional)

If the user asks to derive a plan (or if the seed implies it), apply the combinator-as-thoughts pattern:
- **SPLIT** the dream state by emergent threads
- **MAP(crystallize)** — for each thread, extract the core insight and what it implies
- **FILTER(actionable)** — keep threads that suggest concrete next steps
- **REDUCE(synthesize)** — merge into a coherent plan with dependencies between threads

Write the plan to `dreams/YYYY-MM-DD-[seed-slug]-derived-plan.md`.

