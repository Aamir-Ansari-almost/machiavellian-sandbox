# Machiavellian Sandbox (Multi-Agent Social Architecture)

A scenario-agnostic simulation engine where LLM-driven agents negotiate, form alliances, cooperate, deceive, and betray in natural language — with none of it scripted. Each agent has a persona, a hidden agenda, a persistent memory, and limited resources. Political drama emerges from pure incentive structures and game theory.

> ASE 2026 - PLUS </br>
> **Authors:** Aamir Ansari, Fawad Javed, Ibrahim Saleh </br>
> **Supervised by:** Prof. Christoph Kirsch

---

## Documents

- 📄 **[Project Proposal (PDF)](Machiavellian_Sandbox-Proposal.pdf)** — the original proposal.
- 📄 **[Final Paper (PDF)](MachiavellianSandbox-Paper.pdf)** — *The Machiavellian Sandbox: A Configurable Platform for Testing Game-Theoretic Hypotheses with Language-Model Agents.* LaTeX source in [`paper/main.tex`](paper/main.tex).
- 📄 **[Project Proposal (PDF)](MachiavellianSandbox-Presentation.pdf)** — Project presentation

---

## What we built

A three-layer engine in which the "players" of a game are LLM agents that reason in natural language, carry persistent salience-weighted memory, and act under configurable incentives. The core idea: **agents are never given a strategy — they are given a personality and must improvise.** The same engine expresses settings from a two-player prisoner's dilemma to a five-agent Roman senate by swapping a single config file.

Each agent runs a full cognitive loop every tick:
1. Retrieves relevant memories from a vector database (ChromaDB), weighted by salience and recency.
2. Sees the moves available to it (a payoff matrix computed from the current world state).
3. Calls a local LLM to reason in natural language and decide an action + what to say.
4. Acts; the engine deterministically updates trust, memory, and (in multi-agent scenarios) the social graph.

The social graph — who trusts whom, which alliances are active, who just betrayed whom — emerges entirely from agent decisions. Nothing is hardcoded.

---

## Phases of development

### Phase 1 — Configure local LLM and get structured output back
`🟢 Done`

1. Local LLM running via `llama.cpp` with an OpenAI-compatible endpoint.
2. System prompt sent from Python.
3. JSON-schema-constrained decoding guarantees valid structured output on **every** call — not sometimes.

> Note: the project moved from Ollama/Gemma to **Qwen2.5-7B-Instruct served through `llama.cpp`**, using schema-constrained decoding so malformed output cannot derail a run.

### Phase 2 — Build one agent that thinks, remembers, and acts
`🟢 Done`

1. The agent has a persona and a hidden agenda baked into its system prompt.
2. Before each decision, it retrieves relevant memories from ChromaDB.
3. It receives a payoff matrix telling it which actions are available.
4. It makes a decision and writes that event back into memory.
5. Salience scoring means important events (betrayal) are remembered longer than small ones (idle chat).

### Phase 3 — Run multiple agents together and track how they relate to each other
`🟢 Done`

1. All agents perceive the same world snapshot each tick; decisions are collected concurrently, then resolved in a deterministic order (no race conditions).
2. Every interaction deterministically updates a directed trust score between agent pairs.
3. When mutual trust crosses a threshold, an alliance forms automatically.
4. When an agent betrays an ally, a betrayal event fires and propagates a bounded trust shock to neighbors in the graph.
5. The social graph (NetworkX) is queryable — you can ask "who trusts whom right now?"

### Phase 4 — Experiments, metrics, and analysis
`🟢 Done`

The demo/validation layer, we produced a **controlled experimental study** and static analysis.

1. The iterated prisoner's dilemma reproduced as a constrained instance of the general engine (cooperate/defect only), for an exact comparison to Axelrod's setting.
2. A factorial sweep of controlled conditions (C1–C5): known vs. hidden horizon, communication, a "forgiving" trait, and a naming probe.
3. Every action, trust value, and coalition event logged to SQLite for post-hoc analysis.
4. Metrics computed and plotted (cooperation rate, strategy classification, first-move aggression).
5. A five-agent Roman senate scenario run to test generalization beyond two players.

---

## Key results

The full study is in the [final paper](MachiavellianSandbox.pdf). Headline findings from the iterated prisoner's dilemma (Qwen2.5-7B, N=20 per condition):

- **Agents recover game-theoretic baselines.** Under a *known* finite horizon they defect (cooperation in 1/20 runs, matching backward induction); *hiding* the horizon — the only change — raises cooperation to 10/20, proving they respond to game structure.
- **Communication is not "cheap."** A free-form message channel raises cooperation to 14/20 — and agents spontaneously learn to **deceive** (send a cooperative message, then defect).
- **One trait flips the outcome.** Adding a single "forgiving" trait yields 20/20 sustained cooperation — the same property that won Axelrod's tournament, supplied as character rather than code.
- **A bare name is a personality.** With no persona, an agent named "Alpha" struck first in 5/5 runs; "Beta" did not.
- **Central principle.** The agents do **not** beat tit-for-tat; their play is coarse. But they obey a clean rule: *personality selects among the outcomes the game permits, but cannot induce outcomes the payoffs punish* — **the game draws the boundaries; the personality moves within them.**

---

## Architecture

```
Scenario = ( World, Agents, Infrastructure, LLM )
```

Three independent, swappable layers:

**World layer** — environment definition: resources, scarcity, rules, horizon. Configured entirely via YAML, including a free-form `extra` field for arbitrary natural-language traits. Swap it without touching any agent code.

**Agent layer** — each agent has a persona, a hidden agenda, a ChromaDB memory store, and a dynamic payoff matrix. The LLM reasons within the payoff matrix — it can *say* anything, but it cannot *act* outside its available options.

**Infrastructure layer** — fixed across all scenarios. Vector store (ChromaDB), directed trust graph (NetworkX), SQLite logger, tick engine, and LLM router. Built once, runs everything.

The key insight: output quality scales with LLM capability, not with simulation complexity. Running locally on a 7B model today — swap to a stronger model when needed, and nothing else changes.

---

## Repository layout

```
backend/     — the engine (core tick loop, agents, social graph, infra, scenarios, analysis)
frontend/    — scaffolded React/Vite app (live visualization scoped out)
paper/        — LaTeX source of the final paper (main.tex)
*.pdf         — proposal, presentation and final paper
```

---

## Research question

> Do LLM agents given personalities rather than programs improvise strategic behavior that matches — or exceeds — the canonical hand-designed strategies of classical game theory?

**Answer (this study):** On a small local model, no — they do not beat tit-for-tat, and their play is coarse. But they genuinely reason, respond to game structure, deceive when it pays, and are strongly shaped by framing. The contribution is the **instrument**: a reproducible platform whose findings sharpen automatically as the models placed inside it grow.
