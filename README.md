# Machiavellian Sandbox (Multi-Agent Social Architecture)

A political simulation engine where LLM-driven agents negotiate, form alliances, and betray each other in natural language; without any of it being scripted.
Drop a player into a village. Every NPC has a hidden agenda, a persistent memory, and limited resources. Watch a political drama unfold from pure incentive structures and game theory.

> ASE 2026 - PLUS </br>
> **Authors:** Aamir Ansari, Fawad Javed </br>
> **Supervised by:** Prof. Christoph Kirsch
---
## Phases of development

### Phase 1 — Configure local LLM and get structured output back
`🟢 Done`

1. Ollama installed and Gemma 4 running locally
2. Send a system prompt from Python
3. The response comes back as valid JSON every time, not sometimes

### Phase 2 — Build one agent that thinks, remembers, and acts
`🟡 In Progress`

1. The agent has a persona and a hidden agenda baked into its system prompt
2. Before each decision, it retrieves relevant memories from ChromaDB
3. It receives a payoff matrix telling it what actions are available
4. It makes a decision and writes that event back into memory
5. Salience scoring means important events (betrayal) are remembered longer than small ones (idle chat)

> Why it's hard: the memory retrieval query has to be meaningful. "What do I know about the player?" needs to return useful context, not random memories. This takes tuning.

### Phase 3 — Run multiple agents together and track how they relate to each other
`🟠 TODO`

1. All agents run in parallel each tick, not one after another
2. Every interaction updates a trust score between agent pairs
3. When trust crosses a threshold, an alliance forms automatically
4. When an agent defects against an ally, a betrayal fires and nearby agents notice
5. The social graph is queryable — you can ask "who trusts who right now?"

> Why it's hard: agent interactions have side effects on other agents. A betrayal between agent A and B changes how agent C behaves toward both of them. Wiring this propagation correctly without creating infinite loops is the main engineering challenge of this phase.

### Phase 4 — Visualize what's happening
`🟠 TODO`

This is the demo layer. The simulation works headlessly by now — this makes it visible and playable.
1. FastAPI serves the simulation state to the frontend each tick
2. The player types an action and it gets injected into the next tick
3. The social graph renders live in D3 — you can watch trust shift in real time
4. Agent speech bubbles show what each NPC said last turn
5. The world state panel shows resources, season, tick number

---

## What this is

Each agent runs a full cognitive loop every turn:
1. Retrieves relevant memories from a vector database
2. Computes a payoff matrix based on current resources and trust levels
3. Calls a local LLM to reason in natural language within those constraints
4. Acts, speaks, and updates its relationship with every other agent

The social graph — who trusts who, which alliances are active, who just betrayed whom — emerges entirely from agent decisions. Nothing is hardcoded.

The framework is scenario-agnostic. The same engine runs a medieval village, a Roman senate, a startup, or a commodity market. Swap the YAML config, rerun, compare.

---

## Research question

> Do LLM agents under game-theoretic constraints and resource scarcity exhibit coalition formation patterns consistent with classical political theory?

This project is being developed alongside a research paper targeting AAMAS (Autonomous Agents and Multi-Agent Systems). The simulation generates a full dataset of social graph trajectories across controlled experiments — varying scarcity, agent count, and LLM capability — to measure betrayal rate, coalition lifespan, and trust inequality (Gini coefficient).

---

## Architecture

```
Scenario = ( World, Agents, Infrastructure, LLM )
```

Three independent layers:

**World layer** — environment definition: resources, rules, time model, event queue. Configured entirely via YAML. Swap it without touching any agent code.

**Agent layer** — each NPC has a persona, a hidden agenda, a ChromaDB memory store, and a dynamic payoff matrix. The LLM reasons within the payoff matrix — it can say anything, but it cannot act outside its available options.

**Infrastructure layer** — fixed across all scenarios. Vector store, social graph (NetworkX), SQLite logger, LLM router. Built once, runs everything.

The key insight: output quality scales with LLM capability, not with simulation complexity. Running locally on Gemma 4 today. Swap to a stronger model when needed — nothing else changes.

---

## Metrics to capture

- **Gini coefficient of trust** — inequality in the trust distribution across agents
- **Clustering coefficient** — density of coalition formation
- **Betrayal rate** — fraction of ticks where a betrayal event fires
- **Coalition lifespan** — average ticks an alliance remains active
- **Persona consistency score** — whether agent behavior matches stated agenda over time

---

## Hypotheses under investigation

**H1** — Betrayal rate correlates positively with resource scarcity (r > 0.6)

**H2** — Alliances form when mutual trust exceeds 0.6, consistent across all scenarios and world types

**H3** — Agent persona consistency degrades after tick 30 without memory retrieval, and recovers when memory is re-enabled
