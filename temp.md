# MASA — Implementation Guide
**Multi-Agent Social Architecture** | Build order optimized for 90 hours

---

## Phase 0 — Setup (Hours 0–5)

### 0.1 Install dependencies
```bash
# Python environment
python -m venv masa-env
source masa-env/bin/activate

pip install chromadb networkx sqlalchemy fastapi uvicorn \
            sentence-transformers pandas matplotlib pyyaml \
            httpx asyncio python-dotenv

# Node (frontend)
npm install -g create-react-app
```

### 0.2 Install Ollama + Gemma 4
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Gemma 4
ollama pull gemma3:27b   # or gemma3:12b if VRAM limited

# Test it works
ollama run gemma3:27b "Reply with: OK"
```

### 0.3 Initialize project structure
```bash
mkdir masa && cd masa
mkdir core agents social infra scenarios analysis frontend
touch core/{tick_engine,world_state,scenario_loader}.py
touch agents/{agent,memory,payoff,prompt_builder}.py
touch social/{graph,trust,coalition}.py
touch infra/{llm_router,logger,db}.py
```

### 0.4 Create your first scenario config
Create `scenarios/village.yaml`:
```yaml
scenario:
  name: "Village"
  world:
    environment: "medieval village"
    resources:
      grain: { total: 100, regen_per_tick: 5 }
      gold: { total: 50, regen_per_tick: 2 }
    ticks_per_season: 20
  agents:
    - name: "Aldric"
      role: "Merchant"
      agenda: "Corner the grain market before winter"
      risk_tolerance: 0.7
      starting_resources: { grain: 20, gold: 10 }
    - name: "Seline"
      role: "Guard Captain"
      agenda: "Protect the noble family at any cost"
      risk_tolerance: 0.4
      starting_resources: { grain: 5, gold: 20 }
    - name: "Brother Oswin"
      role: "Priest"
      agenda: "Build influence over the noble and merchant both"
      risk_tolerance: 0.3
      starting_resources: { grain: 5, gold: 5 }
```

---

## Phase 1 — LLM Router (Hours 5–12)

This is the most critical piece. Get it right before building anything on top.

### 1.1 Build `infra/llm_router.py`
```python
import httpx
import asyncio
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:27b"

async def call_llm(system_prompt: str, user_prompt: str) -> dict:
    """
    Single LLM call. Returns dict with:
      - action: str
      - speech: str  (shown to player)
      - reasoning: str  (hidden, logged only)
    """
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "stream": False,
        "format": "json"
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        raw = response.json()["message"]["content"]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: return raw string as speech
            return {"action": "idle", "speech": raw, "reasoning": "parse error"}

async def call_llm_batch(calls: list[tuple[str, str]]) -> list[dict]:
    """Run multiple agents in parallel."""
    tasks = [call_llm(sys, usr) for sys, usr in calls]
    return await asyncio.gather(*tasks)
```

### 1.2 Test with a hardcoded persona
```python
# test_llm.py
import asyncio
from infra.llm_router import call_llm

system = """You are Aldric, a cunning merchant in a medieval village.
Your hidden goal is to corner the grain market before winter.
You are risk-tolerant and will lie if it benefits you.

Always respond in this exact JSON format:
{
  "action": "one of: cooperate / negotiate / defect / ignore",
  "speech": "what you say out loud to the other person",
  "reasoning": "your true internal thought — never shown to others"
}"""

user = """Situation: The village guard approaches you at the market.
She says: 'Aldric, share some grain with the poor quarter or there will be trouble.'
Your current grain: 20 units. Market price is rising.
Choose your action and respond."""

result = asyncio.run(call_llm(system, user))
print(result)
```

Run this. Iterate on the prompt until the output is consistently valid JSON with plausible strategic reasoning. This iteration is essential — budget 2–3 hours here.

---

## Phase 2 — Database + Logger (Hours 12–18)

### 2.1 Build `infra/db.py`
```python
from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, JSON, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import uuid

Base = declarative_base()
engine = create_engine("sqlite:///masa.db")
Session = sessionmaker(bind=engine)

class Scenario(Base):
    __tablename__ = "scenarios"
    id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    world_type    = Column(String)
    llm_model     = Column(String)
    scarcity_param = Column(Float)
    created_at    = Column(DateTime, default=datetime.utcnow)

class AgentRecord(Base):
    __tablename__ = "agents"
    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    scenario_id  = Column(String)
    name         = Column(String)
    role         = Column(String)
    hidden_agenda = Column(String)
    persona_config = Column(JSON)

class TickSnapshot(Base):
    __tablename__ = "ticks"
    id             = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    scenario_id    = Column(String)
    tick_number    = Column(Integer)
    world_state    = Column(JSON)
    gini_trust     = Column(Float)
    clustering_coeff = Column(Float)
    recorded_at    = Column(DateTime, default=datetime.utcnow)

class AgentAction(Base):
    __tablename__ = "agent_actions"
    id              = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tick_id         = Column(String)
    agent_id        = Column(String)
    action_type     = Column(String)
    public_speech   = Column(String)
    hidden_reasoning = Column(String)
    payoff_score    = Column(Float)

class SocialEdge(Base):
    __tablename__ = "social_edges"
    id             = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tick_id        = Column(String)
    from_agent_id  = Column(String)
    to_agent_id    = Column(String)
    trust_score    = Column(Float)
    alliance_active = Column(Boolean)

Base.metadata.create_all(engine)
```

---

## Phase 3 — Agent Memory (Hours 18–26)

### 3.1 Build `agents/memory.py`
```python
import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # fast, local
client   = chromadb.Client()

class AgentMemory:
    def __init__(self, agent_id: str):
        self.agent_id   = agent_id
        self.collection = client.get_or_create_collection(f"agent_{agent_id}")

    def write(self, event_text: str, tick: int, salience: float, metadata: dict = {}):
        """Store a memory. High salience = retrieved more often."""
        vector = embedder.encode(event_text).tolist()
        self.collection.add(
            ids=[f"{self.agent_id}_{tick}_{hash(event_text) % 99999}"],
            embeddings=[vector],
            documents=[event_text],
            metadatas=[{"tick": tick, "salience": salience, **metadata}]
        )

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Return top-K memories most relevant to the query."""
        vector = embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=min(k, self.collection.count() or 1)
        )
        return results["documents"][0] if results["documents"] else []

    def decay(self, decay_rate: float = 0.97):
        """Reduce salience of all memories each tick. Prune below threshold."""
        all_items = self.collection.get(include=["metadatas", "embeddings", "documents"])
        to_delete = []
        for i, meta in enumerate(all_items["metadatas"]):
            new_salience = meta["salience"] * decay_rate
            if new_salience < 0.05:
                to_delete.append(all_items["ids"][i])
            else:
                meta["salience"] = new_salience
                # Re-upsert with updated salience
                self.collection.upsert(
                    ids=[all_items["ids"][i]],
                    embeddings=[all_items["embeddings"][i]],
                    documents=[all_items["documents"][i]],
                    metadatas=[meta]
                )
        if to_delete:
            self.collection.delete(ids=to_delete)
```

### 3.2 Salience scoring rules
```python
# agents/memory.py — add this helper
SALIENCE_MAP = {
    "betrayal":       0.95,
    "alliance_formed":0.85,
    "alliance_broken":0.80,
    "trade_accepted": 0.60,
    "threat_issued":  0.70,
    "neutral_speech": 0.20,
    "idle":           0.10,
}

def score_salience(event_type: str, is_self_involved: bool) -> float:
    base = SALIENCE_MAP.get(event_type, 0.3)
    return min(base * 1.3, 1.0) if is_self_involved else base
```

---

## Phase 4 — Social Graph (Hours 26–34)

### 4.1 Build `social/graph.py`
```python
import networkx as nx
from dataclasses import dataclass, field

@dataclass
class EdgeData:
    trust_score:      float = 0.0   # -1.0 to +1.0
    public_stance:    float = 0.0   # may differ from trust
    interaction_count: int  = 0
    alliance_active:  bool  = False
    last_tick:        int   = 0

class SocialGraph:
    ALLIANCE_THRESHOLD  = 0.6
    ALLIANCE_BREAK      = 0.2
    BETRAYAL_PENALTY    = 0.4

    def __init__(self, agent_ids: list[str]):
        self.G = nx.DiGraph()
        for a in agent_ids:
            for b in agent_ids:
                if a != b:
                    self.G.add_edge(a, b, **EdgeData().__dict__)

    def update(self, from_id: str, to_id: str, action: str, tick: int):
        edge = self.G[from_id][to_id]
        delta_map = {
            "cooperate":  +0.10,
            "negotiate":  +0.05,
            "defect":     -0.20,
            "ignore":     -0.01,
            "betrayal":   -self.BETRAYAL_PENALTY,
        }
        delta = delta_map.get(action, 0)
        edge["trust_score"]       = max(-1.0, min(1.0, edge["trust_score"] + delta))
        edge["interaction_count"] += 1
        edge["last_tick"]          = tick

        # Alliance logic
        if edge["trust_score"] > self.ALLIANCE_THRESHOLD:
            edge["alliance_active"] = True
        elif edge["trust_score"] < self.ALLIANCE_BREAK:
            if edge["alliance_active"]:
                # This is a betrayal — propagate penalty to witnesses
                edge["alliance_active"] = False

    def get_context(self, agent_id: str) -> dict:
        """Returns trust scores as seen by agent_id — input to LLM prompt."""
        return {
            neighbor: self.G[agent_id][neighbor]["trust_score"]
            for neighbor in self.G.successors(agent_id)
        }

    def metrics(self) -> dict:
        scores = [d["trust_score"] for _, _, d in self.G.edges(data=True)]
        return {
            "clustering": nx.average_clustering(self.G.to_undirected()),
            "gini_trust": _gini(scores),
            "active_alliances": sum(
                1 for _, _, d in self.G.edges(data=True) if d["alliance_active"]
            )
        }

def _gini(values: list[float]) -> float:
    if not values: return 0
    v = sorted([x + 1 for x in values])  # shift to positive
    n = len(v)
    return (2 * sum((i+1)*v[i] for i in range(n))) / (n * sum(v)) - (n+1)/n
```

---

## Phase 5 — Payoff Matrix (Hours 34–40)

### 5.1 Build `agents/payoff.py`
```python
from dataclasses import dataclass

@dataclass
class PayoffOption:
    action: str
    score:  float
    description: str

def compute_payoff(
    agent_resources: dict,
    world_resources: dict,
    trust_scores: dict,         # {other_agent_id: float}
    agenda_keywords: list[str], # parsed from agenda string
    tick: int
) -> list[PayoffOption]:
    """
    Returns ranked list of available actions with scores.
    LLM must choose from this list — cannot invent new actions.
    """
    scarcity = 1 - (world_resources.get("grain", 50) / 100)
    options = []

    # Cooperate — better when trust is high, worse under scarcity
    avg_trust = sum(trust_scores.values()) / max(len(trust_scores), 1)
    options.append(PayoffOption(
        action="cooperate",
        score=avg_trust * (1 - scarcity * 0.5),
        description="Share resources or assist another agent"
    ))

    # Negotiate — always available, moderate payoff
    options.append(PayoffOption(
        action="negotiate",
        score=0.4 + avg_trust * 0.2,
        description="Propose a trade or alliance"
    ))

    # Defect — high payoff under scarcity, destroys trust
    options.append(PayoffOption(
        action="defect",
        score=scarcity * 0.8 - avg_trust * 0.3,
        description="Act against another agent's interests"
    ))

    # Ignore — safe but passive
    options.append(PayoffOption(
        action="ignore",
        score=0.1,
        description="Take no action this tick"
    ))

    return sorted(options, key=lambda x: x.score, reverse=True)
```

---

## Phase 6 — Prompt Builder (Hours 40–46)

### 6.1 Build `agents/prompt_builder.py`

This is the most important prompt engineering file. The quality of your simulation depends on it.

```python
def build_system_prompt(agent: dict) -> str:
    return f"""You are {agent['name']}, a {agent['role']} in a medieval village simulation.

IDENTITY:
- Your public persona: {agent['role']} — known to everyone
- Your hidden agenda: {agent['hidden_agenda']} — known only to you. NEVER reveal this directly.
- Your risk tolerance: {agent['risk_tolerance']} (0 = very cautious, 1 = very bold)

RULES:
- You may lie, manipulate, or deceive others in your SPEECH.
- Your REASONING must always reflect your true internal state.
- You must choose your ACTION from the options provided. No other actions exist.
- Stay in character across all turns. You remember past interactions.
- Your goal is to advance your hidden agenda over time.

OUTPUT FORMAT (strict JSON, no markdown):
{{
  "action": "<one of the provided options>",
  "speech": "<what you say out loud — can be deceptive>",
  "reasoning": "<your true internal thought — never shown to others>"
}}"""

def build_user_prompt(
    world_state: dict,
    memories: list[str],
    trust_context: dict,
    payoff_options: list,
    player_action: str | None
) -> str:
    memories_text = "\n".join(f"- {m}" for m in memories) or "No relevant memories."
    trust_text    = "\n".join(
        f"- {name}: {score:+.2f}" for name, score in trust_context.items()
    )
    options_text  = "\n".join(
        f"- {opt.action} (score {opt.score:.2f}): {opt.description}"
        for opt in payoff_options
    )

    player_text = f"\nPLAYER JUST DID: {player_action}" if player_action else ""

    return f"""CURRENT WORLD STATE:
{world_state}
{player_text}

YOUR RELEVANT MEMORIES:
{memories_text}

YOUR TRUST LEVELS:
{trust_text}

YOUR OPTIONS THIS TICK (you must choose one):
{options_text}

What do you do?"""
```

---

## Phase 7 — Tick Engine (Hours 46–58)

### 7.1 Build `core/tick_engine.py` — the main loop

```python
import asyncio
from agents.agent import Agent
from social.graph import SocialGraph
from core.world_state import WorldState
from infra.logger import Logger
from infra.llm_router import call_llm_batch

class TickEngine:
    def __init__(self, scenario: dict):
        self.world    = WorldState(scenario["world"])
        self.agents   = [Agent(a) for a in scenario["agents"]]
        self.graph    = SocialGraph([a.id for a in self.agents])
        self.logger   = Logger(scenario["name"])
        self.tick     = 0

    async def run(self, num_ticks: int, player_callback=None):
        for _ in range(num_ticks):
            self.tick += 1
            player_action = player_callback(self.world.state) if player_callback else None

            # Build all LLM calls in parallel
            calls = [
                agent.build_llm_call(
                    world_state   = self.world.state,
                    trust_context = self.graph.get_context(agent.id),
                    player_action = player_action
                )
                for agent in self.agents
            ]

            # Fire all agents simultaneously
            responses = await call_llm_batch(calls)

            # Process responses
            for agent, response in zip(self.agents, responses):
                action  = response.get("action", "ignore")
                speech  = response.get("speech", "")
                reasoning = response.get("reasoning", "")

                # Update world
                self.world.apply_action(agent.id, action)

                # Update social graph
                for other in self.agents:
                    if other.id != agent.id:
                        self.graph.update(agent.id, other.id, action, self.tick)

                # Write memory
                event = f"Tick {self.tick}: I chose to {action}. {speech}"
                agent.memory.write(event, self.tick,
                    salience=agent.memory.score_salience(action, is_self_involved=True))

                # Log
                self.logger.log_action(agent.id, action, speech, reasoning, self.tick)

            # Log graph metrics
            self.logger.log_tick(self.tick, self.world.state, self.graph.metrics())
            self.world.tick()  # resource regen, season update
```

---

## Phase 8 — FastAPI Backend (Hours 58–64)

```python
# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio, yaml
from core.tick_engine import TickEngine

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

engine_state = {"engine": None, "history": []}

@app.post("/start/{scenario_name}")
async def start(scenario_name: str):
    with open(f"scenarios/{scenario_name}.yaml") as f:
        scenario = yaml.safe_load(f)["scenario"]
    engine_state["engine"] = TickEngine(scenario)
    return {"status": "started"}

@app.post("/tick")
async def tick(player_action: str | None = None):
    engine = engine_state["engine"]
    if not engine:
        return {"error": "no engine running"}
    await engine.run(num_ticks=1,
        player_callback=lambda _: player_action)
    return {
        "tick":       engine.tick,
        "world":      engine.world.state,
        "graph":      engine.graph.metrics(),
        "speeches":   engine.logger.last_speeches()
    }

@app.get("/graph")
async def get_graph():
    engine = engine_state["engine"]
    edges = [
        {"from": u, "to": v, "trust": d["trust_score"], "alliance": d["alliance_active"]}
        for u, v, d in engine.graph.G.edges(data=True)
    ]
    return {"nodes": [a.name for a in engine.agents], "edges": edges}
```

---

## Phase 9 — Frontend (Hours 64–76)

### 9.1 Key React components to build

**`SocialGraph.jsx`** — D3 force-directed graph
- Nodes = agents (size by resource wealth)
- Edges = trust score (green positive, red negative, thick = alliance)
- Updates live every tick

**`AgentPanel.jsx`** — shows for each agent:
- Name + role (visible)
- Latest speech bubble
- Trust bar toward player
- Action taken last tick

**`PlayerInput.jsx`** — text box where player types actions
- "I offer grain to Aldric"
- "I accuse Brother Oswin of lying"
- Sent to `/tick` endpoint

**`WorldState.jsx`** — resource bars, current season, tick counter

---

## Phase 10 — Research Pipeline (Hours 76–84)

### 10.1 Build `analysis/metrics.py`

```python
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

def load_simulation(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    return {
        "ticks":   pd.read_sql("SELECT * FROM ticks ORDER BY tick_number", conn),
        "actions": pd.read_sql("SELECT * FROM agent_actions", conn),
        "edges":   pd.read_sql("SELECT * FROM social_edges", conn),
    }

def plot_trust_over_time(data: dict, scenario_name: str):
    ticks   = data["ticks"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ticks["tick_number"], ticks["gini_trust"],       label="Gini (trust inequality)")
    ax.plot(ticks["tick_number"], ticks["clustering_coeff"], label="Clustering (coalition density)")
    ax.set_title(f"Social dynamics — {scenario_name}")
    ax.legend()
    fig.savefig(f"analysis/figures/{scenario_name}_dynamics.png", dpi=150)

def betrayal_rate(data: dict) -> float:
    actions = data["actions"]
    return (actions["action_type"] == "betrayal").mean()

def coalition_lifespan(data: dict) -> float:
    """Average number of ticks an alliance stays active."""
    edges = data["edges"].sort_values(["from_agent_id", "to_agent_id", "tick_id"])
    # ... compute run lengths of alliance_active == True
    pass
```

---

## Phase 11 — Paper Writing (Hours 84–90)

### Research paper outline

**Title:** Emergent Coalition Formation in LLM-Driven Multi-Agent Systems Under Resource Scarcity

**Abstract** (write last)

**Section 1 — Introduction**
- Why current NPC AI is insufficient
- The faithfulness gap in LLM reasoning
- Contribution: MASA framework + empirical findings

**Section 2 — Related Work**
- Axelrod (1984) — evolution of cooperation
- Park et al. (2023) — Generative Agents (Stanford)
- Classic game theory: Nash, Prisoner's Dilemma

**Section 3 — MASA Architecture**
- Formal definition: S = (W, A, I, M)
- Tick loop, agent cognitive stack, social graph

**Section 4 — Experimental Setup**
- 3 scenarios, 50 runs each
- Variables: scarcity parameter (0.2 / 0.5 / 0.8)
- Metrics: Gini coefficient, clustering coeff, betrayal rate, coalition lifespan

**Section 5 — Results**
- Key finding: betrayal rate correlates with scarcity (r = ?)
- Key finding: coalition formation follows predicted Nash threshold
- Key finding: persona consistency degrades after N ticks

**Section 6 — Discussion**
- Limitations: single LLM backbone, no human baselines
- Future work: human participants, larger agent populations

**References**

---

## Milestone checklist

| Milestone | Target hour | Done? |
|---|---|---|
| Ollama + Gemma 4 running | 5 | |
| Single LLM call returns valid JSON | 12 | |
| Database schema live | 18 | |
| Agent memory write + retrieve working | 26 | |
| Social graph updating correctly | 34 | |
| Payoff matrix computed per tick | 40 | |
| One agent making decisions end-to-end | 46 | |
| All agents running in parallel (1 tick) | 52 | |
| Full simulation running 20 ticks | 58 | |
| FastAPI backend serving frontend | 64 | |
| Social graph rendering in D3 | 70 | |
| Playable demo working | 76 | |
| 50-run research dataset generated | 82 | |
| Paper draft complete | 90 | |
