from typing import Optional

import chromadb

from core.scenario_loader import AgentConfig
from core.world_state import WorldState
from agents.memory import AgentMemory
from agents.prompt_builder import build_system_prompt, build_user_prompt
from infra.llm_router import call_agent, AgentDecision


class Agent:
    """
    One NPC. Runs the full cognitive loop each tick:

        perceive  — retrieve relevant memories for the current situation
        decide    — build the prompt and call the LLM
        act        — (handled by the tick engine, which resolves the decision)
        remember  — write the tick's events back to memory

    Trust levels and alliances are supplied by the tick engine from the social
    graph (Week 2). Until that exists, the agent runs with empty trust/allies.
    """

    def __init__(
        self,
        config: AgentConfig,
        run_id: str,
        memory_client: Optional["chromadb.api.ClientAPI"] = None,
        recency_halflife: float = 10.0,
    ) -> None:
        self.config = config
        self.name = config.name
        self.system_prompt = build_system_prompt(config)
        self.memory = AgentMemory(
            agent_name=config.name,
            run_id=run_id,
            client=memory_client,
            recency_halflife=recency_halflife,
        )

    # ── Perceive ──────────────────────────────────────────────────────────────

    def _derive_situation(self, world: WorldState) -> str:
        """Summarise the agent's current pressure into a retrieval-query phrase."""
        res = world.get_agent_resources(self.name)
        parts = ", ".join(f"{k}={v:.0f}" for k, v in res.items())
        return f"my resources ({parts}) under scarcity {world.scarcity}"

    # ── Decide ────────────────────────────────────────────────────────────────

    async def decide(
        self,
        world: WorldState,
        others: list[str],
        trust_view: Optional[dict[str, float]] = None,
        allies: Optional[set[str]] = None,
        concern: Optional[str] = None,
        k_memories: int = 5,
        recent_window: int = 3,
    ) -> AgentDecision:
        """
        Full perceive → decide step. Returns a validated AgentDecision.
        `concern` optionally names the agent this turn's memory query should
        focus on (e.g. a known betrayer); the tick engine sets it once trust exists.

        Two memory channels feed the prompt: a recent buffer (working memory —
        the last `recent_window` ticks, unconditionally) and salience-weighted
        long-term recall.
        """
        trust_view = trust_view or {}
        allies = allies or set()

        situation = self._derive_situation(world)
        query = AgentMemory.build_query(concern, situation)
        memories = self.memory.retrieve(query, current_tick=world.tick, k=k_memories)
        recent = self.memory.recent(current_tick=world.tick, window=recent_window)

        user_prompt = build_user_prompt(
            agent_name=self.name,
            world=world,
            others=others,
            trust_view=trust_view,
            allies=allies,
            memories=memories,
            recent=recent,
        )

        return await call_agent(self.system_prompt, user_prompt)

    # ── Remember ──────────────────────────────────────────────────────────────

    def remember(
        self,
        text: str,
        tick: int,
        action: str,
        tags: Optional[list[str]] = None,
    ) -> None:
        """Write one event into this agent's memory; salience derives from action."""
        self.memory.write(text=text, tick=tick, action=action, tags=tags)
