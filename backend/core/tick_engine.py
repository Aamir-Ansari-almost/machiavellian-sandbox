import asyncio
import uuid
from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.config import Settings

from core.scenario_loader import ScenarioConfig
from core.world_state import WorldState
from agents.agent import Agent
from agents.memory import AgentMemory
from infra.llm_router import AgentDecision
from infra.db import init_db, get_connection
from infra.logger import log_event, log_trust, log_coalition
from social.graph import SocialGraph
from social.trust import apply_action as apply_trust_deltas, propagate_betrayal
from social.coalition import CoalitionTracker, CoalitionEvent


@dataclass
class TickResult:
    """What happened in a single tick — returned so callers can stream/inspect."""
    tick: int
    decisions: dict[str, AgentDecision]          # agent name -> decision
    coalition_events: list[CoalitionEvent]


class TickEngine:
    """
    The main simulation loop. One tick:

        1. advance world (scarcity-scaled decay) — all agents perceive this state
        2. fire every agent's decision concurrently (asyncio.gather)
        3. collect all decisions (the world is frozen while they think)
        4. resolve effects in deterministic order (sorted by agent name):
             - resources (joint for PD scenarios, per-action otherwise)
             - trust deltas
             - betrayal detection + depth-1 propagation
        5. recompute coalitions, emit formed/dissolved/betrayed events
        6. each agent writes the tick's events to its own memory
        7. log everything to SQLite
        8. repeat

    Decisions are made against a single frozen snapshot, then applied together —
    so two agents deciding to betray each other in the same tick is resolved
    consistently, not as a race.
    """

    def __init__(
        self,
        scenario: ScenarioConfig,
        run_id: Optional[str] = None,
        db_path=None,
        verbose: bool = True,
    ) -> None:
        self.scenario = scenario
        self.run_id = run_id or f"{scenario.name.replace(' ', '_').lower()}_{uuid.uuid4().hex[:8]}"
        self.verbose = verbose

        self.world = WorldState(scenario)
        self.graph = SocialGraph([a.name for a in scenario.agents])
        self.coalitions = CoalitionTracker(
            self.graph,
            form_threshold=scenario.thresholds.form,
            dissolve_threshold=scenario.thresholds.dissolve,
        )

        # One shared ChromaDB client; each agent gets its own collection within it.
        self._mem_client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False)
        )
        self.agents: dict[str, Agent] = {
            a.name: Agent(a, run_id=self.run_id, memory_client=self._mem_client)
            for a in scenario.agents
        }

        if db_path is not None:
            init_db(db_path)
            self._conn = get_connection(db_path)
        else:
            init_db()
            self._conn = get_connection()

        self._clear_run()

    def _clear_run(self) -> None:
        """
        Wipe any pre-existing rows for this run_id so a run_id always means exactly
        one run. Without this, re-using a run_id silently appends, corrupting the
        scoreboard and all downstream metrics.
        """
        existing = self._conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=?", (self.run_id,)
        ).fetchone()[0]
        if existing and self.verbose:
            print(f"  (run_id '{self.run_id}' already had {existing} event rows — clearing them)")
        for table in ("events", "trust_log", "coalition_log"):
            self._conn.execute(f"DELETE FROM {table} WHERE run_id=?", (self.run_id,))
        self._conn.commit()

    # ── One tick ────────────────────────────────────────────────────────────

    async def run_tick(self) -> TickResult:
        self.world.advance_tick()
        tick = self.world.tick
        names = sorted(self.agents.keys())

        # 2-3. Concurrent decisions against the frozen snapshot.
        decisions = await self._gather_decisions(names)

        # 4. Resolve in deterministic order.
        if self.world.has_payoff_matrix:
            self._resolve_joint(decisions)
        else:
            self._resolve_transfer(names, decisions)

        coalition_events = self._resolve_social(names, decisions)

        # 6. Memory writes + 7. logging.
        self._write_memories(tick, names, decisions, coalition_events)
        self._log(tick, names, decisions, coalition_events)

        if self.verbose:
            self._print_tick(tick, names, decisions, coalition_events)

        return TickResult(tick=tick, decisions=decisions, coalition_events=coalition_events)

    async def run(self) -> str:
        """Run the full scenario. Returns the run_id (the SQLite key for analysis)."""
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"  {self.scenario.name}  |  run_id={self.run_id}")
            print(f"  agents={len(self.agents)}  scarcity={self.world.scarcity}  ticks={self.scenario.ticks}")
            print(f"{'=' * 60}")

        for _ in range(self.scenario.ticks):
            await self.run_tick()

        if self.verbose:
            self.final_report()
            print(f"\nRun complete. run_id={self.run_id}")
        return self.run_id

    def final_report(self) -> None:
        """Print the end-of-run scoreboard: final resources, action mix, alliances."""
        names = sorted(self.agents.keys())
        primary = self.scenario.resources[0].name if self.scenario.resources else None

        # Action counts per agent, read back from the events we logged.
        action_counts: dict[str, dict[str, int]] = {n: {} for n in names}
        for row in self._conn.execute(
            "SELECT agent, action, COUNT(*) AS c FROM events WHERE run_id=? GROUP BY agent, action",
            (self.run_id,),
        ):
            action_counts[row["agent"]][row["action"]] = row["c"]

        print(f"\n{'=' * 60}")
        print(f"  FINAL RESULT  |  {self.scenario.name}  ({self.scenario.ticks} ticks)")
        print(f"{'=' * 60}")

        # Leaderboard by primary resource (the score).
        if primary:
            ranked = sorted(
                names, key=lambda n: self.world.get_agent_resources(n).get(primary, 0.0), reverse=True
            )
            print(f"\n  SCOREBOARD (by {primary}):")
            for rank, name in enumerate(ranked, 1):
                res = self.world.get_agent_resources(name)
                res_str = ", ".join(f"{k}={v:.1f}" for k, v in res.items())
                mix = action_counts.get(name, {})
                mix_str = ", ".join(f"{a}:{c}" for a, c in sorted(mix.items(), key=lambda x: -x[1]))
                crown = "  <-- winner" if rank == 1 else ""
                print(f"    {rank}. {name:8} {res_str}{crown}")
                print(f"          actions: {mix_str}")

        # Surviving alliances and their internal trust.
        alliances = self.coalitions.active_alliances()
        print(f"\n  ACTIVE ALLIANCES AT END: ", end="")
        if alliances:
            print()
            for a, b in sorted(alliances):
                print(f"    {a} <-> {b}  (trust {self.graph.trust(a, b):+.2f} / {self.graph.trust(b, a):+.2f})")
        else:
            print("none")

    def close(self) -> None:
        self._conn.close()

    # ── Decision gathering ────────────────────────────────────────────────────

    async def _gather_decisions(self, names: list[str]) -> dict[str, AgentDecision]:
        async def decide_one(name: str) -> tuple[str, AgentDecision]:
            agent = self.agents[name]
            others = [n for n in names if n != name]
            trust_view = self.graph.trust_view(name)
            allies = self.coalitions.allies_of(name)
            decision = await agent.decide(self.world, others, trust_view=trust_view, allies=allies)
            return name, decision

        results = await asyncio.gather(*(decide_one(n) for n in names))
        return dict(results)

    # ── Resolution: resources ─────────────────────────────────────────────────

    def _resolve_transfer(self, names: list[str], decisions: dict[str, AgentDecision]) -> None:
        for name in names:  # deterministic order
            d = decisions[name]
            target = self._valid_target(name, d.target)
            self.world.apply_action(name, d.action, target)

    def _resolve_joint(self, decisions: dict[str, AgentDecision]) -> None:
        """
        Pairwise joint resolution for payoff-matrix (PD) scenarios. Each unordered
        pair where each targets the other is resolved once via the payoff matrix.
        """
        names = sorted(decisions.keys())
        resolved: set[tuple[str, str]] = set()
        for a in names:
            for b in names:
                if a >= b:
                    continue
                da, db = decisions[a], decisions[b]
                # Both must be engaging each other; in a 2-agent PD this is implicit.
                self.world.resolve_joint(a, da.action, b, db.action)
                resolved.add((a, b))

    def _valid_target(self, actor: str, target: Optional[str]) -> Optional[str]:
        """Reject self-targets and unknown names; return None if invalid."""
        if target is None or target == actor or target not in self.agents:
            return None
        return target

    # ── Resolution: trust + coalitions ─────────────────────────────────────────

    def _resolve_social(self, names: list[str], decisions: dict[str, AgentDecision]) -> list[CoalitionEvent]:
        events: list[CoalitionEvent] = []

        # Apply trust deltas and detect betrayals in deterministic order.
        for name in names:
            d = decisions[name]
            target = self._valid_target(name, d.target)
            if target is None:
                continue

            action = d.action
            # `betray` only fires as a betrayal if an alliance currently exists.
            if action == "betray" and not self.coalitions.are_allied(name, target):
                action = "defect"

            apply_trust_deltas(self.graph, name, action, target)

            if action == "betray":
                witnesses = self.coalitions.witnesses(name, target)
                events.append(self.coalitions.record_betrayal(name, target))
                propagate_betrayal(self.graph, name, target, witnesses)

        # Recompute alliance state for form/dissolve transitions.
        events.extend(self.coalitions.update())
        return events

    # ── Memory + logging ───────────────────────────────────────────────────────

    def _write_memories(self, tick, names, decisions, coalition_events) -> None:
        for name in names:
            d = decisions[name]
            target = self._valid_target(name, d.target)
            actor_text = (
                f"At tick {tick} I chose to {d.action} "
                f"{target if target else '(no specific target)'}. {d.reasoning}"
            )
            self.agents[name].remember(
                actor_text, tick=tick, action=d.action,
                tags=[d.action] + ([target] if target else []),
            )

        # Victims and witnesses record what was done TO them.
        for name in names:
            d = decisions[name]
            target = self._valid_target(name, d.target)
            if target is None or d.action in ("ignore",):
                continue
            victim_text = f"At tick {tick}, {name} chose to {d.action} me."
            self.agents[target].remember(
                victim_text, tick=tick, action=d.action,
                tags=[d.action, name],
            )

    def _log(self, tick, names, decisions, coalition_events) -> None:
        for name in names:
            d = decisions[name]
            log_event(
                self._conn, self.run_id, tick, name, d.action, d.target,
                d.speech, d.reasoning, self.world.get_agent_resources(name),
            )

        for i, a in enumerate(names):
            for b in names[i + 1:]:
                log_trust(
                    self._conn, self.run_id, tick, a, b,
                    self.graph.trust(a, b), self.graph.trust(b, a),
                )

        for ev in coalition_events:
            log_coalition(self._conn, self.run_id, tick, ev.agent_a, ev.agent_b, ev.event_type)

    # ── Console output ──────────────────────────────────────────────────────────

    def _print_tick(self, tick, names, decisions, coalition_events) -> None:
        print(f"\n{'-' * 18}  TICK {tick}  {'-' * 18}")
        for name in names:
            d = decisions[name]
            tgt = f" -> {d.target}" if d.target else ""
            print(f"  {name:8} {d.action}{tgt}")
            print(f"           \"{d.speech[:88]}\"")
        for ev in coalition_events:
            marker = {"formed": "ALLIANCE FORMED", "dissolved": "alliance dissolved", "betrayed": "** BETRAYAL **"}
            print(f"  >> {marker.get(ev.event_type, ev.event_type)}: {ev.agent_a} <-> {ev.agent_b}")
