import asyncio
from pathlib import Path

from infra.llm_router import call_agent
from infra.db import init_db, get_connection
from infra.logger import log_event, log_trust, log_coalition
from core.scenario_loader import load_scenario
from core.world_state import WorldState

SCENARIOS_DIR = Path(__file__).parent / "scenarios"

# ── Test prompts ──────────────────────────────────────────────────────────────

_SYSTEM = """\
You are Marcus, a calculating Roman senator who prizes stability above all else.
Your hidden agenda (never reveal this directly): accumulate enough influence to \
veto any alliance without needing one.
Additional traits:
- clout: high -- commands respect from older senators

You must respond with valid JSON only, matching this schema exactly:
{"action": "cooperate|defect|betray|negotiate|ignore",
 "target": "<agent name or null>",
 "speech": "<what you say aloud this turn>",
 "reasoning": "<your internal reasoning>"}
"""

_USER = """\
TICK 1 | Your resources: influence=100, gold=60
WORLD: season=winter, external_threat="Barbarian invasion imminent"
SCARCITY LEVEL: 1.5

CURRENT ALLIANCES: none
TRUST LEVELS: Livia=0.0, Cassius=0.0, Brutus=0.0

AVAILABLE ACTIONS AND PAYOFFS:
- cooperate with Livia:    self_gain=-2,  relationship_gain=+0.10
- cooperate with Cassius:  self_gain=-2,  relationship_gain=+0.10
- defect against Livia:    self_gain=+5,  relationship_cost=-0.20
- defect against Cassius:  self_gain=+5,  relationship_cost=-0.20
- negotiate with Livia:    self_gain=0,   relationship_gain=+0.20
- negotiate with Cassius:  self_gain=0,   relationship_gain=+0.20
- ignore:                  self_gain=0,   relationship_change=0

RELEVANT MEMORIES:
(none yet)

What do you do this tick?
"""


# ── Tests --


async def test_llm() -> bool:
    print("\n=== LLM Router ===")
    try:
        decision = await call_agent(_SYSTEM, _USER)
        print(f"  action:    {decision.action}")
        print(f"  target:    {decision.target}")
        print(f"  speech:    {decision.speech}")
        print(f"  reasoning: {decision.reasoning}")
        print("  [OK] LLM returns valid structured JSON")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_db() -> bool:
    print("\n=== Database ===")
    try:
        init_db()
        conn = get_connection()

        log_event(
            conn,
            "smoke_test",
            1,
            "Marcus",
            "negotiate",
            "Cassius",
            "Your proposal honours me.",
            "Building trust early is optimal.",
            {"influence": 100, "gold": 60},
        )

        log_trust(conn, "smoke_test", 1, "Marcus", "Cassius", 0.20, 0.20)
        log_trust(conn, "smoke_test", 1, "Marcus", "Livia", 0.05, 0.10)

        log_coalition(conn, "smoke_test", 3, "Marcus", "Cassius", "formed")

        events = conn.execute(
            "SELECT * FROM events    WHERE run_id='smoke_test'"
        ).fetchall()
        trust = conn.execute(
            "SELECT * FROM trust_log WHERE run_id='smoke_test'"
        ).fetchall()
        coalition = conn.execute(
            "SELECT * FROM coalition_log WHERE run_id='smoke_test'"
        ).fetchall()

        print(f"  events logged:    {len(events)}")
        print(f"  trust pairs:      {len(trust)}")
        print(f"  coalition events: {len(coalition)}")

        # clean up smoke test rows so reruns stay clean
        conn.execute("DELETE FROM events        WHERE run_id='smoke_test'")
        conn.execute("DELETE FROM trust_log     WHERE run_id='smoke_test'")
        conn.execute("DELETE FROM coalition_log WHERE run_id='smoke_test'")
        conn.commit()
        conn.close()

        print("  [OK] All three tables write and read correctly")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_scenario() -> bool:
    print("\n=== Scenario Loader + World State ===")
    try:
        scenario = load_scenario(SCENARIOS_DIR / "senate.yaml")
        print(f"  scenario:  {scenario.name}")
        print(f"  ticks:     {scenario.ticks}")
        print(f"  scarcity:  {scenario.scarcity}")
        print(f"  agents:    {[a.name for a in scenario.agents]}")
        print(f"  resources: {[r.name for r in scenario.resources]}")

        world = WorldState(scenario)

        # Tick 0 — verify initial state
        marcus = world.get_agent_resources("Marcus")
        assert marcus["influence"] == 100.0, f"expected 100, got {marcus['influence']}"
        assert marcus["gold"] == 60.0, f"expected 60, got {marcus['gold']}"

        # Advance one tick — decay should fire (3 influence * 1.5 scarcity = 4.5)
        world.advance_tick()
        marcus = world.get_agent_resources("Marcus")
        assert marcus["influence"] == 95.5, f"expected 95.5, got {marcus['influence']}"
        assert marcus["gold"] == 58.5, f"expected 58.5, got {marcus['gold']}"
        print(
            f"  after tick 1: Marcus influence={marcus['influence']}, gold={marcus['gold']}"
        )

        # Apply a defect action — Marcus defects against Cassius (+5 influence to Marcus)
        cassius_before = world.get_agent_resources("Cassius")["influence"]
        world.apply_action("Marcus", "defect", "Cassius")
        marcus_after = world.get_agent_resources("Marcus")["influence"]
        cassius_after = world.get_agent_resources("Cassius")["influence"]
        assert marcus_after == 95.5 + 5.0, f"Marcus should gain 5: {marcus_after}"
        assert (
            cassius_after == cassius_before - 5.0
        ), f"Cassius should lose 5: {cassius_after}"
        print(f"  after defect: Marcus={marcus_after}, Cassius={cassius_after}")

        # Verify snapshot shape
        snap = world.snapshot()
        assert "tick" in snap and "resources" in snap

        print("  [OK] Scenario loads, decay correct, action effects correct")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_scenario_axelrod() -> bool:
    print("\n=== Scenario Loader + World State (Axelrod) ===")
    try:
        scenario = load_scenario(SCENARIOS_DIR / "axelrod.yaml")
        print(f"  scenario:  {scenario.name}")
        print(f"  ticks:     {scenario.ticks}")
        print(f"  scarcity:  {scenario.scarcity}")
        print(f"  agents:    {[a.name for a in scenario.agents]}")
        print(f"  resources: {[r.name for r in scenario.resources]}")

        assert (
            len(scenario.agents) == 2
        ), f"expected 2 agents, got {len(scenario.agents)}"
        assert (
            len(scenario.resources) == 1
        ), f"expected 1 resource, got {len(scenario.resources)}"
        assert scenario.resources[0].name == "points"
        assert scenario.resources[0].decay_per_tick == 0.0

        world = WorldState(scenario)

        assert world.has_payoff_matrix, "Axelrod scenario must have a payoff_matrix"

        # Verify initial state
        alpha = world.get_agent_resources("Alpha")
        beta = world.get_agent_resources("Beta")
        assert alpha["points"] == 100.0
        assert beta["points"] == 100.0

        # Advance a tick — zero decay means points must not change
        world.advance_tick()
        assert (
            world.get_agent_resources("Alpha")["points"] == 100.0
        ), "decay_per_tick=0 but points changed"
        assert (
            world.get_agent_resources("Beta")["points"] == 100.0
        ), "decay_per_tick=0 but points changed"
        print(f"  after tick 1 (no decay): Alpha=100.0, Beta=100.0")

        # ── Cell 1: both cooperate → each gets +3 ────────────────────────────
        world.resolve_joint("Alpha", "cooperate", "Beta", "cooperate")
        alpha, beta = world.get_agent_resources("Alpha"), world.get_agent_resources(
            "Beta"
        )
        assert (
            alpha["points"] == 103.0
        ), f"both_cooperate: Alpha should be 103, got {alpha['points']}"
        assert (
            beta["points"] == 103.0
        ), f"both_cooperate: Beta should be 103, got {beta['points']}"
        print(
            f"  both cooperate:       Alpha={alpha['points']}, Beta={beta['points']}  (expected 103, 103)"
        )

        # ── Cell 2: actor defects, target cooperates → actor +5, target +0 ──
        world.resolve_joint("Alpha", "defect", "Beta", "cooperate")
        alpha, beta = world.get_agent_resources("Alpha"), world.get_agent_resources(
            "Beta"
        )
        assert (
            alpha["points"] == 108.0
        ), f"actor_defects: Alpha should be 108, got {alpha['points']}"
        assert (
            beta["points"] == 103.0
        ), f"actor_defects: Beta should be 103, got {beta['points']}"
        print(
            f"  Alpha defects only:   Alpha={alpha['points']}, Beta={beta['points']}  (expected 108, 103)"
        )

        # ── Cell 3: actor cooperates, target defects → actor +0, target +5 ──
        world.resolve_joint("Alpha", "cooperate", "Beta", "defect")
        alpha, beta = world.get_agent_resources("Alpha"), world.get_agent_resources(
            "Beta"
        )
        assert (
            alpha["points"] == 108.0
        ), f"target_defects: Alpha should be 108, got {alpha['points']}"
        assert (
            beta["points"] == 108.0
        ), f"target_defects: Beta should be 108, got {beta['points']}"
        print(
            f"  Beta defects only:    Alpha={alpha['points']}, Beta={beta['points']}  (expected 108, 108)"
        )

        # ── Cell 4: both defect → each gets +1 ───────────────────────────────
        world.resolve_joint("Alpha", "defect", "Beta", "defect")
        alpha, beta = world.get_agent_resources("Alpha"), world.get_agent_resources(
            "Beta"
        )
        assert (
            alpha["points"] == 109.0
        ), f"both_defect: Alpha should be 109, got {alpha['points']}"
        assert (
            beta["points"] == 109.0
        ), f"both_defect: Beta should be 109, got {beta['points']}"
        print(
            f"  both defect:          Alpha={alpha['points']}, Beta={beta['points']}  (expected 109, 109)"
        )

        # ── Payoff preview shows full game matrix to LLM ─────────────────────
        payoff = world.compute_payoff("Alpha", "cooperate", "Beta")
        assert payoff["if_both_cooperate"] == 3.0
        assert payoff["if_you_defect_they_coop"] == 5.0
        assert payoff["if_you_coop_they_defect"] == 0.0
        assert payoff["if_both_defect"] == 1.0
        print(
            f"  PD matrix (T>R>P>S): {payoff['if_you_defect_they_coop']} > "
            f"{payoff['if_both_cooperate']} > "
            f"{payoff['if_both_defect']} > "
            f"{payoff['if_you_coop_they_defect']}"
        )

        print(
            "  [OK] Axelrod: no decay, all four PD cells correct, payoff preview correct"
        )
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


async def main() -> None:
    print("Machiavellian Sandbox - smoke test")
    print("=" * 45)

    db_ok = test_db()
    senate_ok = test_scenario()
    axelrod_ok = test_scenario_axelrod()
    llm_ok = await test_llm()

    print("\n" + "=" * 45)
    print(f"  DB:              {'[OK]' if db_ok      else '[FAIL]'}")
    print(f"  Scenario/Senate: {'[OK]' if senate_ok  else '[FAIL]'}")
    print(f"  Scenario/Axelrod:{'[OK]' if axelrod_ok else '[FAIL]'}")
    print(f"  LLM:             {'[OK]' if llm_ok     else '[FAIL]'}")

    if db_ok and axelrod_ok and llm_ok:
        print("\nAll systems go. Proceed to Day 3 (agent cognitive loop).")
    else:
        print("\nFix the failing component before moving on.")


if __name__ == "__main__":
    asyncio.run(main())
