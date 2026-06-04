import asyncio

from infra.llm_router import call_agent
from infra.db import init_db, get_connection
from infra.logger import log_event, log_trust, log_coalition

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


# ? test llm
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

# ? test db
def test_db() -> bool:
    print("\n=== Database ===")
    try:
        init_db()
        conn = get_connection()

        log_event(conn, "smoke_test", 1, "Marcus", "negotiate", "Cassius",
                  "Your proposal honours me.", "Building trust early is optimal.",
                  {"influence": 100, "gold": 60})

        log_trust(conn, "smoke_test", 1, "Marcus", "Cassius", 0.20, 0.20)
        log_trust(conn, "smoke_test", 1, "Marcus", "Livia",   0.05, 0.10)

        log_coalition(conn, "smoke_test", 3, "Marcus", "Cassius", "formed")

        events    = conn.execute("SELECT * FROM events    WHERE run_id='smoke_test'").fetchall()
        trust     = conn.execute("SELECT * FROM trust_log WHERE run_id='smoke_test'").fetchall()
        coalition = conn.execute("SELECT * FROM coalition_log WHERE run_id='smoke_test'").fetchall()

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


async def main() -> None:
    print("Machiavellian Sandbox - Day 1/2 smoke test")
    print("=" * 45)

    db_ok  = test_db()
    llm_ok = await test_llm()

    print("\n" + "=" * 45)
    print(f"  DB:  {'[OK]'   if db_ok  else '[FAIL]'}")
    print(f"  LLM: {'[OK]'   if llm_ok else '[FAIL]'}")

    if db_ok and llm_ok:
        print("\nAll systems go. Proceed to Day 3 (agent cognitive loop).")
    else:
        print("\nFix the failing component before moving on.")


if __name__ == "__main__":
    asyncio.run(main())
