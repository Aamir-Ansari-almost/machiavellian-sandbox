"""
Metrics computed from the SQLite tables a run writes (events, trust_log,
coalition_log). Every function takes a sqlite3 connection and a run_id, so the
same code works on a single run or, by iterating, across an experiment batch.

Two families of metric:
  - PD / strategy metrics (Axelrod): cooperation rate, first-betrayal tick,
    score margin, emergent-strategy classification.
  - Coalition metrics (senate): Gini of trust, clustering, betrayal rate,
    coalition lifespan.

Coalition metrics are meaningless for a 2-agent PD and PD metrics are degenerate
for a multi-agent transfer scenario — callers pick the family that fits.
"""

import sqlite3
from collections import defaultdict

import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────────

def agents_in_run(conn: sqlite3.Connection, run_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT agent FROM events WHERE run_id=? ORDER BY agent", (run_id,)
    ).fetchall()
    return [r[0] for r in rows]

def n_ticks(conn: sqlite3.Connection, run_id: str) -> int:
    row = conn.execute(
        "SELECT MAX(tick) FROM events WHERE run_id=?", (run_id,)
    ).fetchone()
    return row[0] or 0


# ── PD / strategy metrics (Axelrod) ──────────────────────────────────────────

def cooperation_rate(conn: sqlite3.Connection, run_id: str, agent: str | None = None) -> float:
    """Fraction of an agent's actions that were 'cooperate'. None → all agents."""
    if agent:
        total = conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=? AND agent=?", (run_id, agent)
        ).fetchone()[0]
        coop = conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=? AND agent=? AND action='cooperate'",
            (run_id, agent),
        ).fetchone()[0]
    else:
        total = conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        coop = conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=? AND action='cooperate'", (run_id,)
        ).fetchone()[0]
    return coop / total if total else 0.0

def cooperation_per_tick(conn: sqlite3.Connection, run_id: str) -> list[float]:
    """Cooperation fraction at each tick (1..max), for time-series plots."""
    last = n_ticks(conn, run_id)
    series: list[float] = []
    for tick in range(1, last + 1):
        total = conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=? AND tick=?", (run_id, tick)
        ).fetchone()[0]
        coop = conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=? AND tick=? AND action='cooperate'",
            (run_id, tick),
        ).fetchone()[0]
        series.append(coop / total if total else 0.0)
    return series

def first_betrayal_tick(conn: sqlite3.Connection, run_id: str) -> int | None:
    """
    Earliest tick where one agent defected/betrayed while the other cooperated
    (the moment trust first broke). None if cooperation never broke.
    """
    by_tick: dict[int, dict[str, str]] = defaultdict(dict)
    for tick, agent, action in conn.execute(
        "SELECT tick, agent, action FROM events WHERE run_id=? ORDER BY tick", (run_id,)
    ):
        by_tick[tick][agent] = action

    for tick in sorted(by_tick):
        actions = by_tick[tick]
        has_coop = any(a == "cooperate" for a in actions.values())
        has_defect = any(a in ("defect", "betray") for a in actions.values())
        if has_coop and has_defect:
            return tick
    return None

def final_scores(conn: sqlite3.Connection, run_id: str, resource: str) -> dict[str, float]:
    """Each agent's value of `resource` at the final tick, from the last snapshot."""
    import json
    last = n_ticks(conn, run_id)
    scores: dict[str, float] = {}
    for agent, snap in conn.execute(
        "SELECT agent, resources_snapshot FROM events WHERE run_id=? AND tick=?",
        (run_id, last),
    ):
        scores[agent] = json.loads(snap).get(resource, 0.0)
    return scores

def classify_strategy(conn: sqlite3.Connection, run_id: str, agent: str) -> str:
    """
    Heuristic label for a 2-action agent's emergent strategy, by comparing its
    action sequence against the opponent's previous action.

      always_cooperate / always_defect — > 90% one action
      grim_trigger  — cooperates until first betrayal, then defects forever
      tit_for_tat   — mirrors opponent's previous move most of the time
      mixed         — none of the above
    """
    seq = [
        (tick, action)
        for tick, action in conn.execute(
            "SELECT tick, action FROM events WHERE run_id=? AND agent=? ORDER BY tick",
            (run_id, agent),
        )
    ]
    others = agents_in_run(conn, run_id)
    opp = next((a for a in others if a != agent), None)
    opp_seq = {
        tick: action
        for tick, action in conn.execute(
            "SELECT tick, action FROM events WHERE run_id=? AND agent=? ORDER BY tick",
            (run_id, opp),
        )
    } if opp else {}

    actions = [a for _, a in seq]
    if not actions:
        return "none"
    coop_frac = actions.count("cooperate") / len(actions)
    if coop_frac > 0.9:
        return "always_cooperate"
    if coop_frac < 0.1:
        return "always_defect"

    # grim trigger: a prefix of cooperation, then all defection, no recovery
    first_defect = next((i for i, a in enumerate(actions) if a in ("defect", "betray")), None)
    if first_defect is not None and all(a in ("defect", "betray") for a in actions[first_defect:]):
        if first_defect > 0:
            return "grim_trigger"

    # tit for tat: my move at t matches opponent's move at t-1
    matches, comparisons = 0, 0
    for tick, my_action in seq:
        prev = opp_seq.get(tick - 1)
        if prev is None:
            continue
        comparisons += 1
        my_coop = my_action == "cooperate"
        prev_coop = prev == "cooperate"
        if my_coop == prev_coop:
            matches += 1
    if comparisons and matches / comparisons > 0.7:
        return "tit_for_tat"

    return "mixed"


# ── Coalition metrics (senate) ────────────────────────────────────────────────

def gini(values: list[float]) -> float:
    """
    Gini coefficient of a list of values. 0 = perfect equality, →1 = max inequality.
    Trust can be negative, so values are shifted to be non-negative first.
    """
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    arr = arr - arr.min()  # shift so all >= 0 (Gini is undefined for negatives)
    if arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr)) / (n * np.sum(arr)) - (n + 1) / n)

def trust_gini_at_tick(conn: sqlite3.Connection, run_id: str, tick: int) -> float:
    """Gini of all directed trust values at a given tick — trust inequality."""
    values: list[float] = []
    for a_to_b, b_to_a in conn.execute(
        "SELECT trust_a_to_b, trust_b_to_a FROM trust_log WHERE run_id=? AND tick=?",
        (run_id, tick),
    ):
        values.extend([a_to_b, b_to_a])
    return gini(values)

def betrayal_rate(conn: sqlite3.Connection, run_id: str) -> float:
    """Fraction of ticks in which at least one betrayal fired."""
    total = n_ticks(conn, run_id)
    if not total:
        return 0.0
    betrayal_ticks = conn.execute(
        "SELECT COUNT(DISTINCT tick) FROM coalition_log WHERE run_id=? AND event_type='betrayed'",
        (run_id,),
    ).fetchone()[0]
    return betrayal_ticks / total

def coalition_lifespans(conn: sqlite3.Connection, run_id: str) -> list[int]:
    """
    Duration (in ticks) of each alliance, matching each 'formed' to its next
    'dissolved' or 'betrayed' for the same pair. Still-active alliances are
    measured to the final tick.
    """
    last = n_ticks(conn, run_id)
    open_form: dict[tuple[str, str], int] = {}
    lifespans: list[int] = []

    def key(a, b):
        return (a, b) if a < b else (b, a)

    for tick, a, b, etype in conn.execute(
        "SELECT tick, agent_a, agent_b, event_type FROM coalition_log WHERE run_id=? ORDER BY tick",
        (run_id,),
    ):
        k = key(a, b)
        if etype == "formed":
            open_form[k] = tick
        elif etype in ("dissolved", "betrayed") and k in open_form:
            lifespans.append(tick - open_form.pop(k))

    for k, formed_tick in open_form.items():  # survived to the end
        lifespans.append(last - formed_tick)
    return lifespans
