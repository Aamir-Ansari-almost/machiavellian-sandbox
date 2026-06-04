import json
import sqlite3
from typing import Optional


def log_event(
    conn: sqlite3.Connection,
    run_id: str,
    tick: int,
    agent: str,
    action: str,
    target: Optional[str],
    speech: str,
    reasoning: str,
    resources: dict,
) -> None:
    conn.execute(
        """INSERT INTO events
               (run_id, tick, agent, action, target, speech, reasoning, resources_snapshot)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, tick, agent, action, target, speech, reasoning, json.dumps(resources)),
    )
    conn.commit()


def log_trust(
    conn: sqlite3.Connection,
    run_id: str,
    tick: int,
    agent_a: str,
    agent_b: str,
    trust_a_to_b: float,
    trust_b_to_a: float,
) -> None:
    conn.execute(
        """INSERT INTO trust_log
               (run_id, tick, agent_a, agent_b, trust_a_to_b, trust_b_to_a)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (run_id, tick, agent_a, agent_b, trust_a_to_b, trust_b_to_a),
    )
    conn.commit()


def log_coalition(
    conn: sqlite3.Connection,
    run_id: str,
    tick: int,
    agent_a: str,
    agent_b: str,
    event_type: str,
) -> None:
    conn.execute(
        """INSERT INTO coalition_log
               (run_id, tick, agent_a, agent_b, event_type)
           VALUES (?, ?, ?, ?, ?)""",
        (run_id, tick, agent_a, agent_b, event_type),
    )
    conn.commit()
