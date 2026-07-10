import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).parent.parent / "simulation.db"


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create all tables if they don't exist. Safe to call multiple times."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id              TEXT    NOT NULL,
            tick                INTEGER NOT NULL,
            agent               TEXT    NOT NULL,
            action              TEXT    NOT NULL,
            target              TEXT,
            speech              TEXT,
            reasoning           TEXT,
            resources_snapshot  TEXT
        );

        CREATE TABLE IF NOT EXISTS trust_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id       TEXT    NOT NULL,
            tick         INTEGER NOT NULL,
            agent_a      TEXT    NOT NULL,
            agent_b      TEXT    NOT NULL,
            trust_a_to_b REAL    NOT NULL,
            trust_b_to_a REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS coalition_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id     TEXT    NOT NULL,
            tick       INTEGER NOT NULL,
            agent_a    TEXT    NOT NULL,
            agent_b    TEXT    NOT NULL,
            event_type TEXT    NOT NULL
        );
    """)
    conn.commit()
    conn.close()


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn
