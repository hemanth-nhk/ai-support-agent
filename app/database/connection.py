"""
SQLite connection management. A single module owns connection creation so
every repository goes through the same path — no ad-hoc sqlite3.connect()
calls scattered around the codebase.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.config import settings


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    _ensure_parent_dir(settings.database_path)
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    from app.database.models import init_schema
    init_schema(conn)
    return conn


@contextmanager
def get_db():
    """Context-managed connection for use in repositories."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
