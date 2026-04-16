import json
import os

import psycopg2
import psycopg2.extras
from psycopg2 import pool

DB_URL = os.environ.get("DB_URL", "postgresql://ragapp:ragapp@localhost:5432/ragapp")

_pool: pool.SimpleConnectionPool | None = None


def _get_pool() -> pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = pool.SimpleConnectionPool(1, 10, DB_URL)
    return _pool


def init_db() -> None:
    """Create the query_history table if it doesn't exist."""
    conn = _get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS query_history (
                    id          SERIAL PRIMARY KEY,
                    session_id  TEXT        NOT NULL,
                    query       TEXT        NOT NULL,
                    answer      TEXT        NOT NULL DEFAULT '',
                    sources     JSONB       NOT NULL DEFAULT '[]',
                    cached      BOOLEAN     NOT NULL DEFAULT FALSE,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            # Index to speed up retrieval of session history
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_history_session
                ON query_history (session_id, created_at DESC)
            """)
        conn.commit()
    finally:
        _get_pool().putconn(conn)


def save_query(
    session_id: str,
    query: str,
    answer: str,
    sources: list[dict],
    cached: bool,
) -> None:
    """Persist query results."""
    try:
        conn = _get_pool().getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_history (session_id, query, answer, sources, cached)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (session_id, query, answer, json.dumps(sources), cached),
                )
            conn.commit()
        finally:
            _get_pool().putconn(conn)
    except Exception:
        pass


def get_history(session_id: str) -> list[dict]:
    """Return all queries for a session, oldest first. Empty list on error."""
    try:
        conn = _get_pool().getconn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, query, answer, sources, cached, created_at
                    FROM query_history
                    WHERE session_id = %s
                    ORDER BY created_at ASC
                    """,
                    (session_id,),
                )
                rows = cur.fetchall()
                return [dict(r) for r in rows]
        finally:
            _get_pool().putconn(conn)
    except Exception:
        return []
