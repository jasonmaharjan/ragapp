import hashlib
import json
import os

import redis as redis_lib

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", "1800"))  # 30-minutes by default

_client: redis_lib.Redis | None = None


def _get_client() -> redis_lib.Redis:
    global _client
    if _client is None:
        _client = redis_lib.from_url(REDIS_URL, decode_responses=True)
    return _client


def cache_key(query: str) -> str:
    """Normalise the query and return its Redis key."""
    normalised = query.strip().lower()
    digest = hashlib.sha256(normalised.encode()).hexdigest()
    return f"rag:query:{digest}"


def get_cached(query: str) -> dict | None:
    """Return ``{"answer": ..., "sources": [...]}`` for *query*, or ``None`` on miss."""
    try:
        raw = _get_client().get(cache_key(query))
        print(raw)
        return json.loads(raw) if raw else None
    except (redis_lib.RedisError, json.JSONDecodeError):
        print(f"Get Cache error for query {query!r}:")
        return None


def set_cached(query: str, answer: str, sources: list[dict]) -> None:
    """Store answer and sources for query with a TTL of ``CACHE_TTL`` seconds."""
    try:
        payload = json.dumps({"answer": answer, "sources": sources})
        _get_client().setex(cache_key(query), CACHE_TTL, payload)
    except redis_lib.RedisError:
        print(f"Failed to set cache for query {query!r}")
        pass
