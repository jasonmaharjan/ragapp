import os

from cache import get_cached, set_cached
from claude_client import stream_answer
from db import save_query
from embedder import get_embedding
from vector_db import query_db


def retrieve(
    query_text: str,
    k: int = 5,
    alpha: float = 0.5,
) -> list[dict]:
    """
    Embed query_text and return the top-k hybrid search results.
    Return empty list if no documents have been ingested yet.
    """
    embedding = get_embedding(query_text)
    return query_db(query_text, embedding, k=k, alpha=alpha)


def run_query(
    query_text: str,
    k: int = 5,
    alpha: float = 0.5,
    session_id: str | None = None,
) -> dict:
    """Full retrieval pipeline: cache check → retrieve → generate → cache + persist."""
    cached_result = get_cached(query_text)
    if cached_result:
        result = {**cached_result, "query": query_text, "cached": True}
        if session_id:
            save_query(session_id, query_text, result.get("answer", ""), result.get("sources", []), True)
        return result

    sources = retrieve(query_text, k=k, alpha=alpha)

    answer = ""
    if os.environ.get("ANTHROPIC_API_KEY"):
        answer = "".join(stream_answer(query_text, sources))

    set_cached(query_text, answer, sources)
    if session_id:
        save_query(session_id, query_text, answer, sources, False)

    return {"query": query_text, "sources": sources, "cached": False, "answer": answer}
