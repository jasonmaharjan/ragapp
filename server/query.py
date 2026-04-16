from cache import get_cached
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
    """Full retrieval pipeline with cache check. Persists to DB if session_id given."""
    cached_result = get_cached(query_text)
    if cached_result:
        result = {**cached_result, "query": query_text, "cached": True}
        if session_id:
            save_query(session_id, query_text, result.get("answer", ""), result.get("sources", []), True)
        return result

    sources = retrieve(query_text, k=k, alpha=alpha)
    result = {"query": query_text, "sources": sources, "cached": False, "answer": ""}
    if session_id:
        save_query(session_id, query_text, "", sources, False)
    return result
