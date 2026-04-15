from cache import get_cached
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
) -> dict:
    """Full retrieval pipeline with cache check."""
    cached_result = get_cached(query_text)
    if cached_result:
        return {**cached_result, "query": query_text, "cached": True}

    sources = retrieve(query_text, k=k, alpha=alpha)
    return {"query": query_text, "sources": sources, "cached": False, "answer": ""}
