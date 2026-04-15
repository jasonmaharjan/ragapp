"""Vector store with hybrid retrieval: semantic (ChromaDB) + lexical/keyword (BM25).

Hybrid scoring formula:
    score = alpha * semantic_score + (1 - alpha) * bm25_score

Both sub-scores are normalised to [0, 1] before combining, so alpha is an
intuitive dial: 1.0 = pure vector, 0.0 = pure BM25, 0.5 = equal weight.
"""

import chromadb
from rank_bm25 import BM25Okapi


class VectorStore:
    def __init__(self) -> None:
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection("docs")
        # In-memory BM25 corpus mirrors the ChromaDB collection order
        self._corpus: list[str] = []
        self._metadatas: list[dict] = []
        self._bm25: BM25Okapi | None = None

    # Indexing
    def add_documents(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        start_id = len(self._corpus)
        ids = [str(start_id + i) for i in range(len(chunks))]
        metas = metadatas or [{}] * len(chunks)

        # Add to ChromaDB collection (first to ensure we have IDs for BM25)
        self._collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metas,
        )
        self._corpus.extend(chunks)
        self._metadatas.extend(metas)
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        tokenised = [doc.lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenised)

    # Retrieval
    def hybrid_query(
        self,
        query_text: str,
        query_embedding: list[float],
        k: int = 5,
        alpha: float = 0.5,
    ) -> list[dict]:
        """Return the *k* best chunks by combined semantic + lexical BM25 score.

        Each resulting dict has keys: ``text``, ``source``, ``chunk_index``,
        ``score``.
        """
        n = len(self._corpus)
        if n == 0:
            return []

        top_n = min(k * 3, n)  # over-fetch for better score normalisation

        # Semantic scores (L2 distance → similarity)
        vector_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n,
            include=["documents", "metadatas", "distances"],
        )
        distances = vector_results["distances"][0]
        ids = vector_results["ids"][0]
        max_dist = max(distances) if distances else 1.0
        semantic_map: dict[str, float] = {
            doc_id: 1.0 - (dist / max_dist)
            for doc_id, dist in zip(ids, distances)
        }

        # BM25 scores
        tokenised_query = query_text.lower().split()
        raw_bm25 = self._bm25.get_scores(tokenised_query)  # shape (n,)
        bm25_max = float(raw_bm25.max()) if raw_bm25.max() > 0 else 1.0
        norm_bm25 = raw_bm25 / bm25_max

        # Combine & calculate rank 
        scores: list[tuple[str, float]] = [
            (str(i), alpha * semantic_map.get(str(i), 0.0) + (1.0 - alpha) * float(norm_bm25[i]))
            for i in range(n)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_ids = [doc_id for doc_id, _ in scores[:k]]
        score_lookup = {doc_id: s for doc_id, s in scores[:k]}

        fetched = self._collection.get(ids=top_ids, include=["documents", "metadatas"])
        results = [
            {
                "text": text,
                "source": meta.get("source", ""),
                "chunk_index": meta.get("chunk_index", int(doc_id)),
                "score": round(score_lookup[doc_id], 4),
            }
            for doc_id, text, meta in zip(
                fetched["ids"], fetched["documents"], fetched["metadatas"]
            )
        ]
        results.sort(key=lambda r: r["score"], reverse=True)
        return results


# Module-level singleton
# other modules import the methods of VectorStore directly
vector_store = VectorStore()


def add_documents(
    chunks: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict] | None = None,
) -> None:
    vector_store.add_documents(chunks, embeddings, metadatas)


def query_db(
    query_text: str,
    query_embedding: list[float],
    k: int = 5,
    alpha: float = 0.5,
) -> list[dict]:
    return vector_store.hybrid_query(query_text, query_embedding, k, alpha)
