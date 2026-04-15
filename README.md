# ragapp

Document Q&A system built on RAG (Retrieval-Augmented Generation). Upload documents, ask questions, get answers grounded in the source material.

## Stack

- **FastAPI** — API server
- **sentence-transformers** — local embeddings (`all-MiniLM-L6-v2`, cached offline)
- **ChromaDB** — vector store
- **BM25** — keyword search fused with vector scores for hybrid retrieval

## Chunking strategy

Text is split on paragraph boundaries (double newlines) rather than a raw character count. When the accumulated text buffer exceeds a `max_chars`, it is saved as a chunk and the next one starts with the last `overlap_chars` characters of the previous chunk to provide cross-boundary context.

## Hybrid retrieval

Each query runs two searches in parallel (Following the weighted score summation method):
- **Semantic** — cosine similarity via ChromaDB embeddings
- **Lexical / Keyword** — BM25 scoring over the same corpus

Scores are normalised to [0, 1] and combined: 

`score = alpha·semantic + (1-alpha)·BM25`.

where, `alpha` defaults to 0.5.

## Running

```bash
make install
make run      
make test     
```

## Redis query caching

Redis survives restarts, is shared across multiple API workers, and gives us TTL expiry for free so stale answers eventually purge themselves.

The query text is normalised before hashing so variations like extra whitespace or capitalisation still hit the same cache entry. SHA-256 is chosen as the collision-free, fixed-width key in Redis to avoid having to store raw user input data.
