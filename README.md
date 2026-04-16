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

## Query pipeline

Each request runs through the following steps:

1. **Cache check** — Redis lookup on the normalised query. Cache hit returns immediately with zero embedding, retrieval, or LLM cost.

2. **Embed** — query text is converted to a 384-dim vector via sentence-transformers.

3. **Hybrid retrieval** — BM25 + vector search, fused and ranked.

4. **Source attribution** — every returned chunk includes its source filename, chunk index, and hybrid score so the caller knows exactly which part of which document the result came from.

## Source attribution

The `/query` endpoint returns each chunk with the following schema:

```json
{
  "text": ".......",
  "source": "filename.pdf",
  "chunk_index": 3,
  "score": 0.521
}
```

This makes it possible to cite sources in the answer and prevent hallucination.

## Retrieval evaluation

Execution of `scripts/eval.py` calculates how well the hybrid search retrieves relevant chunks according to four metrics:

**Precision@k** : Of the top-k results, what fraction are relevant? 
**Recall@k** : Of all relevant chunks, what fraction appear in top-k? 
**MRR** : Mean Reciprocal Rank — how high does the first correct result appear?
**NDCG@k** : Normalised Discounted Cumulative Gain. Similar to P@k but penalises relevant results at rank k vs rank 1.

Using these metrics, we can tune `alpha` — the BM25/vector blend ratio. A higher alpha gives weightage to semantic search; lower gives weightage to keyword matching.

Run the built-in smoke test (corpus included, no external data needed):

```bash
python scripts/eval.py
```

## Conversation history

Each query can be associated with a `session_id`. When provided, the query and its retrieved sources are persisted to PostgreSQL so you can retrieve the full history for a session later.

The session ID is caller-defined — any string works (UUID, username, etc.). If omitted, the query still runs but nothing is saved.

Both Redis and Postgres degrade gracefully: the API functions normally if either service is unavailable.

## Local Setup

```bash
make install
make setup-db   # creates ragapp user and database using postgres superuser
make run
make test
```

## Running with Docker

```bash
cp .env.example .env   # fill in ANTHROPIC_API_KEY
docker compose up
```

This starts the API on port 8000, Redis on 6379, and Postgres on 5432. The Postgres user and database are created automatically from the credentials in `docker-compose.yml`. The API creates the `query_history` table on startup.

## API

`GET`
- `/` : Health check
- `/history/{session_id}` : Return all queries for a session, oldest first

`POST`
- `/ingest` : Upload a document (PDF / DOCX / TXT / MD)
- `/query` : Ask a question — returns top-k chunks with source attribution

Optional `session_id` in the query request body saves the query to conversation history:

```json
{
  "query": "What is RAG?",
  "k": 5,
  "alpha": 0.5,
  "session_id": "user-abc"
}
```