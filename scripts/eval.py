#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

from chunker import chunk_text
from embedder import get_embedding
from query import retrieve
from vector_db import add_documents


def precision_at_k(retrieved: list[dict], relevant: set[tuple], k: int) -> float:
    hits = sum(1 for r in retrieved[:k] if (r["source"], r["chunk_index"]) in relevant)
    return hits / k if k else 0.0


def recall_at_k(retrieved: list[dict], relevant: set[tuple], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if (r["source"], r["chunk_index"]) in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[dict], relevant: set[tuple]) -> float:
    for i, r in enumerate(retrieved, start=1):
        if (r["source"], r["chunk_index"]) in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[dict], relevant: set[tuple], k: int) -> float:
    def dcg(hits: list[int]) -> float:
        return sum(h / math.log2(i + 2) for i, h in enumerate(hits))

    gains = [1 if (r["source"], r["chunk_index"]) in relevant else 0 for r in retrieved[:k]]
    ideal = sorted(gains, reverse=True)
    actual_dcg = dcg(gains)
    ideal_dcg = dcg(ideal)
    return actual_dcg / ideal_dcg if ideal_dcg else 0.0


def evaluate(ground_truth: list[dict], k: int = 5) -> dict:
    p_scores, r_scores, mrr_scores, ndcg_scores = [], [], [], []

    for item in ground_truth:
        query = item["query"]
        relevant = {(r["source"], r["chunk_index"]) for r in item["relevant"]}
        results = retrieve(query, k=k)

        p = precision_at_k(results, relevant, k)
        r = recall_at_k(results, relevant, k)
        rr = reciprocal_rank(results, relevant)
        nd = ndcg_at_k(results, relevant, k)

        p_scores.append(p)
        r_scores.append(r)
        mrr_scores.append(rr)
        ndcg_scores.append(nd)

        print(f"  Q: {query[:60]!r}")
        for i, res in enumerate(results[:k], 1):
            hit = "✓" if (res["source"], res["chunk_index"]) in relevant else " "
            print(f"    [{i}]{hit} score={res['score']}  {res['source']}[{res['chunk_index']}]")
        print(f"    → P@{k}={p:.3f}  R@{k}={r:.3f}  RR={rr:.3f}  NDCG@{k}={nd:.3f}\n")

    n = len(ground_truth)
    return {
        f"precision@{k}": round(sum(p_scores) / n, 4),
        f"recall@{k}": round(sum(r_scores) / n, 4),
        "MRR": round(sum(mrr_scores) / n, 4),
        f"NDCG@{k}": round(sum(ndcg_scores) / n, 4),
        "queries_evaluated": n,
    }


# Built-in smoke test (can use external data with --data)
_SMOKE_CORPUS = {
    "rag.txt": (
        "Retrieval-Augmented Generation (RAG) combines a retrieval step with a "
        "language model to produce grounded answers.\n\n"
        "The retrieval step uses vector similarity search to find relevant document chunks.\n\n"
        "The generation step feeds those chunks as context to the LLM."
    ),
    "bm25.txt": (
        "BM25 is a probabilistic keyword ranking function used in information retrieval.\n\n"
        "It improves on TF-IDF by normalising for document length.\n\n"
        "Hybrid search combines BM25 with vector search for better coverage."
    ),
}

# Each smoke query to map to a specific chunk in the corpus for evaluation purposes
_SMOKE_QUERIES = [
    {
        "query": "How does RAG use retrieval?",
        "relevant": [{"source": "rag.txt", "chunk_index": 0}],
    },
    {
        "query": "What is BM25 and how does it work?",
        "relevant": [{"source": "bm25.txt", "chunk_index": 0}],
    },
    {
        "query": "What is hybrid search?",
        "relevant": [{"source": "bm25.txt", "chunk_index": 1}],
    },
]


def _run_smoke_test(k: int) -> dict:
    print("Ingesting smoke-test corpus...\n")
    for filename, text in _SMOKE_CORPUS.items():
        # max_chars=100 forces each paragraph into its own chunk
        chunks = chunk_text(text, source=filename, max_chars=100, overlap_chars=30)
        texts = [c.text for c in chunks]
        metas = [{"source": c.source, "chunk_index": c.index} for c in chunks]
        add_documents(texts, [get_embedding(t) for t in texts], metas)
        print(f"  {filename}: {len(chunks)} chunks")

    print(f"\nRunning {len(_SMOKE_QUERIES)} queries at k={k}...\n")
    return evaluate(_SMOKE_QUERIES, k=k)


def main():
    parser = argparse.ArgumentParser(description="RAG retrieval evaluation")
    parser.add_argument("--data", help="Path to ground truth JSONL file")
    parser.add_argument("--k", type=int, default=5, help="Rank cutoff (default: 5)")
    args = parser.parse_args()

    if args.data:
        path = Path(args.data)
        if not path.exists():
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)
        ground_truth = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        print(f"Loaded {len(ground_truth)} queries from {path}\n")
        summary = evaluate(ground_truth, k=args.k)
    else:
        print("No --data file provided. Running built-in smoke test.\n")
        summary = _run_smoke_test(args.k)

    print("=" * 45)
    print("Summary")
    print("=" * 45)
    for metric, value in summary.items():
        print(f"  {metric:<20} {value}")


if __name__ == "__main__":
    main()