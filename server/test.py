"""Manual Integration tests for components"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1" # Offline mode for Hugging Face models (e.g. sentence-transformers) 

from chunker import chunk_text
from embedder import get_embedding
from ingestion import ingest_document
from vector_db import add_documents, query_db


# 1. Chunker
print("---- 1. Chunker ----") 
sample_text = (
    "Retrieval-Augmented Generation (RAG) is a technique that combines retrieval "
    "with language model generation.\n\n"
    "The retrieval step fetches relevant documents from a knowledge base using "
    "vector similarity search.\n\n"
    "HERE IS A RANDOM FACT TO TEST CHUNKING: The flight from Sydney to Melbourne takes about 1 hour!\n\n"
    "The generation step feeds those documents as context to a language model, "
    "which then produces a grounded answer.\n\n"
)

chunks = chunk_text(sample_text, source="sample.txt", max_chars=100, overlap_chars=50)
print(f"  Produced a total of {len(chunks)} chunks")
for c in chunks:
    print(f"  [{c.index}] {c.text[:60]!r}... (Length: {len(c.text)} chars) ")
assert len(chunks) >= 2, "Expected multiple chunks from give paragraph text"
print("  PASS\n")


# 2. Embedder
print("---- 2. Embedder ----")
query = "What is Retrieval-Augmented Generation?"
emb = get_embedding(query)
# Check that we get a non-empty list of embedding vectors (number array)
assert isinstance(emb, list) and len(emb) > 0
print(f"  Embedding dim: {len(emb)}")
print("  PASS\n")


# 3. Ingestion (TXT file)
print("---- 3. Ingestion (TXT) ----") 
import tempfile, pathlib

tmp = pathlib.Path(tempfile.mktemp(suffix=".txt"))
tmp.write_text(sample_text, encoding="utf-8")
doc_chunks = ingest_document(str(tmp), max_chars=100, overlap_chars=50)
tmp.unlink()

print(f"  Ingested {len(doc_chunks)} chunks from TXT")
assert len(doc_chunks) >= 2
# Check that source filename is correctly attached to the chunks
assert doc_chunks[0].source == tmp.name
assert doc_chunks[-1].source == tmp.name
print(f"  Correctly attached temporary source filename to chunks: {doc_chunks[0].source}")
print("  PASS\n")


# 4. Hybrid Search (vector + BM25)
print("---- 4. Hybrid Search (vector + BM25) ----")
test_docs = [
    "HELLO WORLD! This is a test document.",
    "Python is a popular programming language for machine learning.",
    "RAG combines retrieval with generation to reduce hallucinations.",
    "BM25 is a classic keyword-based ranking function used in search engines.",
]
metas = [{"source": "test.txt", "chunk_index": i} for i in range(len(test_docs))]
embeddings = [get_embedding(doc) for doc in test_docs]
add_documents(test_docs, embeddings, metas)

results = query_db("How does RAG work?", get_embedding("How does RAG work?"), k=3)
print(f"  Top {len(results)} results:")
for r in results:
    print(f"    score={r['score']}  {r['text'][:60]!r}")
assert len(results) == 3, f"Expected 3 results, got {len(results)}"
assert any("RAG" in r["text"] for r in results), "Expected RAG chunk in top results"
print("  PASS\n")


print("All tests passed.")
