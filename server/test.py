from embedder import get_embedding
from vector_db import add_documents, query_db

# think input from document
test_chunks = [
    "Hello world",
    "This is a RAG pipeline",
    "This is a garbage text",
    "This is a 'RAG' garbage text",
]
embeddings = [get_embedding(chunk) for chunk in test_chunks]

print(embeddings)

add_documents(test_chunks, embeddings)

# think response
query_emb = get_embedding("What do you do in RAG??")
res = query_db(query_emb)

print("Here you go\n")
print(res["documents"][0])
