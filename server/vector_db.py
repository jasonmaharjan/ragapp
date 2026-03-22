import chromadb
import os

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "../chroma_db")

client = chromadb.Client()

collection = client.get_or_create_collection(name="docs")

# add documents along with their embeddings to Chroma
# give unique ID to each chunk
def add_documents(chunks, embeddings):
    ids = [str(i) for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)


# retrieve the top "k" similar documents
def query_db(query_embedding, k=3):
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results
