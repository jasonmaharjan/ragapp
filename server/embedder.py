from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


# convert text -> embedding vector (number)
def get_embedding(text: str):
    return model.encode(text).tolist()
