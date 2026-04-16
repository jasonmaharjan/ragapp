import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from db import get_history, init_db
from embedder import get_embedding
from ingestion import SUPPORTED_EXTENSIONS, ingest_document
from query import run_query
from vector_db import add_documents


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        init_db()
    except Exception:
        pass  # Postgres not required — graceful degradation
    yield


app = FastAPI(title="RAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    k: int = 5
    alpha: float = 0.5
    session_id: str | None = None


@app.get("/")
def health():
    return {"status": "200 OK", "message": "Hello RAG API!"}


@app.post("/ingest")
async def ingest(file: UploadFile):
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=422, detail=f"UNSUPPORTED File Type: '{suffix}'.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        chunks = ingest_document(tmp_path)
    finally:
        os.unlink(tmp_path)

    texts = [c.text for c in chunks]
    metas = [{"source": file.filename, "chunk_index": c.index} for c in chunks]
    add_documents(texts, [get_embedding(t) for t in texts], metas)

    return {"source": file.filename, "chunks_ingested": len(chunks)}


@app.post("/query")
async def query(request: QueryRequest):
    return run_query(request.query, k=request.k, alpha=request.alpha, session_id=request.session_id)


@app.get("/history/{session_id}")
def history(session_id: str):
    return {"session_id": session_id, "history": get_history(session_id)}