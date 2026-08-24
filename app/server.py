"""
FastAPI server. Two endpoints:
  POST /ask/voice  — upload an audio file (webm/wav/mp3), get a grounded answer
  POST /ask/text   — text query, bypasses ASR (useful for testing/UI fallback)

Run: uvicorn app.server:app --reload --port 8000
The frontend (browser) records mic audio and POSTs it to /ask/voice —
MediaRecorder -> Blob -> FormData is the standard path; no special
frontend framework required, plain JS works fine for a demo.
"""
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

#from app.asr import warm_up
from app.config import settings
from app.pipeline import run_pipeline
from app.schemas import RAGResponse
from app.vectorstore import VectorStore

app = FastAPI(title="Voice-Enabled RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


_store: VectorStore | None = None


def _store() -> VectorStore:
    global _store

    if _store is None:
        _store = VectorStore.load(settings.index_dir)

    return _store


@app.post("/ask/voice", response_model=RAGResponse)
async def ask_voice(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    return run_pipeline(get_store, audio_path=tmp_path)


@app.post("/ask/text", response_model=RAGResponse)
async def ask_text(query: str):
    return run_pipeline(get_store, query_text=query)


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(_store.chunks) if _store else 0}
