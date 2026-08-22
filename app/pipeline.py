"""
The harness. Orchestrates ASR -> retrieve -> rerank -> guardrail ->
generate, with per-stage timing (for the latency report) and structured
error recovery: every stage that can fail is caught, logged into the
response, and degrades to a safe abstain instead of a 500.

This is the single entrypoint both the FastAPI server and the benchmark
script call, so "runs inside a real harness" is true of every code path,
not just the demo one.
"""
from __future__ import annotations
import time

from app.asr import transcribe
from app.config import settings
from app.generation import GenerationError, generate_answer
from app.retrieval import expand_context, retrieve_and_rerank, should_abstain
from app.schemas import LatencyBreakdown, RAGResponse
from app.vectorstore import VectorStore, embed_texts


def run_pipeline(store: VectorStore, *, audio_path: str | None = None,
                  query_text: str | None = None) -> RAGResponse:
    """Provide either audio_path (voice input) or query_text (text input,
    useful for benchmarking retrieval+generation latency in isolation
    from ASR — see scripts/benchmark.py)."""
    t_start = time.perf_counter()
    latency = LatencyBreakdown()
    transcript_confidence = None

    # --- Stage 1: ASR (optional) ---
    if audio_path:
        try:
            query_text, transcript_confidence, asr_ms = transcribe(audio_path)
            latency.asr_ms = asr_ms
        except Exception as e:
            return RAGResponse(
                query_text="", answer=settings.abstain_message, abstained=True,
                retrieved=[], latency=latency, error=f"ASR failed: {e}",
            )
    if not query_text or not query_text.strip():
        latency.total_ms = (time.perf_counter() - t_start) * 1000
        return RAGResponse(
            query_text=query_text or "", transcript_confidence=transcript_confidence,
            answer="I couldn't hear a question — could you try again?", abstained=True,
            retrieved=[], latency=latency, error="empty transcript",
        )

    t_post_asr = time.perf_counter()

    # --- Stage 2 & 3: retrieve + rerank ---
    try:
        retrieved, retrieve_ms, rerank_ms = retrieve_and_rerank(store, query_text)
        latency.retrieve_ms = retrieve_ms
        latency.rerank_ms = rerank_ms
    except Exception as e:
        latency.total_ms = (time.perf_counter() - t_start) * 1000
        latency.post_asr_ms = (time.perf_counter() - t_post_asr) * 1000
        return RAGResponse(
            query_text=query_text, transcript_confidence=transcript_confidence,
            answer=settings.abstain_message, abstained=True,
            retrieved=[], latency=latency, error=f"retrieval failed: {e}",
        )

    # --- Stage 4: guardrail ---
    if should_abstain(retrieved):
        latency.total_ms = (time.perf_counter() - t_start) * 1000
        latency.post_asr_ms = (time.perf_counter() - t_post_asr) * 1000
        return RAGResponse(
            query_text=query_text, transcript_confidence=transcript_confidence,
            answer=settings.abstain_message, abstained=True,
            retrieved=retrieved, latency=latency,
        )

    retrieved = expand_context(retrieved)

    # --- Stage 5: generation (with retries handled inside) ---
    try:
        t_gen = time.perf_counter()
        answer, retries_used = generate_answer(query_text, retrieved)
        latency.generate_ms = (time.perf_counter() - t_gen) * 1000
    except GenerationError as e:
        latency.total_ms = (time.perf_counter() - t_start) * 1000
        latency.post_asr_ms = (time.perf_counter() - t_post_asr) * 1000
        return RAGResponse(
            query_text=query_text, transcript_confidence=transcript_confidence,
            answer="I retrieved relevant context but couldn't generate an answer "
                   "right now — please try again.",
            abstained=True, retrieved=retrieved, latency=latency,
            error=f"generation failed: {e}",
        )

    latency.total_ms = (time.perf_counter() - t_start) * 1000
    latency.post_asr_ms = (time.perf_counter() - t_post_asr) * 1000

    return RAGResponse(
        query_text=query_text, transcript_confidence=transcript_confidence,
        answer=answer, abstained=False, retrieved=retrieved,
        latency=latency, retries_used=retries_used,
    )


def build_index_from_texts(texts_and_sources: list[tuple[str, str]]) -> VectorStore:
    """Convenience used by scripts/ingest.py — chunks every doc with the
    configured strategy and builds the FAISS index."""
    from app.chunking import chunk_document

    all_chunks = []
    for text, source in texts_and_sources:
        embed_fn = embed_texts if settings.chunk_strategy == "semantic" else None
        chunks = chunk_document(
            text, source, settings.chunk_strategy,
            chunk_size=settings.chunk_size, overlap=settings.chunk_overlap,
            window=settings.sentence_window, embed_fn=embed_fn,
        )
        all_chunks.extend(chunks)

    dim = embed_texts(["dimension probe"]).shape[1]
    store = VectorStore(dim=dim)
    store.add(all_chunks)
    return store
