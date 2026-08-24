"""
Retrieve -> (optionally) rerank -> guardrail decision.

The guardrail lives here, not in generation.py, on purpose: we should
decide whether we're even willing to answer BEFORE spending a generation
call. That's both a latency win (skip the LLM call entirely on abstain)
and the actually-correct place to gate — the LLM doesn't know what it
doesn't know, but the retrieval scores do.
"""
from __future__ import annotations
import time
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import settings
from app.schemas import Chunk, RetrievedChunk
from app.vectorstore import VectorStore


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(
    settings.rerank_model,
    device="cpu",
)


def retrieve_and_rerank(store: VectorStore, query: str) -> tuple[list[RetrievedChunk], float, float]:
    """Returns (retrieved_chunks_sorted, retrieve_ms, rerank_ms)."""
    t0 = time.perf_counter()
    hits = store.search(query, settings.top_k_retrieve)
    retrieve_ms = (time.perf_counter() - t0) * 1000

    if not hits:
        return [], retrieve_ms, 0.0

    t1 = time.perf_counter()
    reranker = get_reranker()
    pairs = [(query, chunk.text) for chunk, _ in hits]
    rerank_scores = reranker.predict(pairs).tolist()
    rerank_ms = (time.perf_counter() - t1) * 1000

    combined = [
        RetrievedChunk(chunk=chunk, retrieval_score=score, rerank_score=rscore)
        for (chunk, score), rscore in zip(hits, rerank_scores)
    ]
    combined.sort(key=lambda r: r.rerank_score, reverse=True)
    return combined[: settings.top_k_final], retrieve_ms, rerank_ms


def should_abstain(retrieved: list[RetrievedChunk]) -> bool:
    """The core guardrail. Abstain if nothing was retrieved, or if the
    best result doesn't clear both the raw-similarity and rerank bars.
    Two independent thresholds catch different failure modes: a query
    that's off-topic entirely (low retrieval_score) vs. a query that's
    topically close but not actually answered by the text (low rerank_score,
    which the cross-encoder is much better at judging than cosine sim)."""
    if not retrieved:
        return True
    best = retrieved[0]
    if best.retrieval_score < settings.min_retrieval_score:
        return True
    if best.rerank_score is not None and best.rerank_score < settings.min_rerank_score:
        return True
    return False


def expand_context(retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """If chunks came from sentence_window strategy, swap the matched
    sentence's text for its stored neighbor window before generation,
    so the LLM sees enough context to actually answer well."""
    for r in retrieved:
        window_text = r.chunk.metadata.get("window_text")
        if window_text:
            r.chunk = Chunk(
                id=r.chunk.id, source=r.chunk.source,
                text=window_text, metadata=r.chunk.metadata,
            )
    return retrieved
