"""
Structured I/O contracts. Every stage of the pipeline consumes/produces
one of these — this is what makes the harness debuggable and lets the
benchmark script log a clean, typed trace per request instead of
grepping strings out of logs.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    text: str
    source: str
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    retrieval_score: float
    rerank_score: Optional[float] = None


class LatencyBreakdown(BaseModel):
    asr_ms: Optional[float] = None
    embed_query_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0
    # total pipeline latency excluding ASR — the number worth bragging about
    # when voice input is involved, since ASR is bounded by audio duration.
    post_asr_ms: float = 0.0


class RAGResponse(BaseModel):
    query_text: str
    transcript_confidence: Optional[float] = None
    answer: str
    abstained: bool
    retrieved: list[RetrievedChunk]
    latency: LatencyBreakdown
    retries_used: int = 0
    error: Optional[str] = None
