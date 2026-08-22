"""
LLM generation with retries. Uses a small/fast model on purpose
(settings.gen_model) — Haiku-class latency matters far more than
Opus-class reasoning depth for a RAG-over-short-context answer.

Retries use tenacity with exponential backoff, capped, so a transient
5xx/network blip doesn't tank your demo — but capped low (default 2)
so a genuinely broken call fails fast instead of eating your latency
budget.
"""
from __future__ import annotations
import os
import time

from anthropic import Anthropic, APIError, APIStatusError

from app.config import settings
from app.schemas import RetrievedChunk

_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a grounded question-answering assistant. You must
answer ONLY using the provided context. If the context does not contain
the answer, say so plainly instead of guessing. Do not use outside
knowledge. Keep answers concise and cite which passage you used by number."""


def _build_prompt(query: str, retrieved: list[RetrievedChunk]) -> str:
    context_block = "\n\n".join(
        f"[{i+1}] (source: {r.chunk.source})\n{r.chunk.text}"
        for i, r in enumerate(retrieved)
    )
    return f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer using only the context above."


class GenerationError(Exception):
    pass


def _call_model_once(prompt: str) -> str:
    resp = _client.messages.create(
        model=settings.gen_model,
        max_tokens=settings.gen_max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "".join(text_parts).strip()


def generate_answer(query: str, retrieved: list[RetrievedChunk]) -> tuple[str, int]:
    """Returns (answer_text, retries_used). Raises GenerationError on
    total failure so the pipeline can fall back gracefully instead of
    crashing the request. Manual retry loop (rather than a bare tenacity
    decorator) so retries_used is exact per-call, which matters for the
    benchmark report and for debugging flaky demo runs."""
    prompt = _build_prompt(query, retrieved)
    last_err: Exception | None = None
    for attempt in range(settings.gen_max_retries + 1):
        try:
            answer = _call_model_once(prompt)
            return answer, attempt
        except (APIError, APIStatusError) as e:
            last_err = e
            if attempt < settings.gen_max_retries:
                time.sleep(min(2.0, 0.2 * (2 ** attempt)))
                continue
    raise GenerationError(str(last_err)) from last_err
