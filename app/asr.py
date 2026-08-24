"""
Voice -> text. Uses faster-whisper (CTranslate2 backend) — it's the
practical speed choice over stock openai-whisper: int8 quantization on
CPU alone typically gives a 2-4x speedup with negligible accuracy loss,
and it drops in cleanly if you get GPU access at the venue (device="cuda").

For the "blazing fast" claim: the honest lever here is model size
(tiny.en/base.en) and streaming partials, not this file alone. This
module exposes both a blocking transcribe() and a streaming generator
so you can start retrieval on partial text if you want to push further.
"""
import time
from functools import lru_cache

from faster_whisper import WhisperModel

from app.config import settings


@lru_cache(maxsize=1)
def _get_model() -> WhisperModel:
    # Loaded once per process — model load itself is slow (~seconds),
    # so this must NOT be in the request hot path. Warm it at server startup.
    return WhisperModel(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )


#def warm_up():
    """Call this once at server startup so the first real request isn't
    the one that pays model-load latency."""
    _get_model()


def transcribe(audio_path: str) -> tuple[str, float, float]:
    """
    Returns (text, avg_confidence, elapsed_ms).
    Confidence is derived from Whisper's avg_logprob per segment, mapped
    into a rough [0,1] so it can gate downstream behavior (e.g. re-ask
    the user if confidence is very low) if you want that as a guardrail too.
    """
    model = _get_model()
    t0 = time.perf_counter()
    segments, info = model.transcribe(audio_path, beam_size=1, vad_filter=True)
    segments = list(segments)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    text = " ".join(s.text.strip() for s in segments).strip()
    if segments:
        avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)
        # avg_logprob is typically in [-1, 0]; map to a rough 0-1 confidence
        confidence = max(0.0, min(1.0, 1.0 + avg_logprob))
    else:
        confidence = 0.0

    return text, confidence, elapsed_ms


def transcribe_stream(audio_path: str):
    """
    Generator yielding partial transcript segments as they're decoded.
    Use this if you want to overlap ASR with early (speculative) retrieval
    on the first few words — a real technique for shaving perceived
    latency, not a gimmick, but adds complexity: only worth it if you
    have time before the demo.
    """
    model = _get_model()
    segments, info = model.transcribe(audio_path, beam_size=1, vad_filter=True)
    for seg in segments:
        yield seg.text.strip()
