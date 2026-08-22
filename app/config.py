"""
Central configuration. Everything here is intentionally overridable via
env vars so the harness / benchmark script can sweep settings without
touching code (useful for the P50/P70/P100 runs across chunking strategies).
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # --- ASR ---
    whisper_model: str = os.getenv("WHISPER_MODEL", "small.en")   # tiny.en for speed, small.en for accuracy
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")      # "cuda" if you have a GPU — big latency win
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE", "int8")  # int8 on CPU is the speed unlock

    # --- Embeddings / retrieval ---
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # small & fast
    rerank_model: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_k_retrieve: int = int(os.getenv("TOP_K_RETRIEVE", 8))
    top_k_final: int = int(os.getenv("TOP_K_FINAL", 3))

    # --- Chunking ---
    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "recursive")  # fixed | recursive | sentence_window | semantic
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 400))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 60))
    sentence_window: int = int(os.getenv("SENTENCE_WINDOW", 2))

    # --- Guardrails ---
    # Below this cosine-similarity score, we don't trust retrieval enough to answer.
    min_retrieval_score: float = float(os.getenv("MIN_RETRIEVAL_SCORE", 0.20))
    # Below this rerank score, same deal, applied after cross-encoder reranking.
    min_rerank_score: float = float(os.getenv("MIN_RERANK_SCORE", -1.50))
    abstain_message: str = "I don't have enough grounded information to answer that confidently."

    # --- Generation ---
    gen_model: str = os.getenv("GEN_MODEL", "claude-haiku-4-5-20251001")  # small/fast for latency
    gen_max_tokens: int = int(os.getenv("GEN_MAX_TOKENS", 300))
    gen_max_retries: int = int(os.getenv("GEN_MAX_RETRIES", 2))

    # --- Storage ---
    index_dir: str = os.getenv("INDEX_DIR", "data/index")


settings = Settings()
