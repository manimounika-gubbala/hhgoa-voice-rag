"""
Thin FAISS wrapper. Uses IndexFlatIP over L2-normalized embeddings, which
is mathematically equivalent to cosine similarity search and keeps scores
in a nice interpretable [-1, 1] range for the guardrail threshold.

For a hackathon-scale corpus (thousands of chunks) a flat index is the
right call — it's exact (no recall loss) and still fast. Don't reach for
IVF/HNSW unless your corpus is huge; it adds tuning risk for no benefit
at this scale.
"""
from __future__ import annotations
import json
import os
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.schemas import Chunk


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embed_model)


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedder()
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False)
    return vecs.astype("float32")


class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]):
        vecs = embed_texts([c.text for c in chunks])
        self.index.add(vecs)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        qvec = embed_texts([query])
        scores, idxs = self.index.search(qvec, min(top_k, len(self.chunks) or 1))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.json"), "w") as f:
            json.dump([c.model_dump() for c in self.chunks], f)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.json")) as f:
            raw = json.load(f)
        store = cls.__new__(cls)
        store.index = index
        store.chunks = [Chunk(**r) for r in raw]
        return store
