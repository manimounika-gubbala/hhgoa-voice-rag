"""
Build the FAISS index from data/sample_docs/*.txt using the chunking
strategy set in app/config.py (env var CHUNK_STRATEGY).

Usage:
  python scripts/ingest.py
  CHUNK_STRATEGY=semantic python scripts/ingest.py   # try a different strategy
"""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings
from app.pipeline import build_index_from_texts


def main():
    docs = []
    for path in glob.glob("data/sample_docs/*.txt"):
        with open(path, encoding="utf-8") as f:
            docs.append((f.read(), os.path.basename(path)))

    if not docs:
        print("No .txt files found in data/sample_docs/. Add some and re-run.")
        return

    print(f"Ingesting {len(docs)} documents with strategy='{settings.chunk_strategy}' ...")
    store = build_index_from_texts(docs)
    store.save(settings.index_dir)
    print(f"Indexed {len(store.chunks)} chunks -> {settings.index_dir}")


if __name__ == "__main__":
    main()
