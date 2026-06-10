"""Milestone 4 — Embedding + vector store.

Embeds every chunk with all-MiniLM-L6-v2 (sentence-transformers, local, no API key,
no rate limit) and stores them in a persistent ChromaDB collection together with
their source metadata, so retrieval can return both the text and where it came from.

Cosine space is used so distances match the thresholds we reason about in planning.md
(≈0 = identical, <0.5 = strong match).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunk import chunk_all

EMBED_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION = "rmp_reviews"


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Encode texts to normalized vectors (so cosine distance behaves as expected)."""
    vecs = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vecs.tolist()


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def build_index() -> int:
    """(Re)build the vector store from the chunk pipeline. Idempotent via stable ids."""
    chunks = chunk_all()
    documents = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [f"{c.metadata['source_file']}#{c.metadata['position']}" for c in chunks]

    col = get_collection()
    col.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embed(documents))
    return col.count()


if __name__ == "__main__":
    print(f"Embedding model: {EMBED_MODEL}")
    n = build_index()
    print(f"Indexed {n} chunks into ChromaDB collection '{COLLECTION}' at {PERSIST_DIR}")
