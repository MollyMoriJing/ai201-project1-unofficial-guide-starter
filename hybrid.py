"""Stretch — Hybrid search (BM25 + semantic) via Reciprocal Rank Fusion.

Combines lexical BM25 — great at exact tokens like surnames ("Ahmed") and course
codes ("CS3200") — with semantic embedding search — great at matching meaning
without shared words. The two rankings are merged with Reciprocal Rank Fusion
(RRF), which needs no score normalization: each document scores
sum(1 / (rrf_k + rank_in_method)).

This directly targets the documented Q1 failure: pure semantic top-5 dropped
"Ahmed" entirely; BM25 guarantees the exact surname token pulls Ahmed's chunks
into the fused result.

The chunk embeddings and BM25 index are built once and cached.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from chunk import chunk_all
from embed_store import embed

_TOK = re.compile(r"[a-z0-9]+")


def _tok(s: str) -> list[str]:
    return _TOK.findall(s.lower())


@lru_cache(maxsize=1)
def _index():
    chunks = chunk_all()
    texts = [c.text for c in chunks]
    metas = [c.metadata for c in chunks]
    embs = np.array(embed(texts))                      # normalized → dot = cosine
    bm25 = BM25Okapi([_tok(t) for t in texts])
    return texts, metas, embs, bm25


def _ranks(scores: np.ndarray) -> dict[int, int]:
    """index -> rank (0 = best) by descending score."""
    return {idx: r for r, idx in enumerate(np.argsort(-scores))}


def semantic_only(query: str, k: int = 5) -> list[dict]:
    texts, metas, embs, _ = _index()
    sem = embs @ np.array(embed([query])[0])
    return [{"text": texts[i], "metadata": metas[i], "distance": 1.0 - float(sem[i])}
            for i in np.argsort(-sem)[:k]]


def bm25_only(query: str, k: int = 5) -> list[dict]:
    texts, metas, _, bm25 = _index()
    bm = np.array(bm25.get_scores(_tok(query)))
    return [{"text": texts[i], "metadata": metas[i], "bm25": float(bm[i])}
            for i in np.argsort(-bm)[:k]]


def hybrid_retrieve(query: str, k: int = 5, rrf_k: int = 60) -> list[dict]:
    texts, metas, embs, bm25 = _index()
    sem = embs @ np.array(embed([query])[0])
    bm = np.array(bm25.get_scores(_tok(query)))
    sem_rank, bm_rank = _ranks(sem), _ranks(bm)
    rrf = {i: 1.0 / (rrf_k + sem_rank[i]) + 1.0 / (rrf_k + bm_rank[i]) for i in range(len(texts))}
    top = sorted(rrf, key=rrf.get, reverse=True)[:k]
    # keep the semantic distance so the generation distance-gate still works
    return [{"text": texts[i], "metadata": metas[i],
             "distance": 1.0 - float(sem[i]), "rrf": rrf[i]} for i in top]


if __name__ == "__main__":
    queries = [
        "Among the Fundies (CS2500) professors Lerner, Derbinsky, and Ahmed, who has the highest student rating, and what do students praise about them?",
        "How do students describe the difference between the two CS3200 Database professors, Mark Fontenot and Wolfgang Gatterbauer?",
        "Which Northeastern CS professor is most often described as caring and giving good or affirming feedback?",
    ]
    for q in queries:
        print("=" * 95)
        print("QUERY:", q)
        for name, fn in [("semantic", semantic_only), ("BM25", bm25_only), ("hybrid", hybrid_retrieve)]:
            profs = [r["metadata"]["professor"] for r in fn(q, k=5)]
            print(f"  {name:9} top-5 → {profs}")
