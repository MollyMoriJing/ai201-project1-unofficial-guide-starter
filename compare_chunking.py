"""Stretch — Chunking Strategy Comparison.

Compares two chunking strategies on the SAME 5 evaluation queries, with the same
embedding model (all-MiniLM-L6-v2):

  A. Review-level (the project's approach) — one chunk per review, prefixed with
     professor/course/scores, plus a per-professor summary chunk.
  B. Naive fixed-size — split each raw document into 500-character windows with
     50-character overlap, ignoring review/sentence boundaries.

Metric: for each query we know which professor file(s) SHOULD answer it. We
retrieve the top-5 chunks for each strategy and report (a) whether the #1 chunk
comes from an expected file and (b) how many of the top-5 do ("hits@5"). Higher
is better — it means retrieval surfaced the right professor's content.

Retrieval here is done in-memory (cosine similarity over normalized embeddings)
so it doesn't touch the production ChromaDB collection.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from chunk import chunk_all
from embed_store import embed

DOCS = Path(__file__).parent / "documents"

# Each eval query paired with the source file(s) whose reviews should answer it.
QUERIES = [
    ("Among the Fundies CS2500 professors Lerner, Derbinsky, and Ahmed, who is rated highest and what do students praise?",
     {"rmp_lerner.txt", "rmp_derbinsky.txt", "rmp_ahmed.txt"}),
    ("What do students say about Nat Tuck's CS3650 Computer Systems workload?",
     {"rmp_tuck.txt"}),
    ("Which professor is most caring and gives good or affirming feedback?",
     {"rmp_rajaraman.txt", "rmp_derbinsky.txt"}),
    ("How do students describe the difference between CS3200 Database professors Fontenot and Gatterbauer?",
     {"rmp_fontenot.txt", "rmp_gatterbauer.txt"}),
    ("What are the main complaints about Karl Lieberherr's CS1100/CS1101 intro courses?",
     {"rmp_lieberherr.txt"}),
]


def review_chunks() -> list[dict]:
    """Strategy A — the project's review-level chunks."""
    return [{"text": c.text, "source_file": c.metadata["source_file"]} for c in chunk_all()]


def fixed_size_chunks(size: int = 500, overlap: int = 50) -> list[dict]:
    """Strategy B — naive fixed-size windows over each raw document."""
    out = []
    for p in sorted(DOCS.glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        step = size - overlap
        for i in range(0, len(text), step):
            piece = text[i:i + size].strip()
            if piece:
                out.append({"text": piece, "source_file": p.name})
    return out


def evaluate(chunks: list[dict], k: int = 5):
    matrix = np.array(embed([c["text"] for c in chunks]))          # (N, d), normalized
    qvecs = np.array(embed([q for q, _ in QUERIES]))               # (Q, d), normalized
    rows, total_hits, total_top1 = [], 0, 0
    for (query, expected), qv in zip(QUERIES, qvecs):
        sims = matrix @ qv                                         # cosine similarity
        top = np.argsort(-sims)[:k]
        srcs = [chunks[i]["source_file"] for i in top]
        hits = sum(s in expected for s in srcs)
        top1 = srcs[0] in expected
        total_hits += hits
        total_top1 += int(top1)
        rows.append((query, hits, k, srcs[0], top1))
    return rows, total_hits, total_top1


if __name__ == "__main__":
    for label, chunks in [
        ("A: review-level (project)", review_chunks()),
        ("B: fixed 500-char / 50-overlap", fixed_size_chunks()),
    ]:
        rows, hits, top1 = evaluate(chunks)
        print(f"\n===== Strategy {label} — {len(chunks)} chunks =====")
        for query, h, k, s1, t1 in rows:
            mark = "✓" if t1 else "✗"
            print(f"  hits@5={h}/{k}  top1={mark} {s1:<22} | {query[:55]}")
        print(f"  TOTAL relevant-hits@5 = {hits}/25 ({hits/25:.0%})   top-1 correct = {top1}/5")
