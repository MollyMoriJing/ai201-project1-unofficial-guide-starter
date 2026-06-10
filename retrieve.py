"""Milestone 4 — Retrieval.

Given a query string, embed it with the same model used to build the index and
return the top-k most similar chunks, each with its source metadata and cosine
distance. Run this module directly to sanity-check retrieval before wiring up
generation (M5): print the returned chunks + distances and eyeball relevance.
"""
from __future__ import annotations

from embed_store import embed, get_collection


def retrieve(query: str, k: int = 5, where: dict | None = None) -> list[dict]:
    """Semantic top-k retrieval. `where` is an optional ChromaDB metadata filter,
    e.g. {"professor": "Nathaniel Tuck"} or {"quality": {"$gte": 4.0}} (stretch:
    metadata filtering)."""
    col = get_collection()
    kwargs = dict(
        query_embeddings=embed([query]),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where
    res = col.query(**kwargs)
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


if __name__ == "__main__":
    queries = [
        "Among the Fundies CS2500 professors, who is rated highest and why?",
        "What do students say about Nat Tuck's CS3650 computer systems workload?",
        "Which professor is most caring and gives the most helpful feedback?",
    ]
    for q in queries:
        print(f"\n{'=' * 80}\nQUERY: {q}\n{'=' * 80}")
        for r in retrieve(q, k=5):
            m = r["metadata"]
            print(f"\n  distance={r['distance']:.3f}  [{m['professor']} | {m['source_file']} | {m['chunk_type']}]")
            print("  " + r["text"].replace("\n", "\n  "))
