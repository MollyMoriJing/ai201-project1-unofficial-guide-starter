"""Milestone 3 — Chunking.

Strategy (see planning.md):
  * One chunk per *review* — we split on review boundaries, NOT a fixed character
    count, because a single review is the natural self-contained unit of opinion.
  * Each review chunk is prefixed with professor + course + score metadata so the
    chunk is self-contained and self-attributing (a bare "very kind and approachable"
    would otherwise never match a "Rajaraman" query).
  * Plus one short summary chunk per professor carrying the aggregate rating.
  * Overlap = 0: boundaries are whole reviews, so no fact is ever cut mid-thought.

Metadata is kept flat and ChromaDB-friendly (str / float / int only — no None,
no lists) so the same dicts can be handed straight to the vector store in M4.
"""
from __future__ import annotations

from dataclasses import dataclass

from ingest import Document, Review, load_documents


@dataclass
class Chunk:
    text: str
    metadata: dict


def _summary_chunk(doc: Document, position: int) -> Chunk:
    text = (
        f"Professor {doc.professor} — {doc.school}. "
        f"Overall student rating {doc.overall_quality}; "
        f"would take again {doc.would_take_again}; difficulty {doc.difficulty}. "
        f"Courses: {doc.courses}. Source: RateMyProfessors."
    )
    return Chunk(text, {
        "chunk_type": "summary",
        "professor": doc.professor,
        "course": doc.courses,
        "date": "",
        "quality": -1.0,
        "difficulty": -1.0,
        "tags": "",
        "source_file": doc.source_file,
        "source_url": doc.source_url,
        "position": position,
    })


def _review_chunk(doc: Document, review: Review, position: int) -> Chunk:
    head = f"Professor {doc.professor} (Northeastern CS)"
    if review.course:
        head += f" — Course {review.course}"

    scores = []
    if review.quality >= 0:
        scores.append(f"Quality {review.quality}/5")
    if review.difficulty >= 0:
        scores.append(f"Difficulty {review.difficulty}/5")
    detail = " | ".join(filter(None, [
        review.date,
        ", ".join(scores),
        f"Tags: {review.tags}" if review.tags else "",
    ]))

    header = f"{head} | {detail}" if detail else head
    text = f"{header}\n{review.comment}"
    return Chunk(text, {
        "chunk_type": "review",
        "professor": doc.professor,
        "course": review.course,
        "date": review.date,
        "quality": review.quality,
        "difficulty": review.difficulty,
        "tags": review.tags,
        "source_file": doc.source_file,
        "source_url": doc.source_url,
        "position": position,
    })


def chunk_document(doc: Document) -> list[Chunk]:
    chunks = [_summary_chunk(doc, position=0)]
    for i, review in enumerate(doc.reviews, start=1):
        if review.comment.strip():  # never emit an empty chunk
            chunks.append(_review_chunk(doc, review, position=i))
    return chunks


def chunk_all(documents: list[Document] | None = None) -> list[Chunk]:
    docs = documents if documents is not None else load_documents()
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    return chunks


if __name__ == "__main__":
    import random

    chunks = chunk_all()
    lengths = [len(c.text) for c in chunks]
    n_summary = sum(c.metadata["chunk_type"] == "summary" for c in chunks)
    n_review = sum(c.metadata["chunk_type"] == "review" for c in chunks)
    empties = sum(not c.text.strip() for c in chunks)

    print(f"Total chunks: {len(chunks)}  (summary={n_summary}, review={n_review})")
    print(f"Char length: min={min(lengths)}  max={max(lengths)}  mean={sum(lengths) // len(lengths)}")
    print(f"Empty chunks: {empties}")

    print("\n=== 5 random chunks ===")
    random.seed(7)
    for c in random.sample(chunks, 5):
        m = c.metadata
        print(f"\n[{m['chunk_type']}] source={m['source_file']} prof={m['professor']} course={m['course']}")
        print(c.text)
