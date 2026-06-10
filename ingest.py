"""Milestone 3 — Document ingestion.

Loads every RateMyProfessors .txt file in ``documents/`` and parses it into a
structured record: the professor-level header plus a list of individual reviews.

Pure standard library — no heavy dependencies needed at this stage. The raw files
were already stripped of HTML / navigation / ads when collected (see planning.md),
so "cleaning" here is just structural parsing of the uniform header + ``REVIEW |``
line format.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).parent / "documents"


@dataclass
class Review:
    course: str = ""
    date: str = ""
    quality: float = -1.0
    difficulty: float = -1.0
    grade: str = ""
    would_take_again: str = ""
    tags: str = ""
    comment: str = ""


@dataclass
class Document:
    source_file: str
    professor: str = ""
    school: str = ""
    department: str = ""
    source_url: str = ""
    overall_quality: str = ""
    would_take_again: str = ""
    difficulty: str = ""
    courses: str = ""
    reviews: list[Review] = field(default_factory=list)


_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")
_URL_RE = re.compile(r"https?://\S+")

# Header line label -> Document attribute. "Source" is handled separately (URL extract).
_HEADER_KEYS = {
    "professor": "professor",
    "school": "school",
    "department": "department",
    "overall quality": "overall_quality",
    "would take again": "would_take_again",
    "difficulty": "difficulty",
    "courses mentioned": "courses",
}


def _parse_float(value: str) -> float:
    m = _FLOAT_RE.search(value)
    return float(m.group()) if m else -1.0


def _parse_review_line(line: str) -> Review:
    """Parse 'REVIEW | Course: X | Date: Y | Quality: q | ...' into a Review."""
    review = Review()
    # Drop the leading 'REVIEW' token; split the remaining 'key: value' fields on '|'.
    for part in (p.strip() for p in line.split("|")[1:]):
        if ":" not in part:
            continue
        key, _, val = part.partition(":")
        key, val = key.strip().lower(), val.strip()
        if key == "course":
            review.course = val
        elif key == "date":
            review.date = val
        elif key == "quality":
            review.quality = _parse_float(val)
        elif key == "difficulty":
            review.difficulty = _parse_float(val)
        elif key == "grade":
            review.grade = val
        elif key == "would take again":
            review.would_take_again = val
        elif key == "tags":
            review.tags = val
    return review


def parse_document(path: Path) -> Document:
    doc = Document(source_file=path.name)
    current: Review | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("REVIEW"):
            current = _parse_review_line(line)
            doc.reviews.append(current)
        elif current is not None:
            # A non-empty line after a REVIEW header is that review's comment text.
            current.comment = f"{current.comment} {line}".strip() if current.comment else line
        else:
            # Still in the header block.
            key, _, val = line.partition(":")
            key, val = key.strip().lower(), val.strip()
            if key == "source":
                m = _URL_RE.search(line)
                doc.source_url = m.group() if m else val
            elif key in _HEADER_KEYS:
                setattr(doc, _HEADER_KEYS[key], val)
    return doc


def load_documents(documents_dir: Path = DOCUMENTS_DIR) -> list[Document]:
    files = sorted(documents_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt documents found in {documents_dir}")
    return [parse_document(p) for p in files]


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from {DOCUMENTS_DIR}\n")
    total = 0
    for d in docs:
        total += len(d.reviews)
        print(f"- {d.source_file:24s} {d.professor:22s} reviews={len(d.reviews):<2d} overall={d.overall_quality}")
    print(f"\nTotal reviews: {total}")
