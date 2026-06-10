"""Milestone 5 — Grounded generation.

Connects retrieval to Groq's llama-3.3-70b-versatile. Grounding is enforced on
two levels:

  1. Prompt — the system prompt forbids using any knowledge outside the supplied
     review excerpts and requires an explicit refusal when they don't answer the
     question. Temperature is 0 so the model doesn't embellish.
  2. Pipeline — a distance gate: if even the closest retrieved chunk is far away
     (cosine distance > GATE_DISTANCE), we refuse *before* calling the LLM, so an
     off-topic query (e.g. dining halls) can't be answered from training data.

Source attribution is added programmatically from the retrieved chunks' metadata,
never written by the LLM, so citations cannot be hallucinated.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from retrieve import retrieve

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL = "llama-3.3-70b-versatile"
GATE_DISTANCE = 0.72  # calibrated from observed distances: in-scope best ≤0.55, out-of-scope best ≥0.68
REFUSAL = (
    "I don't have enough information in the student reviews I have access to "
    "to answer that."
)

SYSTEM_PROMPT = f"""You are The Unofficial Guide, a question-answering assistant for Northeastern University students. You answer ONLY using the student reviews provided in the context.

Rules:
- Use ONLY facts stated in the provided reviews. Do NOT use any outside or general knowledge.
- If the reviews do not contain enough information to answer the question, reply with exactly: "{REFUSAL}"
- Never invent professor names, ratings, course codes, or quotes that are not in the context.
- When reviews disagree about a professor, represent both sides instead of picking one.
- Be concise (2–5 sentences). Refer to professors by name and cite concrete details (ratings, workload, exam style) drawn from the reviews.
- Do NOT write your own "Sources" list; sources are attached automatically."""


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        blocks.append(f"[{i}] (from {m['professor']}, {m['source_file']})\n{c['text']}")
    return "\n\n".join(blocks)


def _sources(chunks: list[dict]) -> list[str]:
    """Unique source list, built programmatically from retrieval metadata."""
    seen: dict[str, str] = {}
    for c in chunks:
        m = c["metadata"]
        seen.setdefault(m["source_file"], f"{m['professor']} — RateMyProfessors ({m['source_url']})")
    return list(seen.values())


def _is_refusal(answer: str) -> bool:
    return "don't have enough information" in answer.lower()


@lru_cache(maxsize=1)
def _client() -> Groq:
    key = os.environ.get("GROQ_API_KEY")
    if not key or key == "your_key_here":
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste your free "
            "key from https://console.groq.com"
        )
    return Groq(api_key=key)


def ask(question: str, k: int = 5) -> dict:
    """Return {answer, sources, chunks} for a user question, grounded in retrieved reviews."""
    chunks = retrieve(question, k=k)

    # Pipeline-level grounding: refuse before calling the LLM if nothing is close enough.
    if not chunks or chunks[0]["distance"] > GATE_DISTANCE:
        return {"answer": REFUSAL, "sources": [], "chunks": chunks}

    user_msg = (
        f"Context — student reviews:\n\n{_format_context(chunks)}\n\n"
        f"Question: {question}"
    )
    resp = _client().chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    answer = resp.choices[0].message.content.strip()
    sources = [] if _is_refusal(answer) else _sources(chunks)
    return {"answer": answer, "sources": sources, "chunks": chunks}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Which Fundies (CS2500) professor is rated highest and why?"
    result = ask(q)
    print(f"Q: {q}\n\nA: {result['answer']}\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  • {s}")
