"""Stretch — Conversational memory demo.

A follow-up that only makes sense given the previous turn ("Is his workload
heavy?") is rewritten into a standalone question using the conversation history,
so retrieval still finds the right professor. The rewritten query is printed so
you can see the memory at work (not just topic overlap).

Run: python chat.py
"""
from generate import ask_chat

if __name__ == "__main__":
    history: list[tuple[str, str]] = []
    turns = [
        "What do students say about Nat Tuck's teaching style?",
        "Is his workload heavy?",               # "his" -> Tuck (needs memory)
        "How does that compare to Derbinsky?",  # references the running comparison
    ]
    for q in turns:
        r = ask_chat(q, history=history)
        print("=" * 90)
        print("USER:", q)
        if r["standalone_question"] != q:
            print("   (rewritten with memory → ", r["standalone_question"], ")")
        print("ASSISTANT:", r["answer"])
        srcs = [s.split(" — ")[0] for s in r["sources"]]
        print("SOURCES:", srcs or "(none)")
        history.append((q, r["answer"]))
