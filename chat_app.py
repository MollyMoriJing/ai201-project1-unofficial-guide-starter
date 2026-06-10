"""Stretch — Conversational query interface (Gradio chat, with memory).

A chat UI on top of ask_chat(): follow-up questions are resolved against the
conversation so far. Run `python chat_app.py` and open http://localhost:7860.
"""
import gradio as gr

from generate import ask_chat


def respond(message: str, history):
    # history (type="messages") is a list of {"role", "content"} dicts; pair them
    # into (user, assistant) tuples for ask_chat.
    pairs, pending = [], None
    for m in history:
        role = m["role"] if isinstance(m, dict) else m[0]
        content = m["content"] if isinstance(m, dict) else m[1]
        if role == "user":
            pending = content
        elif role == "assistant" and pending is not None:
            pairs.append((pending, content))
            pending = None

    result = ask_chat(message, history=pairs)
    answer = result["answer"]
    if result.get("standalone_question") and result["standalone_question"] != message:
        answer += f"\n\n_(understood as: {result['standalone_question']})_"
    if result["sources"]:
        answer += "\n\nSources: " + "; ".join(s.split(" — ")[0] for s in result["sources"])
    return answer


demo = gr.ChatInterface(
    respond,
    title="🎓 The Unofficial Guide — chat (with memory)",
    description="Ask about NEU CS professors. Follow-ups like \"is his workload heavy?\" remember who you meant.",
    examples=[
        "What do students say about Nat Tuck's teaching style?",
        "Is his workload heavy?",
        "How does that compare to Derbinsky?",
    ],
)

if __name__ == "__main__":
    demo.launch()
