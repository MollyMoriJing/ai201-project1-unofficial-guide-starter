"""Milestone 5 — Query interface (Gradio).

A minimal web UI: type a question, get a grounded answer plus the list of
RateMyProfessors sources the answer was drawn from. Run with `python app.py`
and open http://localhost:7860.
"""
import gradio as gr

from generate import ask

EXAMPLES = [
    "Which Fundies (CS2500) professor is rated highest, and why?",
    "What do students say about Nat Tuck's CS3650 computer systems workload?",
    "Which Northeastern CS professor is most caring and gives the most helpful feedback?",
    "How do students compare the two CS3200 Database professors, Fontenot and Gatterbauer?",
    "What are the main complaints about Karl Lieberherr's intro courses?",
]


def handle_query(question: str):
    if not question or not question.strip():
        return "Please enter a question.", ""
    result = ask(question)
    sources = "\n".join(f"• {s}" for s in result["sources"])
    if not sources:
        sources = "(no sources — this question is outside the scope of the reviews)"
    return result["answer"], sources


with gr.Blocks(title="The Unofficial Guide — NEU CS Professors") as demo:
    gr.Markdown(
        "# 🎓 The Unofficial Guide\n"
        "Ask about **Northeastern CS professors & courses**. Every answer is grounded "
        "*only* in real student reviews from RateMyProfessors — if the reviews don't "
        "cover it, the system says so instead of guessing."
    )
    inp = gr.Textbox(label="Your question", placeholder="e.g. Which Fundies (CS2500) professor is rated highest?")
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=6)
    sources = gr.Textbox(label="Retrieved from (sources)", lines=4)
    gr.Examples(EXAMPLES, inputs=inp)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
