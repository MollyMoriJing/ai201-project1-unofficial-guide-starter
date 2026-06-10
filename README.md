# The Unofficial Guide — Project 1

A retrieval-augmented (RAG) question-answering system over **real student reviews of Northeastern
University CS professors**. Ask a plain-English question ("Which Fundies professor is rated highest?")
and get an answer grounded *only* in actual RateMyProfessors reviews, with the sources it drew from.

**Stack:** `all-MiniLM-L6-v2` (sentence-transformers) → ChromaDB → Groq `llama-3.3-70b-versatile` → Gradio.

**Run it:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your free Groq key from console.groq.com
python embed_store.py         # build the vector index (one time)
python app.py                 # open http://localhost:7860
```

---

## Domain

**Northeastern University CS professor & course reviews** — what students actually say about Khoury
College of Computer Sciences professors: teaching style, exam/workload reality, grading, and whether a
class is worth taking.

This knowledge is valuable and hard to find officially because the course catalog and the login-gated
TRACE evaluations give you a sanitized title, a one-paragraph description, and aggregate numbers — never
that one CS3650 section "rambles for an hour about a concept he never defined," that a Fundies professor
gives "the best introduction to CS I could have asked for," or that two CS3200 Database sections split
sharply on difficulty. That signal lives in student-written reviews, and it is exactly what an incoming
student needs when choosing a section.

---

## Document Sources

12 documents, one per professor, collected from **RateMyProfessors** (the canonical public source for
this domain), spanning different course areas so the corpus answers a range of questions and supports
*comparative* ones (three Fundies professors; two Database professors). Each file holds the professor's
profile header (overall rating, would-take-again %, difficulty, courses) plus 4–5 verbatim reviews.

| #  | Source (Professor) | Type | URL / file |
|----|--------------------|------|-----------|
| 1  | Leena Razzaq       | RateMyProfessors | [professor/1682644](https://www.ratemyprofessors.com/professor/1682644) → `documents/rmp_razzaq.txt` |
| 2  | David Choffnes     | RateMyProfessors | [professor/2219053](https://www.ratemyprofessors.com/professor/2219053) → `documents/rmp_choffnes.txt` |
| 3  | Benjamin Lerner    | RateMyProfessors | [professor/2034405](https://www.ratemyprofessors.com/professor/2034405) → `documents/rmp_lerner.txt` |
| 4  | Benjamin Hescott   | RateMyProfessors | [professor/2365410](https://www.ratemyprofessors.com/professor/2365410) → `documents/rmp_hescott.txt` |
| 5  | Nathaniel Tuck     | RateMyProfessors | [professor/2044282](https://www.ratemyprofessors.com/professor/2044282) → `documents/rmp_tuck.txt` |
| 6  | Nathaniel Derbinsky| RateMyProfessors | [professor/2308643](https://www.ratemyprofessors.com/professor/2308643) → `documents/rmp_derbinsky.txt` |
| 7  | Jose Annunziato    | RateMyProfessors | [professor/1980304](https://www.ratemyprofessors.com/professor/1980304) → `documents/rmp_annunziato.txt` |
| 8  | Amal Ahmed         | RateMyProfessors | [professor/1626525](https://www.ratemyprofessors.com/professor/1626525) → `documents/rmp_ahmed.txt` |
| 9  | Karl Lieberherr    | RateMyProfessors | [professor/430930](https://www.ratemyprofessors.com/professor/430930) → `documents/rmp_lieberherr.txt` |
| 10 | Mark Fontenot      | RateMyProfessors | [professor/2868024](https://www.ratemyprofessors.com/professor/2868024) → `documents/rmp_fontenot.txt` |
| 11 | Rajmohan Rajaraman | RateMyProfessors | [professor/157371](https://www.ratemyprofessors.com/professor/157371) → `documents/rmp_rajaraman.txt` |
| 12 | Wolfgang Gatterbauer | RateMyProfessors | [professor/2303072](https://www.ratemyprofessors.com/professor/2303072) → `documents/rmp_gatterbauer.txt` |

Course coverage: intro (Razzaq, Lieberherr), Fundies CS2500/2510 (Lerner, Derbinsky, Ahmed), systems
CS3650 (Tuck), theory/discrete (Hescott), algorithms (Rajaraman), networks (Choffnes), databases CS3200
(Fontenot, Gatterbauer), web CS4550 (Annunziato).

---

## Chunking Strategy

**Chunk size:** One **review per chunk** — documents are split on the `REVIEW |` boundary, *not* by a
fixed character count, because each review (1–5 sentences) is the natural self-contained unit of
opinion. Measured chunk length: **min 136, max 539, mean 386 characters**. One short **summary chunk per
professor** (overall rating / would-take-again / difficulty / courses) is also emitted.

**Overlap:** **0.** Overlap exists to rescue a fact a fixed-size splitter cut in half. Splitting on whole
reviews never cuts an opinion mid-thought, so overlap would only paste an unrelated student's words onto
the next chunk and blur the embedding. (Overlap would help a long FAQ; it hurts a review corpus.)

**Preprocessing:** Reviews were collected already free of HTML/nav/ads and normalized into a uniform
`header + REVIEW | …` format. One review in `rmp_lieberherr.txt` containing graphic self-harm statements
unrelated to teaching was removed during cleaning (its 4 remaining reviews convey the same substance).

**Self-containment (key choice):** a bare comment like *"very kind and approachable"* has no professor
name or course in it, so each review chunk is **prefixed with professor + course + scores** before
embedding — making every chunk independently retrievable and self-attributing.

**Final chunk count:** **71** (12 summary + 59 review). Comfortably inside the 50–2000 healthy range; 0
empty chunks.

### Sample chunks (5 labeled, with source)

```
[review · rmp_derbinsky.txt]
Professor Nathaniel Derbinsky (Northeastern CS) — Course CS2500 | 2024-03-20 | Quality 5.0/5, Difficulty 3.0/5 | Tags: Gives good feedback, Caring, Accessible outside class
Derbinsky is awesome!! I took Fundies I as a non-CS major with very little prior programming knowledge, and while the course is definitely challenging, he made it such a positive experience I could succeed in...

[review · rmp_tuck.txt]
Professor Nathaniel Tuck (Northeastern CS) — Course CS3650 | 2021-03-28 | Quality 4.0/5, Difficulty 5.0/5 | Tags: Lots of homework, Skip class? You won't pass, Lecture heavy
Nat (usually) boils his lectures down to the very core: if you listen, you will learn exactly what you need to succeed at the homeworks...

[review · rmp_lieberherr.txt]
Professor Karl Lieberherr (Northeastern CS) — Course CS1101 | 2026-04-28 | Quality 1.0/5, Difficulty 4.0/5 | Tags: Tough grader, Group projects, Test heavy
RUN. taught by claude, assignments and tests written by claude, probably graded by claude. makeup exams are nice and saved my grade, LONG learning curve to get to an A...

[review · rmp_rajaraman.txt]
Professor Rajmohan Rajaraman (Northeastern CS) — Course Algorithms | 2026-04-05 | Quality 5.0/5, Difficulty 4.0/5 | Tags: Amazing lectures, Caring, Accessible outside class, Helpful
He is actually one of the best professors I've had at northeastern. He is so kind and caring and SO affirming. You never feel bad asking a question...

[summary · rmp_gatterbauer.txt]
Professor Wolfgang Gatterbauer — Northeastern University (Khoury College of Computer Sciences). Overall student rating 2.6/5 (19 ratings); would take again 37%; difficulty 4.0/5. Courses: CS3200 (Database Design). Source: RateMyProfessors.
```

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` (384-dim, ~256-token limit, runs locally
on CPU, no API key, no rate limit). Embeddings are L2-normalized and stored in ChromaDB with cosine
distance. Our chunks are far under the token limit, so nothing is truncated.

**Production tradeoff reflection (if cost weren't a constraint):**
- **Accuracy on domain text** — larger models (`bge-large-en-v1.5`, `e5-large-v2`, OpenAI
  `text-embedding-3-large`) score higher on MTEB and would better separate near-duplicate "caring
  professor" reviews. Worth it if retrieval precision is the bottleneck (see Q1 failure below).
- **Context length** — MiniLM's 256 tokens is fine for reviews but would truncate long syllabi/guides;
  a long-context embedder (OpenAI 8k, Voyage) only matters if the domain expands to long documents.
- **Multilingual** — MiniLM is English-centric; multilingual reviews would need
  `paraphrase-multilingual-MiniLM` / `multilingual-e5` / Cohere.
- **Latency & local-vs-API** — local MiniLM has zero network latency and no quota, ideal for a demo and
  for privacy. An API embedder adds round-trips and vendor lock-in but scales without local hardware.
  Since these reviews are already public, privacy isn't decisive; I'd switch to an API model only if its
  accuracy gain clearly beat the added latency and cost.

### Retrieval tests (top-k = 5, cosine distance; lower = closer)

**Test A — "What do students say about Nat Tuck's CS3650 Computer Systems workload?"**
| dist | source | chunk |
|------|--------|-------|
| 0.349 | Tuck CS3650 | "...extremely difficult, as if the course is made to give you no help at all." |
| 0.387 | Tuck CS3650 | "Nat boils his lectures down to the very core: if you listen, you will learn exactly what you need..." |
| 0.420 | Tuck CS5335 | "Homeworks take well over 20 hours (sometimes 30-40)..." |
| 0.438 | Lieberherr CS1101 | "Weekly assignments... shortest one took me 6 hours..." |
| 0.472 | Lerner CS2500 | "I took the accelerated section and learned a lot..." |

*Why relevant:* the top three are Tuck reviews capturing exactly the two poles students describe — "no
help / extremely difficult" vs. "boils lectures to the core, reasonable if you attend" — plus the
workload-hours review. The 4th/5th (Lieberherr, Lerner) are weaker matches pulled in only because they
also discuss "workload/assignment hours"; they sit at higher distance and don't mislead the answer.

**Test B — "Difference between the two CS3200 Database professors, Fontenot and Gatterbauer?"**
| dist | source | chunk |
|------|--------|-------|
| 0.382 | Gatterbauer CS3200 | "I really enjoyed this class and thought Wolfgang was a decent teacher..." |
| 0.423 | Gatterbauer (summary) | overall 2.6/5, would-take-again 37%, difficulty 4.0/5 |
| 0.424 | Fontenot CS3200 | "...he comes off unapproachable which reflects in the lack of participation..." |
| 0.432 | Fontenot (summary) | overall 2.8/5, would-take-again 49%, difficulty 3.2/5 |
| 0.442 | Fontenot CS3200 | "Easy class until the final project" |

*Why relevant:* because there are only two CS3200 professors and the query names both, retrieval cleanly
pulls a balanced set — a review **and** the summary chunk for each. This is the multi-entity case
working *well* (contrast with Q1 below, where one professor's reviews crowd out the others).

**Test C — "Main complaints about Karl Lieberherr's CS1100/CS1101 intro courses?"** — top results were
Lieberherr CS1100 (0.358, "worst professor... you cannot understand what he is saying") and CS1101
(0.392, "RUN. taught by claude..."), correctly surfacing the AI-teaching and comprehension complaints.

---

## Grounded Generation

Grounding is enforced on **two levels** (`generate.py`):

**1. Prompt (primary).** The system prompt restricts the model to the retrieved context and forces an
explicit refusal otherwise. Temperature is **0**. The actual instruction:

> *"You answer ONLY using the student reviews provided in the context. Use ONLY facts stated in the
> provided reviews. Do NOT use any outside or general knowledge. If the reviews do not contain enough
> information to answer, reply with exactly: 'I don't have enough information in the student reviews I
> have access to to answer that.' Never invent professor names, ratings, course codes, or quotes that
> are not in the context. When reviews disagree, represent both sides."*

**2. Pipeline (structural safeguard).** Retrieval returns the closest chunk's cosine distance; if it
exceeds **0.72** the system refuses *before* the LLM is ever called. Calibrated from observed data:
in-scope queries scored ≤0.55, out-of-scope (dining/housing) ≥0.68. So a far-off query can't be answered
from the model's training knowledge.

**How source attribution is surfaced:** sources are built **programmatically** from the retrieved
chunks' metadata (`professor` + `source_url`), never written by the LLM — so a citation can't be
hallucinated. The UI shows them in a separate "Retrieved from" box. On a refusal, the source list is
empty.

### Example responses (source attribution visible)

**Example 1 — in scope**
> **Q:** What are the main student complaints about Karl Lieberherr's CS1100/CS1101 intro courses?
> **A:** Students complain that Karl Lieberherr's CS1100 and CS1101 courses have unclear instructions, a
> huge workload with weekly assignments taking hours to complete, and unhelpful TAs. Students also
> struggle to understand his teaching and cite a steep learning curve. Exam averages are reported around
> 50, and the courses are described as a "tough grader" with high difficulty.
> **Sources:** • Karl Lieberherr — RateMyProfessors (https://www.ratemyprofessors.com/professor/430930)

**Example 2 — in scope**
> **Q:** What do students say about the workload and teaching style of Nat Tuck's CS3650 Computer Systems course?
> **A:** Students have mixed opinions. One found the course "extremely difficult" with no help, while
> another appreciated that Nat "boils his lectures down to the very core" and that the challenges are
> "tough but completely reasonable." It carries "Lots of homework," and skipping class makes it hard to
> pass — a high workload.
> **Sources:** • Nathaniel Tuck — RateMyProfessors (https://www.ratemyprofessors.com/professor/2044282)

**Example 3 — out of scope (refusal)**
> **Q:** What is the best dining hall at Northeastern?
> **A:** I don't have enough information in the student reviews I have access to to answer that.
> **Sources:** (none)

---

## Query Interface

A Gradio web app (`app.py`, served at `http://localhost:7860`).

- **Input:** a single text box, *"Your question"* (Enter or the **Ask** button submits). Five clickable
  example questions are provided for the demo.
- **Output:** two text boxes — **Answer** (the grounded response) and **Retrieved from (sources)** (the
  bulleted list of RateMyProfessors pages the answer drew from, or a note that the question is
  out of scope).

**Sample interaction transcript:**
```
Your question:  Which Fundies (CS2500) professor is rated highest, and why?

Answer:         Professor Nathaniel Derbinsky is rated the highest for CS2500 with an
                overall student rating of 4.8/5, higher than Benjamin Lerner (3.6/5) and
                Amal Ahmed (4.0/5). Students praise Derbinsky as an engaging lecturer, very
                knowledgeable, genuinely caring, and quick to give good feedback.

Retrieved from: • Benjamin Lerner — RateMyProfessors (.../professor/2034405)
                • Nathaniel Derbinsky — RateMyProfessors (.../professor/2308643)
                • Amal Ahmed — RateMyProfessors (.../professor/1626525)
```
*(This is the answer for the shorter, summary-oriented phrasing of the question, where retrieval pulls
all three professors' summary chunks. A longer, praise-oriented phrasing of the same question fails —
see Failure Case Analysis.)*

---

## Evaluation Report

All five planning.md test questions, run through the system. Retrieval = whether returned chunks were on
target; Accuracy = whether the response was correct against the cited files.

| # | Question | Expected answer | System response (summary) | Retrieval | Accuracy |
|---|----------|-----------------|---------------------------|-----------|----------|
| 1 | Among Fundies CS2500 profs Lerner/Derbinsky/Ahmed, who is rated highest & praised? | Derbinsky highest (4.8 overall) vs Ahmed 4.0, Lerner 3.6; praised: engaging, caring, good feedback | Named Derbinsky highest + correct praise, but claimed **Lerner also 5.0/5** (used per-review scores) and said **"no information about Ahmed"** | Off-target (Ahmed dropped) | **Partially accurate — failure case** |
| 2 | Workload & teaching style of Tuck's CS3650? | Very hard, lots of homework, polarized: distills lectures vs. won't explain | Captured both poles + high workload accurately | Relevant | **Accurate** |
| 3 | Which prof is most often described as caring / gives good-affirming feedback? | Rajaraman (kind/affirming, 100% WTA); Derbinsky ("gives good feedback") | Named Derbinsky (feedback) + Rajaraman (affirming) correctly, but listed 4 caring profs without ranking a clear "most" | Relevant (thematic) | **Partially accurate** |
| 4 | Difference between CS3200 DB profs Fontenot vs Gatterbauer? | Gatterbauer easier/intuitive but lower-rated detractors; Fontenot the harder/"learn more" option, called unapproachable by some | Correct qualitative contrast, but compared **per-review** scores (Gatterbauer 5.0 vs Fontenot 1.0–3.0), implying Gatterbauer rated higher when overall they're ~equal (2.6 vs 2.8) | Relevant (both profs) | **Partially accurate** |
| 5 | Main complaints about Lieberherr's CS1100/1101? | Heavy AI-based teaching, 6–13h assignments, low exam averages, hard to understand, unhelpful TAs | All complaints captured and grounded | Relevant | **Accurate** |

*Result: 2 accurate, 3 partially accurate.* The partials are genuine — none of the three is "wrong about
everything," but each shows a real, explainable limitation rather than a suspiciously perfect score.

---

## Failure Case Analysis

**Question that failed:** Q1 — *"Among the Fundies (CS2500) professors Lerner, Derbinsky, and Ahmed, who
has the highest student rating, and what do students praise about them?"*

**What the system returned:** It named Derbinsky as top with praise (correct), but also claimed **Benjamin
Lerner has a "perfect quality rating of 5.0/5"** and stated **"There is no information available about a
professor named Ahmed teaching CS2500."** Both are wrong: Lerner's *overall* rating is 3.6/5 (the lowest
of the three), and Ahmed is in the corpus.

**Root cause (tied to a specific pipeline stage — retrieval):** With global **top-k = 5**, the phrase
*"what do students praise"* matched Lerner's many praise-heavy review chunks (he has 90 ratings, lots of
glowing reviews), so **4 of the 5 retrieved chunks were Lerner**, 1 was Derbinsky, and **Ahmed's chunks
never made the top 5 at all.** The model couldn't mention Ahmed because Ahmed wasn't in its context — a
retrieval-coverage failure, not a generation failure. The secondary error (Lerner "5.0/5") is a
downstream effect: with no summary chunk for Derbinsky/Ahmed retrieved, the model fell back on the
*per-review* quality numbers visible in the Lerner review chunks instead of the *overall* rating.

This is exactly the risk flagged in planning.md ("a global top-5 can be dominated by one professor").
Note it's phrasing-sensitive: a shorter, summary-oriented phrasing of the same question retrieves all
three professors' summary chunks and answers correctly (see the interface transcript).

**What I would change to fix it:** retrieve *per source* for comparative queries (e.g. top-2 chunks for
each professor named, or a small top-k per professor) so every entity is represented; or detect
multi-entity questions and raise k; or add hybrid/keyword search (BM25) so exact surnames like "Ahmed"
guarantee at least one chunk from each. A stronger embedding model would also separate the summary
chunks better so they aren't crowded out by review chunks.

---

## Spec Reflection

**One way the spec helped me during implementation:** Writing the Chunking Strategy section *first* —
specifically the decisions to split on whole reviews with zero overlap and to **prefix each chunk with
professor + course + scores** — meant the chunk code was unambiguous to write and the chunks came out
self-contained on the first try (every retrieved chunk names its professor, so source attribution and
relevance "just worked"). Pre-writing the five evaluation questions also kept retrieval honest: I built
and tuned against real target queries instead of inventing easy ones afterward.

**One way my implementation diverged from the spec, and why:** The spec's Retrieval Approach only
described semantic top-k = 5. During M5 I **added a distance gate** (refuse before calling the LLM if the
best cosine distance > 0.72) after measuring that in-scope queries scored ≤0.55 and out-of-scope ones
≥0.68. It wasn't in the original plan, but the empirical separation was clean enough that a structural
refusal made the system noticeably more trustworthy than relying on the prompt alone. I updated
planning.md's Retrieval Approach to record this change.

---

## AI Usage

**Instance 1 — Implementing the chunking pipeline.**
- *What I gave the AI:* My planning.md **Chunking Strategy** section (one chunk per review, overlap 0,
  prefix professor/course/scores, summary chunk per professor) plus a sample `.txt` file showing the
  `header + REVIEW |` format.
- *What it produced:* `ingest.py` (header/review parser) and `chunk.py` (chunker + ChromaDB-friendly flat
  metadata).
- *What I changed/overrode:* I required the metadata be ChromaDB-safe up front (no `None`/lists — missing
  scores stored as `-1.0`, tags as a comma string) so the same dicts feed straight into the vector store
  in M4; and I added a printed self-check (total count, char min/max/mean, empty-chunk count, 5 random
  chunks) because the spec's checkpoint demands inspecting chunks before embedding. Verified output: 71
  chunks, 0 empty.

**Instance 2 — Grounded generation + refusal.**
- *What I gave the AI:* My grounding requirement (answer only from context; exact refusal string; sources
  added programmatically) and the desired `{answer, sources, chunks}` contract.
- *What it produced:* `generate.py` with the Groq call, a strict system prompt, and a programmatic source
  list.
- *What I changed/overrode:* I added the **distance-gate** refusal (not in the first draft) after testing
  out-of-scope queries, and made `load_dotenv()` read the `.env` next to the script (not the CWD) after a
  path bug. I also kept the source list keyed to *retrieved* chunks and documented the honest consequence
  — a retrieved-but-unused chunk (e.g. a Lieberherr chunk on the Tuck question) can appear in the source
  list even when it didn't shape the answer.

---

## Stretch: Chunking Strategy Comparison (+1)

To validate the review-level chunking decision, I compared it head-to-head against a naive fixed-size
splitter on the same five evaluation queries, with the same embedding model (`compare_chunking.py`,
in-memory cosine retrieval so it doesn't touch the production index).

- **Strategy A — review-level (this project):** one chunk per review, prefixed with
  professor/course/scores, plus a per-professor summary chunk. **71 chunks.**
- **Strategy B — naive fixed-size:** 500-character windows with 50-character overlap over each raw
  document, ignoring review boundaries. **70 chunks.**

**Metric:** each query has known professor file(s) that should answer it. I retrieve the top-5 chunks
for each strategy and count how many come from an expected file (**hits@5**) and whether the #1 chunk
does (**top-1**). Higher = retrieval surfaced the right professor.

| Query | A hits@5 | A top-1 | B hits@5 | B top-1 |
|-------|:---:|:---:|:---:|:---:|
| Q1 Fundies trio (Lerner/Derbinsky/Ahmed) | 5/5 | ✓ lerner | 4/5 | ✓ derbinsky |
| Q2 Tuck CS3650 | 4/5 | ✓ tuck | 2/5 | ✓ tuck |
| Q3 caring / good feedback | 2/5 | ✓ rajaraman | 2/5 | ✗ razzaq |
| Q4 CS3200 DB profs | 5/5 | ✓ gatterbauer | 5/5 | ✓ fontenot |
| Q5 Lieberherr complaints | 3/5 | ✓ lieberherr | **0/5** | **✗ fontenot** |
| **Total** | **19/25 (76%)** | **5/5** | **13/25 (52%)** | **3/5** |

**Which performed better, and why:** Review-level chunking won — **76% vs 52% hits@5, and 5/5 vs 3/5
top-1.** The decisive case is **Q5**: the fixed-size splitter scored **0/5** and its top result was a
*Fontenot* chunk, not Lieberherr at all. The cause is structural — a professor's name appears only once
per file (in the header), so once the file is sliced into 500-char windows, nearly every window is
anonymous review text containing no "Lieberherr" / "CS1100" token; those windows match a query that
names the professor poorly, and an unrelated chunk that merely shares words like "complaints" / "course"
outranks them. **Q2** (Tuck 4/5 vs 2/5) and **Q3** (top-1 ✓ vs ✗, where B wrongly led with Razzaq) show
the same effect more mildly. Review-level chunking avoids it because every chunk is prefixed with the
professor and course, so entity queries reliably land on the right person — the exact reasoning in
planning.md, now confirmed with data. (Run `python compare_chunking.py` to reproduce.)

---

## Stretch: Hybrid Search — BM25 + Semantic (+2)

**Approach.** `hybrid.py` runs two retrievers over the same 71 chunks and fuses them:
- **BM25** (lexical, via `rank_bm25`) — strong on exact tokens like surnames ("Ahmed") and course codes
  ("CS3200").
- **Semantic** (all-MiniLM-L6-v2 cosine) — strong on meaning without shared words.

The two rankings are merged with **Reciprocal Rank Fusion**: each chunk scores
`1/(60 + rank_semantic) + 1/(60 + rank_bm25)`, so a chunk that ranks well in *either* method rises — no
score normalization needed. It's exposed as `ask(question, use_hybrid=True)` and a checkbox in the
Gradio UI.

**Comparison on 3 queries** (top-5 retrieved professors per method):

| Query | Semantic | BM25 | Hybrid |
|-------|----------|------|--------|
| Q1 Fundies (Lerner/Derbinsky/Ahmed) | Lerner ×4, Derbinsky ×1 — **Ahmed absent** | Lerner ×2, Derbinsky ×1, **Ahmed ×2** | **Derbinsky ×2**, Lerner ×2, **Ahmed ×1** |
| Q3 caring / good feedback | Rajaraman only #4; led by Hescott | **Rajaraman #1**, Derbinsky, +Choffnes ("good feedback" tag) | Derbinsky #1, **Rajaraman #2** |
| Q4 CS3200 DB profs | Gatterbauer ×2, Fontenot ×3 | Gatterbauer ×3, Fontenot ×2 | Gatterbauer ×3, Fontenot ×2 |

**Which performed better, and why.** **Hybrid was the best all-rounder.** On **Q1**, pure semantic search
was dominated by Lerner's many praise-heavy reviews and dropped **Ahmed entirely** while burying the
actually-highest-rated professor (Derbinsky) at rank 3; **BM25** recovered Ahmed via the exact surname
token but didn't rank the best professor first; **hybrid** did both — it lifted Derbinsky to #1 *and*
kept Ahmed in the top-5. On **Q3**, BM25's exact match on "affirming/caring" surfaced Rajaraman at #1
(semantic had him at #4), though BM25 also pulled in Choffnes purely because a review carries the "Gives
good feedback" tag; hybrid kept Rajaraman near the top without that noise. On **Q4** all three were
comparable, because naming both professors already constrains retrieval well.

**The payoff — hybrid fixes the documented Q1 failure case.** Same eval question, generation end-to-end:

> **Semantic (default):** "Derbinsky has a perfect quality rating of 5.0/5... Lerner also received a
> perfect quality rating of 5.0/5... **There is no information about a professor named Ahmed.**"
> — *wrong: drops Ahmed, mis-ranks Lerner using per-review scores.*
>
> **Hybrid:** "Nathaniel Derbinsky has a perfect quality rating of 5.0/5... In contrast, **Benjamin
> Lerner has an overall student rating of 3.6/5, and Amal Ahmed has a rating of 4.0/5.**"
> — *correct three-way comparison with the right overall ratings.*

Reproduce with `python hybrid.py` (retrieval comparison) or toggle the hybrid checkbox in `app.py`.

---

## Stretch: Metadata Filtering (+1)

Every chunk stores flat metadata (`professor`, `course`, `quality`, `difficulty`, `chunk_type`,
`source_file`, …), so retrieval takes an optional ChromaDB `where` filter — `retrieve(query, k,
where=…)` and `ask(question, where=…)`. The Gradio UI exposes a **"Filter reviews by rating"** dropdown
(All / positive ≥ 4 / critical ≤ 2).

**Visible effect** — same query *"What is the teaching style and workload like?"*, top-5 retrieved:

| Filter | Returned chunks (professor · rating) |
|--------|--------------------------------------|
| *none* | Lieberherr·1, Choffnes·5, Choffnes·5, Razzaq·2, Tuck·1 — **mixed** |
| `professor == "Mark Fontenot"` | Fontenot·4, Fontenot·3, Fontenot·1, Fontenot·5, Fontenot·(summary) — **one professor** |
| `quality >= 4` (positive only) | Choffnes·5, Choffnes·5, Annunziato·5, Tuck·4, Gatterbauer·5 — **all 4–5★** |
| `chunk_type == review AND quality <= 2` (critical only) | Lieberherr·1, Razzaq·2, Tuck·1, Annunziato·1, Tuck·2 — **all 1–2★** |

The filter visibly changes what comes back: restricting to one professor returns only their reviews,
and the rating filters flip the *same* query between all-positive and all-critical reviews — handy for
"what do happy students say vs. what do critics say." (When a filter is set the system uses the semantic
path, where ChromaDB applies the `where` clause natively.)

---

## Stretch: Conversational Memory (+1)

Multi-turn follow-ups are resolved *before* retrieval: `ask_chat(question, history)` (in `generate.py`)
uses the LLM to rewrite a context-dependent follow-up into a standalone question, so a pronoun like
"his" still retrieves the right professor. `chat.py` runs the CLI demo; `chat_app.py` is a Gradio chat UI.

**Demo transcript** — the rewritten query is shown to prove this is memory, not topic overlap:

> **User:** What do students say about Nat Tuck's teaching style?
> **Assistant:** Mixed — some praise him for boiling lectures down to the core; others find him
> unhelpful and unwilling to explain concepts from scratch. *(source: Tuck)*
>
> **User:** Is his workload heavy?
> &nbsp;&nbsp;*(rewritten with memory → "Is **Nat Tuck's** workload heavy?")*
> **Assistant:** Yes — "Lots of homework," and CS5335 homeworks "take well over 20 hours (sometimes
> 30–40)." *(source: Tuck)*
>
> **User:** How does that compare to Derbinsky?
> &nbsp;&nbsp;*(rewritten → "How does the workload in **Nat Tuck's** classes compare to **Derbinsky's**?")*
> **Assistant:** Tuck's load is heavier (20–40h homeworks); Derbinsky's CS2500 has a lower difficulty
> (3.0/5). *(sources: Tuck, Derbinsky)*

Turn 2's "his" and turn 3's "that" are resolved to the professors named earlier — the response reflects
the conversation, not a coincidental topic match. Grounding is preserved: the rewritten question still
goes through the same retrieve-then-answer-from-context pipeline.

---

## Pipeline / Files

`ingest.py` (load + parse) → `chunk.py` (chunk) → `embed_store.py` (embed with all-MiniLM-L6-v2 + store
in ChromaDB) → `retrieve.py` (semantic top-k) → `generate.py` (Groq, grounded) → `app.py` (Gradio UI).
Stretch modules: `compare_chunking.py` (chunking comparison), `hybrid.py` (BM25 + semantic), and
`chat.py` / `chat_app.py` (conversational memory); metadata filtering lives in `retrieve.py` /
`generate.py`. Each module runs standalone for inspection (e.g. `python chunk.py` prints chunk stats;
`python retrieve.py` and `python hybrid.py` print retrieval for sample queries; `python chat.py` runs a
multi-turn demo). See `planning.md` for the architecture diagram.
