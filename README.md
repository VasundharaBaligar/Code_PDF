# HyperscaleES Explorer

A two-pane web app for exploring [ESHyperscale/HyperscaleES](https://github.com/ESHyperscale/HyperscaleES)
(a JAX codebase implementing **EGGROLL**, a method for scaling Evolution Strategies to
billion-parameter models) and the research paper behind it. A GitHub-style repo browser
on the left, a Claude-style chat assistant on the right that answers questions grounded
in the actual repo content and the paper — citing real files, real equations, and real
line numbers instead of guessing.

**Live:** https://hyperscalees-explorer.onrender.com
**Code:** https://github.com/VasundharaBaligar/Code_PDF

Built to cost **$0 to run, forever** — no paid APIs, no paid hosting, no credit card
anywhere in the stack.

---

## What it does

- Browse the repo's file tree, README, and every source file with syntax highlighting —
  styled like GitHub's own dark mode.
- Ask the assistant anything: how to install the repo, what a function does, why a
  design decision was made, how to fix an error, or what a theorem in the paper proves.
- Answers stream in live, cite their exact sources (file paths, paper section +
  equation/theorem labels), and render real LaTeX math via KaTeX.
- Click a citation to jump the left pane straight to that file.
- Conversation persists across page reloads; a "New Chat" button resets it.

---

## Architecture

```
┌─────────────────────────┐        ┌──────────────────────────┐
│   Left pane (browser)    │        │   Right pane (chat)       │
│   plain HTML/CSS/JS       │        │   plain HTML/CSS/JS       │
└────────────┬─────────────┘        └────────────┬──────────────┘
             │                                     │
             └────────────────┬────────────────────┘
                               │
                      FastAPI backend (Python)
                               │
              ┌────────────────┼────────────────┐
              │                │                 │
        SQLite cache     BM25 search       LLM providers
      (repo + paper       index (in-      Groq (primary) →
       content, one-      memory,~400      Ollama (local
       time ingested)      chunks)          fallback)
```

**No Node, no build step, no vector database.** The frontend is static files served
directly by FastAPI. Search is classic keyword search (BM25) over an in-memory index —
the corpus is small enough (~400 chunks) that a vector DB would be pure overhead.

### Stack

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI + Python 3.13 | Simple, async-native, no framework ceremony |
| Frontend | Vanilla HTML/CSS/JS | No Node install, no build step, easy to reason about |
| Search | `rank-bm25` (pure Python) | Corpus is tiny; a vector DB/embeddings model is overkill |
| Primary LLM | Groq, `llama-3.3-70b-versatile` | Free tier, no card, fast, 12K tokens/min |
| Fallback LLM | Ollama, `qwen2.5-coder:7b` | Always available locally, zero external dependency |
| Math rendering | KaTeX (vendored) | Renders synchronously — matters since answers redraw on every streamed token |
| Hosting | Render.com free tier | No card, real Python runtime (unlike sandboxed alternatives) |

---

## How it works

### 1. Ingestion (one-time, runs at build/deploy time)

- **`scripts/ingest_repo.py`** — one GitHub API call for the file tree, then raw file
  contents from `raw.githubusercontent.com` (doesn't count against the API rate limit).
  Cached into `data/repo_cache.sqlite3`. The app never calls GitHub again at runtime.
- **`scripts/ingest_paper.py`** — downloads the paper's actual **LaTeX source** from
  arXiv's e-print endpoint (not the PDF — no OCR, equations arrive as real LaTeX). Strips
  LaTeX comments (draft text the authors deleted shouldn't be citable) and resolves
  `\input{}` directives in true document order, including a case-insensitive fallback
  (the source has `\input{Sections/04_low_rank_ES}` pointing at a file actually named
  `04_low_rank_es.tex` — fine on a case-insensitive filesystem, silently broken on a
  case-sensitive one).

### 2. Indexing (`scripts/build_index.py`)

Different content types get chunked differently, each split at *semantic* boundaries —
never arbitrary line counts, so nothing gets cut mid-thought:

- **Python** — split by function/class via the `ast` module. Also windows the code
  *between and after* functions (module-level config, training loops), not just
  function bodies — the initial version silently dropped ~35% of a script-style file's
  lines this way.
- **Notebooks** — split per cell.
- **LaTeX** — split at section/subsection headings and theorem/lemma/proof
  environments, carrying forward each chunk's `\label{}` anchor and a
  `Section > Subsection` breadcrumb.
- **Everything else** — overlapping line windows.

### 3. Retrieval (`app/retrieval.py`)

BM25 keyword search, with several targeted layers on top to compensate for BM25's
blind spots:

- **Filename-aware boost** — if a query names a file (exact, typo, or a bare name with
  no extension like "sft_evol"), that file's chunks are guaranteed a slice of the
  context budget regardless of BM25 rank.
- **Identifier-aware boost** — class/function names route to the file that defines
  them, so conceptual questions ("how does the EGGROLL noiser work?") reach the real
  implementation even when no filename is mentioned. Handles name collisions (a file
  that reimplements another's exact class names) by preferring whichever one the rest
  of the repo actually imports.
- **Cross-corpus balancing** — guarantees both code and paper are represented when a
  question spans both, so one doesn't crowd the other out entirely.
- **Enumeration handling** — "how many functions in X" gets the whole file (not a
  sample) plus a fact computed by literally parsing the file with `ast` — never left to
  an LLM to count by eye.
- **Cross-reference fact-checking** — "is this file imported elsewhere?" is answered by
  grepping every other cached file's real content, not by letting the model guess.
- **Stemming + domain synonyms** — conservative plural-stripping, plus a small table
  (`RNN`↔`recurrent`) for concept pairs that share no substring a stemmer could ever
  connect.
- **Conversation-aware, but conditionally** — recent history only gets folded into the
  search query when the current message actually looks referential ("this file", "it").
  Doing it unconditionally let old context hijack unrelated new questions.

### 4. Answering (`app/providers/`)

Groq is tried first; if its very first token fails (missing key, rate limit, network),
the router falls back to Ollama for the entire response — never a broken partial answer
under the wrong label. The prompt distinguishes `CODE` vs `PAPER` context explicitly,
requires LaTeX to be preserved (not flattened to unicode), and instructs the model to
read every provided chunk rather than just the first relevant-looking one.

### 5. Frontend (`frontend/`)

- **Left pane** — GitHub's own dark-mode palette; breadcrumb-based directory navigation
  (not a giant expand-all tree); server-side syntax highlighting via Pygments for files,
  a small custom Org-mode→HTML renderer for the README.
- **Right pane** — streaming NDJSON chat; markdown + KaTeX rendering (math only
  re-rendered once a message completes, not per-token, to avoid flicker on
  not-yet-complete equations); citation chips that jump the left pane to the cited file;
  `localStorage` persistence with an explicit "New Chat" reset.

---

## Running it yourself

```bash
git clone https://github.com/VasundharaBaligar/Code_PDF.git hyperscalees-explorer
cd hyperscalees-explorer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/ingest_repo.py     # one-time: cache the GitHub repo
python scripts/ingest_paper.py    # one-time: cache the paper
python scripts/build_index.py     # one-time: build the search index

uvicorn app.main:app --port 8000
```

Open `http://localhost:8000`. Works immediately with **zero external accounts** if you
have [Ollama](https://ollama.com) running locally:

```bash
brew install ollama && brew services start ollama
ollama pull qwen2.5-coder:7b
```

For faster answers, get a free key at [console.groq.com](https://console.groq.com/keys)
(no card) and add it to `.env` (copy `.env.example`):

```bash
GROQ_API_KEY=your_key_here
```

With no key set, everything runs on Ollama alone — still $0.

## Deploying your own copy

The included `render.yaml` deploys straight to [Render](https://render.com)'s free tier:
**New +** → **Blueprint** → connect your fork. Render reads the build/start commands
automatically; you'll be prompted for `GROQ_API_KEY` in the dashboard (kept out of the
repo). The hosted deployment runs Groq-only — no Ollama fallback there, by design, since
Render's free tier has no room for a local model.

---

## Known limitations

- **Groq's free tier is shared** across everyone using the hosted link — 12,000
  tokens/minute total. Heavy simultaneous use falls back to Ollama-quality answers (or,
  on the hosted deployment with no Ollama, a "try again in a minute" message) until the
  window clears.
- **Model synthesis isn't perfect.** Retrieval can correctly surface the right chunk and
  the model still occasionally only reads the first one instead of scanning all of them
  — a real capability limit of a free 70B model under competing context, not a retrieval
  bug. Deterministic checks (function counts, cross-reference facts) exist specifically
  to route around this where possible; open-ended "why X not Y" reasoning has no
  equivalent shortcut.
- **The paper's figures and bibliography aren't indexed** — text and equations only, by
  design (figures/bib are a lot of low-value bulk for a Q&A assistant; can be added
  later without touching anything else).
- **No paragraph-level citation highlighting yet** — citation chips jump to a *file*,
  not a *passage*. The paper's LaTeX `\label{}` anchors already exist for this; wiring
  it into the UI is the next planned step (see Roadmap).

---

## Notable bugs found and fixed along the way

Kept for the record — most of these were caught by actually testing real questions
against real answers, not by inspection:

- **Chunker silently dropped ~35% of script-style files.** Only function/class bodies
  and the pre-first-function header were indexed; module-level code *between and after*
  functions (config, training loops) was invisible. This is exactly why the assistant
  once hallucinated three functions that don't exist — it saw them *called* inside
  indexed function bodies but never saw their real `x = jax.jit(...)` definitions,
  which lived in the unindexed gaps.
- **Case-sensitive filename matching.** Two of three file-lookup tiers never lowercased
  the query token before comparing against filenames. `"RWKV"` vs `"rwkv7"` scored 0.0
  similarity under case-sensitive matching (0.89 lowercased) — any capitalized or
  acronym file mention was silently falling through to plain BM25 instead of getting
  the filename-match guarantee.
- **Unconditional history-folding hijacked unrelated questions.** A fix for legitimate
  follow-ups ("tell me more about *this file*") backfired the moment a conversation
  moved to a new topic — an old file mention sitting in history several turns back
  silently redirected a completely unrelated new question.
- **One corpus starved the other.** Before cross-corpus balancing, a question naming a
  code file would let the filename boost claim the entire context budget, so a question
  explicitly about "the paper's equation" could retrieve *zero* paper content.
- **BM25's lexical blind spot.** The paper's actual primary explanation for a design
  choice ranked 21st for a plausible paraphrase of the question — the query said
  "transformer" (singular) and "RNN"; the text said "transformers" (plural) and
  "recurrent"/"RWKV", never that acronym. Fixed with conservative stemming and a small
  synonym table, not a full semantic-search rebuild (checked: Groq has no embeddings
  endpoint, and the hosted deployment deliberately has no Ollama, so real semantic
  search isn't consistently deployable everywhere without a fourth external account).
- **`arXiv` LaTeX source `\input` case mismatch.** `main.tex` referenced
  `Sections/04_low_rank_ES` (capital ES); the actual file is `04_low_rank_es.tex`. Fine
  on the authors' case-insensitive filesystem, silently dropped the entire EGGROLL
  algorithm section from a case-sensitive tar archive.
- **LaTeX comments would have been indexed as real content.** 83 commented-out lines
  exist in the paper source, including 3 figure references that exist *only* inside
  comments — without stripping, the assistant could have cited draft text the authors
  deliberately deleted.
- **Render deploy failure from a Python version mismatch.** Render defaulted to Python
  3.14; `pydantic-core` had no prebuilt wheel for it, fell back to compiling from source,
  and failed because Render's build sandbox has a read-only filesystem for the Rust
  compiler's cache. Fixed by pinning `PYTHON_VERSION` explicitly.

---

## Roadmap

- **Paragraph-level citation highlighting** — modeled on how alphaXiv's paper reader
  works: hovering/clicking a citation scrolls to and highlights the exact referenced
  passage, not just the top of a file. The paper's `\label{}` anchors (81 of 150 paper
  chunks carry one) exist specifically to make this possible.
- **Figure rendering** in the paper viewer.
- **Bibliography indexing** (titles/authors only, to stay lightweight) if citation-chase
  questions turn out to matter in practice.
