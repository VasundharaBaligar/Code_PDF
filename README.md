# HyperscaleES Explorer

Two-pane app for exploring [ESHyperscale/HyperscaleES](https://github.com/ESHyperscale/HyperscaleES)
(the EGGROLL evolution-strategies codebase) and its paper — a GitHub-style repo browser
on the left, a chat assistant on the right that answers grounded in the real repo and
paper content. $0 to run.

**Live:** https://hyperscalees-explorer.onrender.com

## Stack

FastAPI + vanilla JS (no Node, no build step) · BM25 search, no vector DB · Groq
(`openai/gpt-oss-120b`, free) primary with local Ollama fallback · KaTeX for math ·
Render free tier.

## Run locally

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

Works with zero external accounts via [Ollama](https://ollama.com) (`ollama pull qwen2.5-coder:7b`),
or faster with a free key from [console.groq.com](https://console.groq.com/keys) in `.env`.

## Deploy

`render.yaml` included — Render free tier, **New +** → **Blueprint**.
