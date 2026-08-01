"""
In-memory BM25 search over the chunk corpus built by scripts/build_index.py.

The repo is tiny (~150-250 chunks), so a pure-Python keyword-matching index
loaded fresh at startup is fast and dependency-light -- no vector DB needed.
"""

import json
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.config import CHUNKS_PATH

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class RetrievedChunk:
    path: str
    chunk_id: int
    start_line: int
    end_line: int
    text: str
    score: float


_corpus: list[dict] | None = None
_bm25: BM25Okapi | None = None


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def load_corpus() -> None:
    global _corpus, _bm25

    chunks: list[dict] = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    _corpus = chunks
    _bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])


def search(query: str, k: int = 6) -> list[RetrievedChunk]:
    if _corpus is None or _bm25 is None:
        load_corpus()
    assert _corpus is not None and _bm25 is not None

    scores = _bm25.get_scores(_tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results = []
    for i in ranked_indices:
        if scores[i] <= 0:
            continue
        chunk = _corpus[i]
        results.append(
            RetrievedChunk(
                path=chunk["path"],
                chunk_id=chunk["chunk_id"],
                start_line=chunk["start_line"],
                end_line=chunk["end_line"],
                text=chunk["text"],
                score=float(scores[i]),
            )
        )
    return results
