"""
In-memory BM25 search over the chunk corpus built by scripts/build_index.py.

The repo is tiny (~150-250 chunks), so a pure-Python keyword-matching index
loaded fresh at startup is fast and dependency-light -- no vector DB needed.
"""

import difflib
import json
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.config import CHUNKS_PATH

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FILENAME_RE = re.compile(r"[\w./-]+\.\w+")


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


def _find_mentioned_path(query: str) -> str | None:
    """
    If the query names a specific file -- exactly, or a close-enough typo --
    return its full path. BM25 has no concept of "this literally is the file
    being asked about": it only ranks by generic word overlap, which can bury
    a lightly-worded file behind unrelated chunks that happen to share common
    terms (e.g. every ES-related file mentions "evolution").
    """
    assert _corpus is not None
    basename_to_path = {c["path"].rsplit("/", 1)[-1]: c["path"] for c in _corpus}

    for token in _FILENAME_RE.findall(query):
        token_name = token.rsplit("/", 1)[-1]
        if token_name in basename_to_path:
            return basename_to_path[token_name]
        close = difflib.get_close_matches(token_name, basename_to_path.keys(), n=1, cutoff=0.72)
        if close:
            return basename_to_path[close[0]]
    return None


def search(query: str, k: int = 6) -> list[RetrievedChunk]:
    if _corpus is None or _bm25 is None:
        load_corpus()
    assert _corpus is not None and _bm25 is not None

    scores = _bm25.get_scores(_tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    selected_indices: list[int] = []
    seen: set[int] = set()

    # Filename match wins a guaranteed slice of the context budget, regardless
    # of how it happens to rank on generic word overlap.
    mentioned_path = _find_mentioned_path(query)
    if mentioned_path:
        file_indices = [i for i, c in enumerate(_corpus) if c["path"] == mentioned_path]
        boosted_budget = min(len(file_indices), max(3, k // 2))
        for i in file_indices[:boosted_budget]:
            selected_indices.append(i)
            seen.add(i)

    for i in ranked_indices:
        if len(selected_indices) >= k:
            break
        if i in seen or scores[i] <= 0:
            continue
        selected_indices.append(i)
        seen.add(i)

    results = []
    for i in selected_indices:
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
