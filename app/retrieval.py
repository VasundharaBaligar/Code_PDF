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
_ENUMERATION_RE = re.compile(
    r"\bhow many\b|\bcount\b|\blist all\b|\benumerate\b|\ball (the )?functions\b|"
    r"\ball (the )?classes\b|\ball (the )?methods\b",
    re.IGNORECASE,
)


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
_identifier_to_paths: dict[str, list[str]] | None = None


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _build_identifier_index() -> dict[str, list[str]]:
    """
    Maps class/function names (e.g. "eggroll" -> the EggRoll class) to the
    file(s) that define them. Conceptual questions ("how does the EGGROLL
    noiser work") name a *concept*, not a file -- BM25 alone routes these to
    whichever file uses that word the most in prose (usually the README or
    notebook), not the actual implementation, since it has no idea "EggRoll"
    is a class defined in eggroll.py.

    A name can collide across files (e.g. alteggroll.py reimplements
    eggroll.py's exact class/function names as an alternate version) --
    stored as a list so find_mentioned_path can disambiguate rather than
    silently picking whichever file happened to be scanned first.
    """
    import ast

    from app import db

    index: dict[str, list[str]] = {}
    for row in db.list_files_with_content():
        if not row["path"].endswith(".py"):
            continue
        try:
            tree = ast.parse(row["content"])
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                index.setdefault(node.name.lower(), []).append(row["path"])
    return index


def load_corpus() -> None:
    global _corpus, _bm25, _identifier_to_paths

    chunks: list[dict] = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    _corpus = chunks
    _bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
    _identifier_to_paths = _build_identifier_index()


def find_mentioned_path(query: str) -> str | None:
    """
    If the query names a specific file -- exactly, as a close-enough typo,
    or as a bare shortened name with no extension (e.g. "sft_evol" for
    "sft_evolution.py") -- return its full path. BM25 has no concept of
    "this literally is the file being asked about": it only ranks by
    generic word overlap, which can bury a lightly-worded file behind
    unrelated chunks that happen to share common terms (e.g. every
    ES-related file mentions "evolution").
    """
    assert _corpus is not None
    basename_to_path = {c["path"].rsplit("/", 1)[-1]: c["path"] for c in _corpus}

    # Filename-shaped tokens first (e.g. "sft_evolution.py", "stf_evolution.py").
    for token in _FILENAME_RE.findall(query):
        token_name = token.rsplit("/", 1)[-1]
        if token_name in basename_to_path:
            return basename_to_path[token_name]
        close = difflib.get_close_matches(token_name, basename_to_path.keys(), n=1, cutoff=0.72)
        if close:
            return basename_to_path[close[0]]

    # Bare-word fallback: people often drop the extension or shorten a name
    # when asking conversationally ("the sft_evol file"). Match against file
    # stems (basename without extension) instead.
    stem_to_path = {name.rsplit(".", 1)[0]: path for name, path in basename_to_path.items()}
    for token in _TOKEN_RE.findall(query):
        if len(token) < 4:
            continue
        if token in stem_to_path:
            return stem_to_path[token]
        close = difflib.get_close_matches(token, stem_to_path.keys(), n=1, cutoff=0.72)
        if close:
            return stem_to_path[close[0]]

    # Concept fallback: route by known top-level class/function names, so
    # questions about what something *does* (not what file it's in) still
    # find the file that implements it. Stricter cutoff than the filename
    # fallbacks since generic-sounding identifiers are more collision-prone.
    if _identifier_to_paths:
        for token in _TOKEN_RE.findall(query):
            if len(token) < 4:
                continue
            key = token.lower()
            candidates = _identifier_to_paths.get(key)
            if candidates is None:
                close = difflib.get_close_matches(key, _identifier_to_paths.keys(), n=1, cutoff=0.85)
                if close:
                    candidates = _identifier_to_paths[close[0]]
            if candidates:
                if len(candidates) == 1:
                    return candidates[0]
                # Name collision across files (e.g. an "alt"/experimental
                # reimplementation reusing the same names) -- prefer whichever
                # one the rest of the repo actually imports, as a proxy for
                # "the real implementation" over an unused alternate.
                return max(candidates, key=lambda p: len(find_cross_references(p)))
    return None


def find_cross_references(mentioned_path: str) -> list[str]:
    """
    Search every other cached file's actual content for the module's
    basename (without extension) as a real import/reference check --
    grounds "is this used/imported elsewhere?" questions in a verified
    fact instead of letting the model guess and hallucinate an answer.
    """
    from app import db

    module_name = mentioned_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if not module_name:
        return []

    referencing_paths = []
    for row in db.list_files_with_content():
        if row["path"] == mentioned_path:
            continue
        if module_name in row["content"]:
            referencing_paths.append(row["path"])
    return referencing_paths


def is_enumeration_query(query: str) -> bool:
    return bool(_ENUMERATION_RE.search(query))


def get_structure_note(mentioned_path: str) -> str | None:
    """
    For "how many functions/classes" questions, don't rely on a small model
    to count correctly across many scattered chunks -- even with full-file
    context, exhaustive enumeration is exactly the kind of task small LLMs
    are unreliable at. Parse the real file with `ast` (the same tool used
    for chunking) and state the actual inventory as a fact, the same way
    find_cross_references grounds import questions instead of guesswork.
    """
    import ast

    from app import db

    if not mentioned_path.endswith(".py"):
        return None

    file_row = db.get_file(mentioned_path)
    if file_row is None or file_row["content"] is None:
        return None

    try:
        tree = ast.parse(file_row["content"])
    except SyntaxError:
        return None

    functions = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]

    func_part = f"{len(functions)} top-level function(s)" + (f": {', '.join(functions)}" if functions else "")
    class_part = f"{len(classes)} top-level class(es)" + (f": {', '.join(classes)}" if classes else "")
    return f"{mentioned_path} contains exactly {func_part}; {class_part}."


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
    mentioned_path = find_mentioned_path(query)
    if mentioned_path:
        file_indices = [i for i, c in enumerate(_corpus) if c["path"] == mentioned_path]
        if _ENUMERATION_RE.search(query):
            # "How many functions" etc. needs the whole file, not a sample --
            # partial context can only ever undercount. The repo is small
            # enough that one full file is still a modest token budget.
            boosted_budget = len(file_indices)
        else:
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
