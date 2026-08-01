"""
Offline step: turn the cached repo files into search-ready chunks.

Python files are split at function/class boundaries (via `ast`), notebooks
are split per cell, everything else is split into overlapping line windows.
Output is written to data/index/chunks.jsonl for app/retrieval.py to load.

Usage:
    python scripts/build_index.py
"""

import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db
from app.config import CHUNKS_PATH, INDEX_DIR

# Not useful to search/retrieve: legal boilerplate and tooling config.
# The two large tokenizer files are already content=NULL and skip naturally.
EXCLUDED_PATHS = {"LICENSE", ".gitignore"}

WINDOW_LINES = 60
OVERLAP_LINES = 10


@dataclass
class Chunk:
    path: str
    chunk_id: int
    start_line: int
    end_line: int
    text: str


def chunk_python_file(path: str, text: str) -> list[Chunk]:
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunk_generic(path, text)

    top_level_defs = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not top_level_defs:
        return chunk_generic(path, text)

    chunks: list[Chunk] = []

    # Header chunk: imports, module docstring, constants before the first def/class.
    first_start = top_level_defs[0].lineno
    if first_start > 1:
        header_text = "\n".join(lines[: first_start - 1]).strip()
        if header_text:
            chunks.append(Chunk(path, len(chunks), 1, first_start - 1, header_text))

    for node in top_level_defs:
        start, end = node.lineno, node.end_lineno or node.lineno
        snippet = "\n".join(lines[start - 1 : end])
        chunks.append(Chunk(path, len(chunks), start, end, snippet))

    return chunks


def chunk_notebook(path: str, text: str) -> list[Chunk]:
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError:
        return chunk_generic(path, text)

    chunks: list[Chunk] = []
    for idx, cell in enumerate(notebook.get("cells", [])):
        source = cell.get("source", [])
        cell_text = ("".join(source) if isinstance(source, list) else str(source)).strip()
        if not cell_text:
            continue
        cell_type = cell.get("cell_type", "code")
        chunks.append(Chunk(path, len(chunks), idx, idx, f"[{cell_type} cell {idx}]\n{cell_text}"))

    return chunks


def chunk_generic(path: str, text: str) -> list[Chunk]:
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    stride = WINDOW_LINES - OVERLAP_LINES
    start = 0
    while start < len(lines):
        end = min(start + WINDOW_LINES, len(lines))
        snippet = "\n".join(lines[start:end]).strip()
        if snippet:
            chunks.append(Chunk(path, len(chunks), start + 1, end, snippet))
        if end == len(lines):
            break
        start += stride

    return chunks


def chunk_file(path: str, text: str) -> list[Chunk]:
    if path.endswith(".py"):
        return chunk_python_file(path, text)
    if path.endswith(".ipynb"):
        return chunk_notebook(path, text)
    return chunk_generic(path, text)


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks: list[Chunk] = []
    indexed_files, skipped_files = 0, 0

    for row in db.list_files():
        path = row["path"]
        if path in EXCLUDED_PATHS:
            skipped_files += 1
            continue

        file_row = db.get_file(path)
        if file_row is None or file_row["content"] is None:
            skipped_files += 1
            continue

        chunks = chunk_file(path, file_row["content"])
        all_chunks.extend(chunks)
        indexed_files += 1

    with open(CHUNKS_PATH, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk)) + "\n")

    print(f"Indexed {len(all_chunks)} chunks from {indexed_files} files ({skipped_files} skipped).")
    print(f"Written to: {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
