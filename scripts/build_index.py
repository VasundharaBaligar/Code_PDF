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
import re
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

# LaTeX semantic boundaries. Theorem-like environments are to a paper what
# functions are to code: self-contained units that lose meaning when split.
LATEX_HEADING_RE = re.compile(r"^\s*\\(?:sub)*section\*?\{(.+?)\}")
LATEX_SECTION_LEVEL_RE = re.compile(r"^\s*\\(sub)*section\*?\{")
LATEX_ENV_BEGIN_RE = re.compile(
    r"^\s*\\begin\{(theorem|lemma|proof|definition|assumption|corollary"
    r"|proposition|remark|algorithm)\*?\}"
)
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")


@dataclass
class Chunk:
    path: str
    chunk_id: int
    start_line: int
    end_line: int
    text: str
    # Paper chunks carry their LaTeX \label anchor (a stable, author-defined
    # citation target) and a Section > Subsection breadcrumb for context.
    # Both stay None for code chunks.
    label: str | None = None
    heading: str | None = None


def _windowed_chunks(path: str, lines: list[str], start_line: int, end_line: int, chunk_id_offset: int) -> list[Chunk]:
    """Split lines[start_line-1:end_line] (1-indexed, inclusive) into ~WINDOW_LINES chunks."""
    chunks: list[Chunk] = []
    stride = WINDOW_LINES - OVERLAP_LINES
    pos = start_line
    while pos <= end_line:
        end = min(pos + WINDOW_LINES - 1, end_line)
        snippet = "\n".join(lines[pos - 1 : end]).strip()
        if snippet:
            chunks.append(Chunk(path, chunk_id_offset + len(chunks), pos, end, snippet))
        if end == end_line:
            break
        pos += stride
    return chunks


def chunk_python_file(path: str, text: str) -> list[Chunk]:
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunk_generic(path, text)

    if not tree.body:
        return chunk_generic(path, text)

    # Walk top-level statements in order. Each function/class gets its own
    # chunk; everything else (imports, config, and -- critically for
    # script-style files -- module-level code *between and after* defs,
    # like `foo = jax.jit(...)` calls or the actual training loop) gets
    # windowed too, so no line of the file is silently left unindexed.
    chunks: list[Chunk] = []
    cursor = 1

    def flush_module_level(start_line: int, end_line: int) -> None:
        if start_line > end_line:
            return
        chunks.extend(_windowed_chunks(path, lines, start_line, end_line, len(chunks)))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            flush_module_level(cursor, node.lineno - 1)
            end = node.end_lineno or node.lineno
            snippet = "\n".join(lines[node.lineno - 1 : end])
            chunks.append(Chunk(path, len(chunks), node.lineno, end, snippet))
            cursor = end + 1

    flush_module_level(cursor, len(lines))

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
    return _windowed_chunks(path, lines, 1, len(lines), 0)


def _latex_blocks(lines: list[str]) -> list[tuple[int, int, str | None]]:
    """
    Split LaTeX into (start_line, end_line, heading) semantic blocks.

    Boundaries are headings and theorem-like environments. A theorem/proof is
    consumed whole through its \\end{}, so a proof never gets split down the
    middle -- the same reason we split Python at function boundaries rather
    than at arbitrary line counts.
    """
    blocks: list[tuple[int, int, str | None]] = []
    section: str | None = None
    subsection: str | None = None
    block_start = 1
    i = 0

    def heading() -> str | None:
        parts = [p for p in (section, subsection) if p]
        return " > ".join(parts) if parts else None

    def close(end_line: int) -> None:
        nonlocal block_start
        if end_line >= block_start:
            blocks.append((block_start, end_line, heading()))
        block_start = end_line + 1

    while i < len(lines):
        line = lines[i]
        lineno = i + 1

        heading_match = LATEX_HEADING_RE.match(line)
        if heading_match:
            close(lineno - 1)
            level_match = LATEX_SECTION_LEVEL_RE.match(line)
            is_subsection = bool(level_match and level_match.group(1))
            if is_subsection:
                subsection = heading_match.group(1)
            else:
                section, subsection = heading_match.group(1), None
            block_start = lineno
            i += 1
            continue

        env_match = LATEX_ENV_BEGIN_RE.match(line)
        if env_match:
            close(lineno - 1)
            env = env_match.group(1)
            end_re = re.compile(rf"\\end\{{{env}\*?\}}")
            j = i
            while j < len(lines) and not end_re.search(lines[j]):
                j += 1
            env_end = min(j + 1, len(lines))
            blocks.append((lineno, env_end, heading()))
            block_start = env_end + 1
            i = env_end
            continue

        i += 1

    close(len(lines))
    return blocks


def chunk_latex_file(path: str, text: str) -> list[Chunk]:
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    for start_line, end_line, heading in _latex_blocks(lines):
        block_lines = lines[start_line - 1 : end_line]
        if not "\n".join(block_lines).strip():
            continue

        # Long prose runs still get windowed so no single chunk dominates the
        # context budget; theorem/proof blocks are almost always under this.
        for windowed in _windowed_chunks(path, lines, start_line, end_line, len(chunks)):
            label_match = LABEL_RE.search(windowed.text)
            chunks.append(
                Chunk(
                    path=windowed.path,
                    chunk_id=len(chunks),
                    start_line=windowed.start_line,
                    end_line=windowed.end_line,
                    text=windowed.text,
                    label=label_match.group(1) if label_match else None,
                    heading=heading,
                )
            )

    return chunks


def chunk_file(path: str, text: str) -> list[Chunk]:
    if path.endswith(".py"):
        return chunk_python_file(path, text)
    if path.endswith(".ipynb"):
        return chunk_notebook(path, text)
    if path.endswith(".tex"):
        return chunk_latex_file(path, text)
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

    code_chunks = len(all_chunks)

    # The paper's LaTeX sections, if scripts/ingest_paper.py has been run.
    paper_files = 0
    for row in db.list_paper_files():
        chunks = chunk_latex_file(row["path"], row["content"])
        all_chunks.extend(chunks)
        paper_files += 1

    with open(CHUNKS_PATH, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk)) + "\n")

    print(f"Indexed {code_chunks} chunks from {indexed_files} repo files ({skipped_files} skipped).")
    print(f"Indexed {len(all_chunks) - code_chunks} chunks from {paper_files} paper sections.")
    print(f"Total: {len(all_chunks)} chunks written to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
