"""
One-time ingestion: pull the paper's LaTeX source from arXiv and cache the
section files into the same local SQLite database used for the repo.

We use the LaTeX source rather than the PDF deliberately: equations arrive as
real LaTeX (no OCR guesswork), and the authors' own \\label{} anchors give us
precise, stable, human-meaningful citation targets.

Usage:
    python scripts/ingest_paper.py
"""

import io
import re
import sqlite3
import sys
import tarfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR, DB_PATH, settings

EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"

# `%` starts a LaTeX comment unless escaped as `\%`. Commented-out lines are
# draft/removed content -- indexing them would let the assistant cite text the
# authors deliberately deleted, so they must be stripped.
COMMENT_RE = re.compile(r"(?<!\\)%.*")

INPUT_RE = re.compile(r"^\s*\\input\{([^}]+)\}", re.MULTILINE)
TITLE_RE = re.compile(r"\\title\{(.+?)\}", re.DOTALL)


def download_source(arxiv_id: str) -> tarfile.TarFile:
    url = EPRINT_URL.format(arxiv_id=arxiv_id)
    resp = httpx.get(url, follow_redirects=True, timeout=60.0)
    resp.raise_for_status()
    return tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz")


def read_member(tar: tarfile.TarFile, name: str) -> str | None:
    """
    Read a .tex member, tolerating the optional .tex extension in \\input and
    case mismatches. The latter matters: main.tex says
    \\input{Sections/04_low_rank_ES} but the file is 04_low_rank_es.tex --
    fine on the authors' case-insensitive filesystem, but the tar archive is
    case-sensitive, so an exact-match-only lookup silently drops the entire
    EGGROLL section.
    """
    for candidate in (name, f"{name}.tex"):
        try:
            member = tar.extractfile(candidate)
        except KeyError:
            member = None
        if member is not None:
            return member.read().decode("utf-8", errors="replace")

    wanted = f"{name}.tex".lower()
    for member_name in tar.getnames():
        if member_name.lower() == wanted:
            member = tar.extractfile(member_name)
            if member is not None:
                return member.read().decode("utf-8", errors="replace")
    return None


def strip_comments(text: str) -> str:
    """Remove LaTeX comments while preserving line count (line numbers stay valid)."""
    return "\n".join(COMMENT_RE.sub("", line).rstrip() for line in text.splitlines())


def resolve_inputs(tar: tarfile.TarFile, root_name: str) -> list[str]:
    """
    Return section file paths in true document order, following nested \\input
    (appendix.tex pulls in the hyperparameter table files, for example).
    """
    ordered: list[str] = []
    seen: set[str] = set()

    def walk(name: str) -> None:
        content = read_member(tar, name)
        if content is None:
            return
        for match in INPUT_RE.finditer(strip_comments(content)):
            child = match.group(1)
            if child in seen:
                continue
            seen.add(child)
            ordered.append(child)
            walk(child)

    walk(root_name)
    return ordered


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS paper_files (
            path TEXT PRIMARY KEY,
            order_index INTEGER NOT NULL,
            content TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS paper_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading arXiv:{settings.arxiv_id} LaTeX source ...")
    tar = download_source(settings.arxiv_id)

    main_tex = read_member(tar, "main.tex")
    if main_tex is None:
        raise SystemExit("main.tex not found in the arXiv source archive.")

    section_paths = resolve_inputs(tar, "main.tex")
    print(f"Found {len(section_paths)} section files in document order.")

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    conn.execute("DELETE FROM paper_files")

    total_chars = 0
    for order_index, path in enumerate(section_paths):
        raw = read_member(tar, path)
        if raw is None:
            print(f"  [missing] {path}")
            continue
        content = strip_comments(raw)
        total_chars += len(content)
        conn.execute(
            "INSERT INTO paper_files (path, order_index, content) VALUES (?, ?, ?)",
            (f"{path}.tex" if not path.endswith(".tex") else path, order_index, content),
        )
        print(f"  [{order_index:2d}] {path}  ({len(content):,} chars)")

    title_match = TITLE_RE.search(strip_comments(main_tex))
    title = " ".join(title_match.group(1).split()) if title_match else "(unknown title)"

    for key, value in {
        "arxiv_id": settings.arxiv_id,
        "title": title,
        "abs_url": f"https://arxiv.org/abs/{settings.arxiv_id}",
    }.items():
        conn.execute(
            "INSERT OR REPLACE INTO paper_meta (key, value) VALUES (?, ?)", (key, value)
        )

    conn.commit()
    conn.close()
    tar.close()

    print(f"\nTitle: {title}")
    print(f"Done. {len(section_paths)} sections, {total_chars:,} chars cached.")
    print(f"Database: {DB_PATH}")


if __name__ == "__main__":
    main()
