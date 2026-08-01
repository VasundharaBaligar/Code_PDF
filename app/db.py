import sqlite3
from contextlib import contextmanager

from app.config import DB_PATH


@contextmanager
def get_connection():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def get_repo_meta() -> dict[str, str]:
    with get_connection() as conn:
        rows = conn.execute("SELECT key, value FROM repo_meta").fetchall()
        return {row["key"]: row["value"] for row in rows}


def list_files() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT path, size, is_binary, is_truncated FROM files ORDER BY path"
        ).fetchall()


def get_file(path: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            "SELECT path, size, sha, content, is_binary, is_truncated, raw_url "
            "FROM files WHERE path = ?",
            (path,),
        ).fetchone()
