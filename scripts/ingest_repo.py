"""
One-time ingestion: pull the HyperscaleES repo's file tree and contents from
GitHub and cache everything into a local SQLite database. After this script
runs, nothing else in the app talks to GitHub again.

Usage:
    python scripts/ingest_repo.py
"""

import sqlite3
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR, DB_PATH, settings

API_BASE = "https://api.github.com"

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf",
    ".whl", ".pyc", ".pkl", ".npz", ".npy",
    ".so", ".dylib", ".dll", ".exe", ".zip", ".tar", ".gz",
}


def auth_headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers


def fetch_repo_meta(client: httpx.Client) -> dict:
    """One API call: get the default branch name for the repo."""
    url = f"{API_BASE}/repos/{settings.github_repo_owner}/{settings.github_repo_name}"
    resp = client.get(url, headers=auth_headers())
    resp.raise_for_status()
    data = resp.json()
    return {"default_branch": data["default_branch"], "full_name": data["full_name"]}


def fetch_tree(client: httpx.Client, branch: str) -> list[dict]:
    """One API call: get the full recursive file tree for the branch."""
    url = (
        f"{API_BASE}/repos/{settings.github_repo_owner}/{settings.github_repo_name}"
        f"/git/trees/{branch}?recursive=1"
    )
    resp = client.get(url, headers=auth_headers())
    resp.raise_for_status()
    data = resp.json()
    if data.get("truncated"):
        print("WARNING: GitHub truncated the tree response; repo may be larger than expected.")
    return [entry for entry in data["tree"] if entry["type"] == "blob"]


def is_large_binary(path: str, size: int) -> bool:
    ext = Path(path).suffix.lower()
    return size > settings.large_file_threshold or ext in BINARY_EXTENSIONS


def fetch_file_content(client: httpx.Client, branch: str, path: str) -> str:
    """Raw content fetch — does NOT count against the api.github.com rate limit."""
    url = (
        f"https://raw.githubusercontent.com/"
        f"{settings.github_repo_owner}/{settings.github_repo_name}/{branch}/{path}"
    )
    resp = client.get(url)
    resp.raise_for_status()
    return resp.text


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            size INTEGER,
            sha TEXT,
            content TEXT,
            is_binary INTEGER NOT NULL DEFAULT 0,
            is_truncated INTEGER NOT NULL DEFAULT 0,
            raw_url TEXT
        );
        CREATE TABLE IF NOT EXISTS repo_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=30.0) as client:
        meta = fetch_repo_meta(client)
        branch = meta["default_branch"]
        print(f"Repo: {meta['full_name']}  branch: {branch}")

        entries = fetch_tree(client, branch)
        print(f"Found {len(entries)} files in tree.")

        conn = sqlite3.connect(DB_PATH)
        init_db(conn)
        conn.execute("DELETE FROM files")

        fetched, skipped = 0, 0
        for entry in entries:
            path = entry["path"]
            size = entry.get("size", 0)
            sha = entry.get("sha", "")
            raw_url = (
                f"https://raw.githubusercontent.com/"
                f"{settings.github_repo_owner}/{settings.github_repo_name}/{branch}/{path}"
            )

            if is_large_binary(path, size):
                conn.execute(
                    "INSERT INTO files (path, size, sha, content, is_binary, is_truncated, raw_url) "
                    "VALUES (?, ?, ?, NULL, 1, 1, ?)",
                    (path, size, sha, raw_url),
                )
                skipped += 1
                print(f"  [skip content] {path} ({size:,} bytes)")
                continue

            try:
                content = fetch_file_content(client, branch, path)
            except httpx.HTTPStatusError as exc:
                print(f"  [error fetching] {path}: {exc}")
                continue

            conn.execute(
                "INSERT INTO files (path, size, sha, content, is_binary, is_truncated, raw_url) "
                "VALUES (?, ?, ?, ?, 0, 0, ?)",
                (path, size, sha, content, raw_url),
            )
            fetched += 1

        conn.execute(
            "INSERT OR REPLACE INTO repo_meta (key, value) VALUES (?, ?)",
            ("default_branch", branch),
        )
        conn.execute(
            "INSERT OR REPLACE INTO repo_meta (key, value) VALUES (?, ?)",
            ("full_name", meta["full_name"]),
        )
        conn.commit()
        conn.close()

        print(f"\nDone. {fetched} files cached with content, {skipped} large/binary files metadata-only.")
        print(f"Database written to: {DB_PATH}")


if __name__ == "__main__":
    main()
