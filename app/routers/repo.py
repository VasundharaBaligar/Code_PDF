from fastapi import APIRouter, HTTPException, Query
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_for_filename, guess_lexer
from pygments.util import ClassNotFound

from app import db
from app.org_render import render_org_to_html
from app.schemas import FileContentResponse, ReadmeResponse, TreeEntry, TreeResponse

router = APIRouter(prefix="/api/repo", tags=["repo"])


def _detect_lexer(path: str, content: str):
    try:
        return get_lexer_for_filename(path, content)
    except ClassNotFound:
        pass
    try:
        return guess_lexer(content)
    except ClassNotFound:
        return TextLexer()


@router.get("/tree", response_model=TreeResponse)
def get_tree() -> TreeResponse:
    meta = db.get_repo_meta()
    files = [
        TreeEntry(
            path=row["path"],
            size=row["size"],
            is_binary=bool(row["is_binary"]),
            is_truncated=bool(row["is_truncated"]),
        )
        for row in db.list_files()
    ]
    return TreeResponse(
        default_branch=meta.get("default_branch", ""),
        full_name=meta.get("full_name", ""),
        files=files,
    )


@router.get("/file", response_model=FileContentResponse)
def get_file(path: str = Query(...)) -> FileContentResponse:
    row = db.get_file(path)
    if row is None:
        raise HTTPException(status_code=404, detail="file not found")

    if row["is_binary"] or row["content"] is None:
        return FileContentResponse(
            path=row["path"],
            size=row["size"],
            language=None,
            is_binary=True,
            is_truncated=bool(row["is_truncated"]),
            content=None,
            highlighted_html=None,
            raw_url=row["raw_url"],
        )

    lexer = _detect_lexer(row["path"], row["content"])
    formatter = HtmlFormatter(nowrap=False, cssclass="highlight")
    highlighted_html = highlight(row["content"], lexer, formatter)

    return FileContentResponse(
        path=row["path"],
        size=row["size"],
        language=lexer.name,
        is_binary=False,
        is_truncated=bool(row["is_truncated"]),
        content=row["content"],
        highlighted_html=highlighted_html,
        raw_url=row["raw_url"],
    )


@router.get("/readme", response_model=ReadmeResponse)
def get_readme() -> ReadmeResponse:
    row = db.get_file("README.org")
    if row is None or row["content"] is None:
        raise HTTPException(status_code=404, detail="README.org not found")
    return ReadmeResponse(path="README.org", html=render_org_to_html(row["content"]))
