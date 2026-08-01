from pydantic import BaseModel


class TreeEntry(BaseModel):
    path: str
    size: int
    is_binary: bool
    is_truncated: bool


class TreeResponse(BaseModel):
    default_branch: str
    full_name: str
    files: list[TreeEntry]


class FileContentResponse(BaseModel):
    path: str
    size: int
    language: str | None
    is_binary: bool
    is_truncated: bool
    content: str | None
    highlighted_html: str | None
    raw_url: str | None


class ReadmeResponse(BaseModel):
    path: str
    html: str
