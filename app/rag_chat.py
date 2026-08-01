"""
Combines retrieval, prompt building, and the provider router into the
actual event stream served by POST /api/chat.
"""

from collections.abc import AsyncIterator

from app.prompt_builder import build_messages
from app.providers.router import stream_answer
from app.retrieval import find_cross_references, find_mentioned_path, search


def _build_retrieval_query(message: str, history: list[dict]) -> str:
    """
    Follow-up questions often use pronouns ("this file", "it") instead of
    naming their subject again. BM25 and the filename-mention check only see
    the literal query text, so without this, retrieval loses the thread the
    moment a question refers back rather than re-naming what it's about.
    Folding in the last couple of turns keeps retrieval "in the room" for
    the conversation -- the model still gets the full structured history
    separately for actually answering.
    """
    recent = history[-4:]
    recent_text = " ".join(m["content"] for m in recent)
    return f"{recent_text} {message}".strip()


async def stream_chat_response(message: str, history: list[dict]) -> AsyncIterator[dict]:
    retrieval_query = _build_retrieval_query(message, history)
    chunks = search(retrieval_query, k=6)

    seen_paths: set[str] = set()
    for chunk in chunks:
        if chunk.path in seen_paths:
            continue
        seen_paths.add(chunk.path)
        yield {"type": "citation", "path": chunk.path, "score": chunk.score}

    file_reference_note = None
    mentioned_path = find_mentioned_path(retrieval_query)
    if mentioned_path:
        cross_refs = find_cross_references(mentioned_path)
        if cross_refs:
            file_reference_note = f"{mentioned_path} is referenced in: {', '.join(cross_refs)}."
        else:
            file_reference_note = (
                f"{mentioned_path} is not imported or referenced by any other file in "
                "this repo — it appears to be a standalone/entry-point script, not a "
                "module meant to be imported."
            )

    messages = build_messages(message, history, chunks, file_reference_note)

    async for kind, payload in stream_answer(messages):
        if kind == "meta":
            yield {"type": "meta", "provider": payload}
        elif kind == "token":
            yield {"type": "token", "text": payload}
        elif kind == "error":
            yield {"type": "error", "message": payload}
            return

    yield {"type": "done"}
