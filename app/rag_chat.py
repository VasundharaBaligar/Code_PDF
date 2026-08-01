"""
Combines retrieval, prompt building, and the provider router into the
actual event stream served by POST /api/chat.
"""

from collections.abc import AsyncIterator

from app.prompt_builder import build_messages
from app.providers.router import stream_answer
from app.retrieval import find_cross_references, find_mentioned_path, search


async def stream_chat_response(message: str, history: list[dict]) -> AsyncIterator[dict]:
    chunks = search(message, k=6)

    seen_paths: set[str] = set()
    for chunk in chunks:
        if chunk.path in seen_paths:
            continue
        seen_paths.add(chunk.path)
        yield {"type": "citation", "path": chunk.path, "score": chunk.score}

    file_reference_note = None
    mentioned_path = find_mentioned_path(message)
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
