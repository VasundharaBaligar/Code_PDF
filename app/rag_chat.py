"""
Combines retrieval, prompt building, and the provider router into the
actual event stream served by POST /api/chat.
"""

from collections.abc import AsyncIterator

from app.prompt_builder import build_messages
from app.providers.router import stream_answer
from app.retrieval import search


async def stream_chat_response(message: str, history: list[dict]) -> AsyncIterator[dict]:
    chunks = search(message, k=6)

    seen_paths: set[str] = set()
    for chunk in chunks:
        if chunk.path in seen_paths:
            continue
        seen_paths.add(chunk.path)
        yield {"type": "citation", "path": chunk.path, "score": chunk.score}

    messages = build_messages(message, history, chunks)

    async for kind, payload in stream_answer(messages):
        if kind == "meta":
            yield {"type": "meta", "provider": payload}
        elif kind == "token":
            yield {"type": "token", "text": payload}
        elif kind == "error":
            yield {"type": "error", "message": payload}
            return

    yield {"type": "done"}
