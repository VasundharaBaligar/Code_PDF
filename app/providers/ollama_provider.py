"""
Thin async streaming client for a locally-running Ollama model. Used as the
always-available fallback when Groq isn't configured or fails.
"""

import json
from collections.abc import AsyncIterator

import httpx

from app.config import settings


class OllamaError(Exception):
    """Raised when Ollama can't be reached or returns an error."""


async def stream_chat(messages: list[dict]) -> AsyncIterator[str]:
    url = f"{settings.ollama_host}/api/chat"
    payload = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": True,
        # Lower temperature: for grounded Q&A over repo content we want literal,
        # instruction-following answers, not creative variety.
        "options": {"temperature": 0.2},
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise OllamaError(
                        f"Ollama returned {response.status_code}: {body.decode(errors='replace')}"
                    )
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise OllamaError(chunk["error"])
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done"):
                        break
    except httpx.ConnectError as exc:
        raise OllamaError(
            f"Could not connect to Ollama at {settings.ollama_host}. "
            "Is it running? (`brew services start ollama`)"
        ) from exc
