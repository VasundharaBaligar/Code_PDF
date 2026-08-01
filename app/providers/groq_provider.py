"""
Thin async streaming client for Groq's free-tier, OpenAI-compatible chat
completions API. Tried first when a key is configured — fast, free
(no credit card, 14,400 req/day on llama-3.1-8b-instant), no local compute.
"""

import json
from collections.abc import AsyncIterator

import httpx

from app.config import settings


class GroqError(Exception):
    """Raised when Groq can't be reached, rejects the key, or rate-limits us."""


async def stream_chat(messages: list[dict]) -> AsyncIterator[str]:
    url = f"{settings.groq_base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.groq_api_key}"}
    payload = {
        "model": settings.groq_model,
        "messages": messages,
        "stream": True,
        "temperature": 0.2,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise GroqError(
                        f"Groq returned {response.status_code}: {body.decode(errors='replace')}"
                    )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content")
                    if content:
                        yield content
    except httpx.RequestError as exc:
        raise GroqError(f"Could not reach Groq: {exc}") from exc
