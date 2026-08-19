"""
Thin async streaming client for Groq's free-tier, OpenAI-compatible chat
completions API. Tried first when a key is configured — fast, free
(no credit card), no local compute.

Groq retires models without notice: it dropped every Llama chat model
partway through this project, and because the router falls back silently,
every answer quietly came from the weaker local model for days before
anyone noticed. So the model isn't pinned — a preference list is resolved
against Groq's live catalogue, and a model that disappears mid-flight
triggers one re-resolve rather than an outage.
"""

import json
import logging
import re
from collections.abc import AsyncIterator

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Groq's catalogue mixes chat models with speech, safety and TTS models that
# would 400 on a chat request. Used only for the last-resort scan, when none
# of the preferred models exist any more.
_NON_CHAT_RE = re.compile(r"whisper|prompt-guard|orpheus|tts|embed|safeguard", re.IGNORECASE)

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)

_resolved_model: str | None = None


def _size_rank(model_id: str) -> float:
    """Parameter count parsed from the model id, for ranking unknown models."""
    match = _SIZE_RE.search(model_id)
    return float(match.group(1)) if match else 0.0


class GroqError(Exception):
    """Raised when Groq can't be reached, rejects the key, or rate-limits us."""


def _is_model_missing(detail: str) -> bool:
    return "model_not_found" in detail or "does not exist" in detail


async def _list_models(client: httpx.AsyncClient) -> list[str]:
    resp = await client.get(
        f"{settings.groq_base_url}/models",
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
    )
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]


async def resolve_model(client: httpx.AsyncClient, force: bool = False) -> str:
    """
    First preferred model that Groq actually serves, cached for the process.

    Falls back to scanning the live catalogue so that even if every preferred
    model is retired at once, the app keeps using Groq instead of silently
    degrading to the local model.
    """
    global _resolved_model
    if _resolved_model and not force:
        return _resolved_model

    preferences = settings.groq_model_preferences
    try:
        available = await _list_models(client)
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        # Can't check the catalogue — optimistically try the top preference.
        logger.warning("Could not list Groq models (%s); trying %s", exc, preferences[0])
        _resolved_model = preferences[0]
        return _resolved_model

    for preferred in preferences:
        if preferred in available:
            if preferred != preferences[0]:
                logger.warning(
                    "Preferred Groq model %s is unavailable; using %s instead",
                    preferences[0],
                    preferred,
                )
            _resolved_model = preferred
            return preferred

    # Rank the leftovers by advertised parameter count, so a last-resort pick
    # doesn't land on a small special-purpose model just because it sorts
    # first alphabetically (allam-2-7b beating openai/gpt-oss-120b).
    survivors = sorted(
        (m for m in available if not _NON_CHAT_RE.search(m)),
        key=_size_rank,
        reverse=True,
    )
    if survivors:
        logger.error(
            "None of the configured Groq models exist any more (%s). "
            "Falling back to %s from the live catalogue — update GROQ_MODELS.",
            ", ".join(preferences),
            survivors[0],
        )
        _resolved_model = survivors[0]
        return survivors[0]

    raise GroqError(f"No usable Groq chat model found. Catalogue: {available}")


async def _stream_once(client: httpx.AsyncClient, model: str, messages: list[dict]) -> AsyncIterator[str]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0.2,
    }
    async with client.stream(
        "POST",
        f"{settings.groq_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        json=payload,
    ) as response:
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
            # Reasoning models put chain-of-thought in a separate `reasoning`
            # field; reading only `content` keeps it out of the answer.
            content = delta.get("content")
            if content:
                yield content


async def stream_chat(messages: list[dict]) -> AsyncIterator[str]:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            model = await resolve_model(client)
            try:
                async for token in _stream_once(client, model, messages):
                    yield token
                return
            except GroqError as exc:
                if not _is_model_missing(str(exc)):
                    raise
                # Retired between resolution and use: re-resolve once and retry.
                logger.error("Groq model %s vanished mid-flight; re-resolving", model)
                new_model = await resolve_model(client, force=True)
                if new_model == model:
                    raise
                async for token in _stream_once(client, new_model, messages):
                    yield token
    except httpx.RequestError as exc:
        raise GroqError(f"Could not reach Groq: {exc}") from exc
