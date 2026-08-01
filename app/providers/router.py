"""
Chooses which LLM provider answers a request. Tries Groq first if an API
key is configured; Ollama is always available as an automatic, zero-setup
fallback so the app keeps working with no external account at all.
"""

from collections.abc import AsyncIterator

from app.config import settings
from app.providers import groq_provider, ollama_provider
from app.providers.groq_provider import GroqError
from app.providers.ollama_provider import OllamaError


async def stream_answer(messages: list[dict]) -> AsyncIterator[tuple[str, str]]:
    """
    Yields ("meta", provider_name) once, then ("token", text) repeatedly,
    and optionally a final ("error", message) if something went wrong.

    If Groq's very first token fails (bad/missing key, rate limit, network),
    nothing has reached the caller yet, so we fall back to Ollama silently.
    A failure *after* the first Groq token is surfaced as an error instead,
    since a partial answer has already been shown under the Groq label.
    """
    if settings.groq_api_key:
        groq_stream = groq_provider.stream_chat(messages)
        try:
            first_token = await groq_stream.__anext__()
        except StopAsyncIteration:
            first_token = None
        except GroqError:
            async for event in _stream_ollama(messages):
                yield event
            return

        yield ("meta", "groq")
        if first_token is not None:
            yield ("token", first_token)
        try:
            async for text in groq_stream:
                yield ("token", text)
        except GroqError as exc:
            yield ("error", str(exc))
        return

    async for event in _stream_ollama(messages):
        yield event


async def _stream_ollama(messages: list[dict]) -> AsyncIterator[tuple[str, str]]:
    yield ("meta", "ollama")
    try:
        async for text in ollama_provider.stream_chat(messages):
            yield ("token", text)
    except OllamaError as exc:
        if settings.groq_api_key:
            # Ollama is only a fallback here (Groq is primary), so its Mac-specific
            # troubleshooting message would be meaningless to a hosted-deployment
            # visitor who has no Ollama at all.
            yield (
                "error",
                "The assistant is temporarily unavailable (rate-limited or offline). "
                "Please try again in a minute.",
            )
        else:
            yield ("error", str(exc))
