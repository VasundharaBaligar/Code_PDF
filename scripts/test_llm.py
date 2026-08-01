"""
CLI smoke test: retrieval + prompt building + provider routing (Groq first
if configured, Ollama fallback otherwise), end to end, no web server needed.

Usage:
    python scripts/test_llm.py "explain what eggroll.py does"
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.prompt_builder import build_messages
from app.providers.router import stream_answer
from app.retrieval import load_corpus, search


async def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/test_llm.py "<question>"')
        sys.exit(1)

    question = sys.argv[1]

    load_corpus()
    chunks = search(question, k=6)
    print(f"Retrieved {len(chunks)} relevant chunks:")
    for chunk in chunks:
        print(f"  - {chunk.path}:{chunk.start_line}-{chunk.end_line} (score={chunk.score:.2f})")
    print()

    messages = build_messages(question, history=[], retrieved_chunks=chunks)

    print("--- model response ---")
    async for kind, payload in stream_answer(messages):
        if kind == "meta":
            print(f"[provider: {payload}]")
        elif kind == "token":
            print(payload, end="", flush=True)
        elif kind == "error":
            print(f"\n[error] {payload}")
            sys.exit(1)
    print("\n----------------------")


if __name__ == "__main__":
    asyncio.run(main())
