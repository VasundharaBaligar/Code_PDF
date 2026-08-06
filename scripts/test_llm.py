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
from app.retrieval import (
    find_cross_references,
    find_mentioned_path,
    get_structure_note,
    is_enumeration_query,
    load_corpus,
    search,
)


async def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/test_llm.py "<question>"')
        sys.exit(1)

    question = sys.argv[1]

    load_corpus()
    chunks = search(question, k=8)
    print(f"Retrieved {len(chunks)} relevant chunks:")
    for chunk in chunks:
        print(f"  - {chunk.path}:{chunk.start_line}-{chunk.end_line} (score={chunk.score:.2f})")

    file_reference_note = None
    mentioned_path = find_mentioned_path(question)
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
        print(f"File reference check: {file_reference_note}")

    file_structure_note = None
    if mentioned_path and is_enumeration_query(question):
        file_structure_note = get_structure_note(mentioned_path)
        print(f"Function/class inventory: {file_structure_note}")
    print()

    messages = build_messages(
        question,
        history=[],
        retrieved_chunks=chunks,
        file_reference_note=file_reference_note,
        file_structure_note=file_structure_note,
    )

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
