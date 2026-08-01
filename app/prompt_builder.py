"""
Assembles the message list sent to the model: a system prompt describing its
role, the retrieved repo chunks as grounding context, and the running chat
history.
"""

from app.retrieval import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a precise assistant answering questions about the GitHub repository "
    "ESHyperscale/HyperscaleES (a JAX codebase for evolutionary strategies at scale), "
    "using only the repo context and conversation history given to you — never general "
    "outside knowledge.\n\n"
    "Special case: if the user asks whether their machine/system can run this repo, or "
    "asks how to check their system against the repo's requirements, do NOT just repeat "
    "install steps. Instead give a short shell/Python snippet the user can run locally "
    "to inspect their own machine (e.g. Python version, GPU/CUDA availability, installed "
    "package versions), then separately state the specific requirements found in the "
    "repo context (e.g. required Python/CUDA version, base image, key dependencies) so "
    "they can compare the two.\n\n"
    "Special case: if the user asks whether a file is imported, used, or referenced "
    "elsewhere in the repo, and a 'File reference check' fact is present in the context "
    "below, state that fact directly and confidently — do not guess, and never say a "
    "file is imported by itself. If no such fact is present, say plainly that you "
    "can't confirm cross-file usage from the given context rather than inferring it.\n\n"
    "If the question has multiple distinct parts (e.g. what a file does, AND where "
    "it's used, AND whether it's imported), answer each part as its own short "
    "sentence rather than blending them into one — clarity over brevity when a "
    "question genuinely has multiple parts.\n\n"
    "Special case: if asked to count or list functions/classes/methods in a file, and "
    "a 'Function/class inventory' fact is present in the context below, use that exact "
    "list and count verbatim — it was computed by parsing the real file, so it is more "
    "reliable than counting chunks yourself. Do not add, remove, or recount items "
    "against it, and do not count a variable merely assigned from a call (e.g. "
    "`x = jax.jit(...)`) as a function."
)

# Repeated right next to the question (not just in the system message) because small
# local models weight instructions near the end of the prompt much more heavily than
# ones buried earlier, especially once a large context block sits in between.
ANSWER_FORMAT_INSTRUCTION = (
    "Answer the question above using ONLY the repo context. Reply in exactly this "
    "shape and nothing else:\n"
    "Source: <file path(s) the answer comes from, or \"none found\">\n"
    "<a short, direct answer — 1-4 sentences, or a short list. Quote exact commands/"
    "code verbatim if the question asks how to do something, but ALWAYS wrap them in "
    "standard markdown triple-backtick code fences (```) — never copy a source file's "
    "own markup syntax verbatim (e.g. strip Org-mode's #+BEGIN_SRC/#+END_SRC markers, "
    "don't reproduce them). No restating the question, no generic advice, no closing "
    "summary.>"
)


def build_messages(
    user_message: str,
    history: list[dict],
    retrieved_chunks: list[RetrievedChunk],
    file_reference_note: str | None = None,
    file_structure_note: str | None = None,
) -> list[dict]:
    if retrieved_chunks:
        context_block = "\n\n".join(
            f"--- {chunk.path} (lines {chunk.start_line}-{chunk.end_line}) ---\n{chunk.text}"
            for chunk in retrieved_chunks
        )
    else:
        context_block = "(no relevant repo content found for this question)"

    if file_reference_note:
        context_block = f"{context_block}\n\nFile reference check: {file_reference_note}"
    if file_structure_note:
        context_block = f"{context_block}\n\nFunction/class inventory: {file_structure_note}"

    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nRelevant repo context:\n{context_block}"}
    ]
    messages.extend(history)
    messages.append(
        {"role": "user", "content": f"{user_message}\n\n{ANSWER_FORMAT_INSTRUCTION}"}
    )
    return messages
