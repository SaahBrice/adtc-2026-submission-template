"""docaware/llm/prompts.py — Prompt templates (pure functions, no I/O).

Kept dependency-free so they are trivially unit-testable: grounded RAG answers
(with citations + conversation memory), history-aware query rewriting, and the
running conversation summary for long chats.
"""

from __future__ import annotations

from typing import Sequence

# --- Document Q&A / RAG (PATH B) ---------------------------------------------

_QA_SYSTEM = (
    "You are an enterprise knowledge assistant. Answer using ONLY the provided "
    "context excerpts; if the answer isn't there, say so plainly. Earlier turns of "
    "this conversation are included for follow-up context. Be concise and direct. "
    "Format answers in Markdown: use tables for tabular data, bullet lists where "
    "they help, and LaTeX for math ($...$ inline, $$...$$ for displayed equations). "
    "Cite every claim with the bracketed source tag shown before each excerpt, e.g. "
    "[report.pdf p.3]. Reference the document name and page when it helps the reader."
)

# Excerpts are labelled with their real source so the model cites file + page.
LabeledContext = tuple[str, str]  # (citation_label, text), e.g. ("report.pdf p.3", "...")


def qa_messages(
    question: str,
    contexts: Sequence[LabeledContext],
    history: Sequence[dict[str, str]] | None = None,
    summary: str | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for grounded Q&A with citations and conversation memory.

    Args:
        question: The user's natural-language question.
        contexts: Retrieved ``(citation_label, text)`` excerpts, most relevant first.
        history: Recent ``{"role","content"}`` turns (raw) for follow-ups.
        summary: Running summary of older turns (for long chats); folded into system.

    Returns:
        OpenAI-style message list ready for ``LLMClient.chat``.
    """
    system = _QA_SYSTEM
    if summary:
        system += f"\n\nSummary of earlier conversation:\n{summary}"
    numbered = "\n\n".join(f"[{label}] {text}" for label, text in contexts)
    user = f"Context excerpts:\n\n{numbered}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": system},
        *(history or []),
        {"role": "user", "content": user},
    ]


_CONDENSE_SYSTEM = (
    "Given the conversation so far and a follow-up message, rewrite the follow-up "
    "as a standalone question that can be understood without the conversation. "
    "Resolve pronouns and references. If it is already standalone, return it "
    "unchanged. Output ONLY the rewritten question, nothing else."
)


def condense_question_messages(
    question: str, history: Sequence[dict[str, str]], summary: str | None = None
) -> list[dict[str, str]]:
    """Build messages to rewrite a follow-up into a standalone retrieval query.

    This is the history-aware retrieval step: the rewritten question is what we
    embed and search with, so follow-ups like "what about Q4?" retrieve correctly.
    """
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    if summary:
        convo = f"[Earlier summary] {summary}\n{convo}"
    user = f"Conversation:\n{convo}\n\nFollow-up: {question}\n\nStandalone question:"
    return [
        {"role": "system", "content": _CONDENSE_SYSTEM},
        {"role": "user", "content": user},
    ]


# --- Running conversation summary (long-chat memory) -------------------------

_RUNNING_SUMMARY_SYSTEM = (
    "You maintain a running summary of a long conversation so the assistant never "
    "forgets earlier turns. Update the existing summary with the new messages. "
    "Preserve the user's goals, decisions, constraints, facts, and any documents or "
    "entities discussed; drop pleasantries. Keep it under 200 words. Output ONLY the "
    "updated summary."
)


def update_summary_messages(
    previous_summary: str, new_messages: Sequence[dict[str, str]]
) -> list[dict[str, str]]:
    """Build messages to fold older turns into a compact running summary."""
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in new_messages)
    prev = previous_summary or "(none yet)"
    user = f"Existing summary:\n{prev}\n\nNew messages to fold in:\n{convo}\n\nUpdated summary:"
    return [
        {"role": "system", "content": _RUNNING_SUMMARY_SYSTEM},
        {"role": "user", "content": user},
    ]
