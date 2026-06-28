"""docaware/llm/prompts.py — Prompt templates (pure functions, no I/O).

Kept dependency-free so they are trivially unit-testable. Two task families:
formatting OCR drafts into clean documents, and answering questions over
retrieved document context (RAG) with grounded, cited answers.
"""

from __future__ import annotations

from typing import Sequence

# --- Document formatting (PATH A) -------------------------------------------

_FORMAT_SYSTEM = (
    "You are a meticulous corporate document editor. You receive raw OCR text "
    "extracted from a photographed or scanned document. Your job is to reconstruct "
    "a clean, well-structured version in GitHub-flavored Markdown.\n"
    "Rules:\n"
    "- Fix obvious OCR errors and spacing, but never invent facts or add content.\n"
    "- Use headings, lists, and tables where the structure implies them.\n"
    "- Preserve mathematical formulas as LaTeX between $...$ (inline) or $$...$$ (block).\n"
    "- Keep figure placeholders exactly as given (e.g. [[FIGURE: fig_1.png]]).\n"
    "- Output only the Markdown document, no commentary."
)


def format_document_messages(raw_ocr_text: str) -> list[dict[str, str]]:
    """Build chat messages that turn a raw OCR draft into clean Markdown."""
    return [
        {"role": "system", "content": _FORMAT_SYSTEM},
        {"role": "user", "content": f"Raw OCR text:\n\n{raw_ocr_text}"},
    ]


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
) -> list[dict[str, str]]:
    """Build chat messages for grounded Q&A with citations and conversation memory.

    Args:
        question: The user's natural-language question.
        contexts: Retrieved ``(citation_label, text)`` excerpts, most relevant first.
        history: Prior ``{"role","content"}`` turns (user/assistant) for follow-ups.

    Returns:
        OpenAI-style message list ready for ``LLMClient.chat``.
    """
    numbered = "\n\n".join(f"[{label}] {text}" for label, text in contexts)
    user = f"Context excerpts:\n\n{numbered}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": _QA_SYSTEM},
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
    question: str, history: Sequence[dict[str, str]]
) -> list[dict[str, str]]:
    """Build messages to rewrite a follow-up into a standalone retrieval query.

    This is the history-aware retrieval step: the rewritten question is what we
    embed and search with, so follow-ups like "what about Q4?" retrieve correctly.
    """
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    user = f"Conversation:\n{convo}\n\nFollow-up: {question}\n\nStandalone question:"
    return [
        {"role": "system", "content": _CONDENSE_SYSTEM},
        {"role": "user", "content": user},
    ]


# --- Summarization (Corporate/Enterprise core task) --------------------------

_SUMMARY_SYSTEM = (
    "You are a corporate analyst. Produce a faithful, well-structured summary of "
    "the document below. Use short paragraphs and bullet points for key findings, "
    "decisions, and action items. Do not add information that is not present."
)


def summary_messages(document_text: str) -> list[dict[str, str]]:
    """Build chat messages to summarize a document."""
    return [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {"role": "user", "content": document_text},
    ]
