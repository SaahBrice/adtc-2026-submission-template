"""adtc_notes/llm/prompts.py — Prompt templates (pure functions, no I/O).

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
    "You are an enterprise knowledge assistant. Answer the user's question using "
    "ONLY the provided context excerpts. If the answer is not in the context, say "
    "you don't have enough information. Be concise and cite sources as [#] using "
    "the numbered excerpts."
)


def qa_messages(question: str, contexts: Sequence[str]) -> list[dict[str, str]]:
    """Build chat messages for grounded question answering over retrieved chunks.

    Args:
        question: The user's natural-language question.
        contexts: Retrieved text excerpts, most relevant first.

    Returns:
        OpenAI-style message list ready for ``LLMClient.chat``.
    """
    numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    user = f"Context excerpts:\n\n{numbered}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": _QA_SYSTEM},
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
