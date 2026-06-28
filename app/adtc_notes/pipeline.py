"""adtc_notes/pipeline.py — End-to-end orchestration for PATH A (digitize).

Image → OCR draft → LLM clean-up/formatting → downloadable document. This is the
"snap a photo, get a tidy file" flow. PATH B (Q&A over documents) lives in
``adtc_notes.rag.retriever``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import CONFIG, AppConfig
from .llm import get_llm, prompts
from .ocr.pipeline import digitize_image
from .render import render_document


@dataclass
class DigitizeResult:
    """Outcome of digitizing one image into a formatted document.

    Attributes:
        markdown: The clean, LLM-formatted Markdown.
        output_path: Path to the rendered downloadable file.
        warnings: Non-fatal issues from OCR (e.g. formula backend missing).
    """

    markdown: str
    output_path: Path
    warnings: list[str]


def _assemble_raw_draft(text: str, latex: list[str]) -> str:
    """Combine OCR body text and any recognized formulas into one LLM input."""
    parts = [text.strip()]
    for i, formula in enumerate(latex, start=1):
        # Surface formulas explicitly so the model places them as block LaTeX.
        parts.append(f"[FORMULA {i}]: $$ {formula} $$")
    return "\n\n".join(p for p in parts if p)


def digitize_to_document(
    image_path: str | Path,
    *,
    fmt: str = "md",
    out_name: str | None = None,
    cfg: AppConfig | None = None,
) -> DigitizeResult:
    """Digitize an image into a clean, downloadable document.

    Args:
        image_path: Path to the photographed/scanned page.
        fmt: Output format — one of ``md``, ``docx``, ``pdf``.
        out_name: Base filename (without extension); defaults to the image stem.
        cfg: Optional config override.

    Returns:
        A ``DigitizeResult`` with the Markdown and the written file path.
    """
    cfg = cfg or CONFIG
    cfg.ensure_dirs()

    ocr = digitize_image(image_path, cfg.ocr)
    raw_draft = _assemble_raw_draft(ocr.text, ocr.latex)

    # LLM turns the noisy OCR draft into clean, structured Markdown.
    llm = get_llm(cfg)
    markdown = llm.chat(prompts.format_document_messages(raw_draft))

    from .config import OUTPUT_DIR

    base = out_name or Path(image_path).stem
    output_path = render_document(markdown, OUTPUT_DIR / base, fmt=fmt)
    return DigitizeResult(markdown=markdown, output_path=output_path, warnings=ocr.warnings)
