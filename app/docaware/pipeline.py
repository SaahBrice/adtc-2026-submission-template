"""docaware/pipeline.py — End-to-end orchestration for PATH A (digitize).

Image → DeepSeek-OCR → clean Markdown → downloadable document. This is the
"snap a photo, get a tidy file" flow. PATH B (Q&A over documents) lives in
``docaware.rag.session``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import CONFIG, OUTPUT_DIR, AppConfig
from .ocr import get_vision
from .render import render_document


@dataclass
class DigitizeResult:
    """Outcome of digitizing one image into a formatted document.

    Attributes:
        markdown: The clean Markdown produced by the OCR engine.
        output_path: Path to the rendered downloadable file.
        warnings: Non-fatal issues (reserved; empty in the current pipeline).
    """

    markdown: str
    output_path: Path
    warnings: list[str] = field(default_factory=list)


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

    # DeepSeek-OCR reads the page straight into clean Markdown (text, tables, LaTeX).
    markdown = get_vision(cfg.vision).transcribe(image_path)

    base = out_name or Path(image_path).stem
    output_path = render_document(markdown, OUTPUT_DIR / base, fmt=fmt)
    return DigitizeResult(markdown=markdown, output_path=output_path)
