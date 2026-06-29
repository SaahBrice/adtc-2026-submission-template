"""docaware/pipeline.py — End-to-end orchestration for PATH A (digitize).

Image or PDF → clean Markdown → downloadable document.
- Images: DeepSeek-OCR (via llama-mtmd-cli).
- PDFs: per page, use the embedded text layer when present (born-digital → instant)
  and OCR only the pages that are scans/images. Pages are combined into one doc.

PATH B (Q&A over documents) lives in ``docaware.rag.session``.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .config import CONFIG, OUTPUT_DIR, AppConfig
from .ocr import get_vision
from .render import render_document

# Pages with at least this many extractable characters are treated as born-digital
# (use the text layer directly); fewer means a scan/photo that needs OCR.
_TEXT_LAYER_MIN_CHARS = 80


@dataclass
class DigitizeResult:
    """Outcome of digitizing one image/PDF into a formatted document."""

    markdown: str
    output_path: Path
    warnings: list[str] = field(default_factory=list)


def _digitize_pdf(pdf_path: Path, cfg: AppConfig, warnings: list[str]) -> str:
    """Transcribe a PDF page-by-page (text layer where possible, OCR for scans)."""
    import fitz  # PyMuPDF (lazy import)

    doc = fitz.open(str(pdf_path))
    vision = None
    parts: list[str] = []
    for i, page in enumerate(doc, start=1):
        text = (page.get_text("text") or "").strip()
        if len(text) >= _TEXT_LAYER_MIN_CHARS:
            parts.append(f"## Page {i}\n\n{text}")  # born-digital: instant, exact
            continue
        # Scanned/image page → render to PNG at ~200 DPI and OCR it.
        if vision is None:
            vision = get_vision(cfg.vision)
        zoom = 200 / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pix.save(tmp.name)
            img = tmp.name
        try:
            parts.append(f"## Page {i}\n\n{vision.transcribe(img)}")
        finally:
            Path(img).unlink(missing_ok=True)
    doc.close()
    if not parts:
        warnings.append("PDF had no extractable text and no pages to OCR.")
    return "\n\n---\n\n".join(parts)


def digitize_to_document(
    source_path: str | Path,
    *,
    fmt: str = "md",
    out_name: str | None = None,
    cfg: AppConfig | None = None,
) -> DigitizeResult:
    """Digitize an image or PDF into a clean, downloadable document.

    Args:
        source_path: Path to an image (png/jpg/...) or a PDF.
        fmt: Output format — one of ``md``, ``docx``, ``pdf``.
        out_name: Base filename (without extension); defaults to the source stem.
        cfg: Optional config override.

    Returns:
        A ``DigitizeResult`` with the Markdown and the written file path.
    """
    cfg = cfg or CONFIG
    cfg.ensure_dirs()
    source_path = Path(source_path)
    warnings: list[str] = []

    if source_path.suffix.lower() == ".pdf":
        markdown = _digitize_pdf(source_path, cfg, warnings)
    else:
        markdown = get_vision(cfg.vision).transcribe(source_path)

    base = out_name or source_path.stem
    output_path = render_document(markdown, OUTPUT_DIR / base, fmt=fmt)
    return DigitizeResult(markdown=markdown, output_path=output_path, warnings=warnings)
