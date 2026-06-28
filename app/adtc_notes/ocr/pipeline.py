"""adtc_notes/ocr/pipeline.py — Compose preprocessing + OCR into usable text.

Two entry points:
* ``ocr_image_to_text`` — quick path used by RAG ingestion (text only).
* ``digitize_image`` — produces a raw draft for PATH A, then handed to the LLM
  for clean formatting (see ``adtc_notes.pipeline``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import CONFIG, AppConfig, OCRConfig
from .preprocess import load_and_clean
from .text_ocr import ocr_text


def resolve_engine(cfg: AppConfig | None = None) -> str:
    """Decide which digitize engine to use: ``"vlm"`` or ``"tesseract"``.

    ``ocr.engine`` may be ``"vlm"``, ``"tesseract"``, or ``"auto"`` (use the VLM
    when its weights are present, otherwise fall back to Tesseract).
    """
    cfg = cfg or CONFIG
    choice = cfg.ocr.engine.lower()
    if choice == "vlm":
        return "vlm"
    if choice == "tesseract":
        return "tesseract"
    from . import vlm  # lazy: avoids importing llama_cpp paths unless needed

    return "vlm" if vlm.is_available(cfg.vision) else "tesseract"


def image_to_markdown(path: str | Path, cfg: AppConfig | None = None) -> str:
    """Transcribe an image to text/Markdown using the resolved engine.

    The VLM returns structured Markdown; Tesseract returns plain text. Used by RAG
    ingestion so image documents are indexed with the best available quality.
    """
    cfg = cfg or CONFIG
    if resolve_engine(cfg) == "vlm":
        from .vlm import get_vision

        return get_vision(cfg.vision).transcribe(path)
    return ocr_image_to_text(path, cfg.ocr)


@dataclass
class OCRResult:
    """Raw OCR output for one image.

    Attributes:
        text: Recognized body text.
        latex: Recognized formulas as LaTeX (empty unless formula OCR ran).
        source: Original image filename.
        warnings: Non-fatal issues (e.g. formula backend unavailable).
    """

    text: str
    source: str
    latex: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def ocr_image_to_text(path: str | Path, cfg: OCRConfig | None = None) -> str:
    """Preprocess and OCR an image, returning plain text (used by RAG ingest)."""
    cfg = cfg or CONFIG.ocr
    return ocr_text(load_and_clean(path), cfg)


def digitize_image(path: str | Path, cfg: OCRConfig | None = None) -> OCRResult:
    """Run the full OCR pass over one image and collect text + optional formulas.

    Formula OCR is attempted only when enabled and installed; failure degrades
    gracefully into a warning rather than aborting the whole digitization.

    Args:
        path: Image file path.
        cfg: OCR configuration.

    Returns:
        An ``OCRResult`` ready to be formatted by the LLM.
    """
    cfg = cfg or CONFIG.ocr
    image = load_and_clean(path)
    result = OCRResult(text=ocr_text(image, cfg), source=Path(path).name)

    if cfg.enable_formula_ocr:
        from . import formula

        if formula.is_available():
            try:
                # Whole-image formula pass; region detection is a future refinement.
                latex = formula.image_to_latex(image)
                if latex:
                    result.latex.append(latex)
            except Exception as exc:  # noqa: BLE001 - never let optional OCR crash the run
                result.warnings.append(f"formula OCR failed: {exc}")
        else:
            result.warnings.append("formula OCR enabled but pix2tex not installed")
    return result
