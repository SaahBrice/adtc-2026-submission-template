"""adtc_notes/ocr/text_ocr.py — Printed/handwritten text OCR via Tesseract.

Tesseract is the lightest reliable CPU OCR engine — ideal for the target laptop.
The Python wrapper (pytesseract) shells out to the native ``tesseract`` binary,
which must be installed separately (apt/winget); see app/README.md.
"""

from __future__ import annotations

from pathlib import Path

from ..config import CONFIG, OCRConfig
from ..errors import BackendNotInstalledError


def ocr_text(image, cfg: OCRConfig | None = None) -> str:
    """Run Tesseract on a preprocessed ``PIL.Image`` and return raw text.

    Args:
        image: A ``PIL.Image`` (typically from ``preprocess.load_and_clean``).
        cfg: OCR configuration (language, etc.).

    Returns:
        Extracted text, whitespace-trimmed.

    Raises:
        BackendNotInstalledError: If pytesseract or the tesseract binary is absent.
    """
    cfg = cfg or CONFIG.ocr
    try:
        import pytesseract  # type: ignore
    except ImportError as exc:
        raise BackendNotInstalledError(
            "pytesseract not installed: pip install pytesseract\n"
            "Also install the tesseract binary (see app/README.md)."
        ) from exc
    try:
        return pytesseract.image_to_string(image, lang=cfg.tesseract_lang).strip()
    except pytesseract.TesseractNotFoundError as exc:  # type: ignore[attr-defined]
        raise BackendNotInstalledError(
            "The 'tesseract' binary was not found on PATH.\n"
            "  Ubuntu: sudo apt install tesseract-ocr\n"
            "  Windows: winget install UB-Mannheim.TesseractOCR"
        ) from exc


def ocr_file(path: str | Path, cfg: OCRConfig | None = None) -> str:
    """Convenience: preprocess an image file and OCR it to text."""
    from .preprocess import load_and_clean

    return ocr_text(load_and_clean(path), cfg)
