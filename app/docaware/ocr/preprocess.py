"""docaware/ocr/preprocess.py — Lightweight image cleanup before OCR.

Uses Pillow only (no OpenCV) to stay light on the 8 GB target: grayscale,
autocontrast, and a simple adaptive-ish threshold improve Tesseract accuracy on
phone photos of documents without a heavy CV stack.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import BackendNotInstalledError


def _require_pil():
    try:
        from PIL import Image, ImageOps, ImageFilter  # type: ignore
    except ImportError as exc:
        raise BackendNotInstalledError("Pillow not installed: pip install pillow") from exc
    return Image, ImageOps, ImageFilter


def load_and_clean(path: str | Path, *, max_side: int = 2200):
    """Load an image and return a cleaned grayscale ``PIL.Image`` for OCR.

    Steps: EXIF-orient, downscale very large photos (caps memory/time), convert
    to grayscale, autocontrast, and light sharpening.

    Args:
        path: Image file path.
        max_side: Longest-edge cap in pixels (phone photos are often huge).

    Returns:
        A processed ``PIL.Image`` in mode "L".
    """
    Image, ImageOps, ImageFilter = _require_pil()
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # honor phone rotation metadata
    # PERF: cap resolution — OCR gains little above ~2200px but RAM/time grow fast.
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    return img
