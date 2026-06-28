"""docaware.ocr — Image → text/formula extraction (the "added advantage")."""

from .pipeline import ocr_image_to_text, digitize_image

__all__ = ["ocr_image_to_text", "digitize_image"]
