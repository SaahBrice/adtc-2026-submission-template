"""docaware.ocr — Image → Markdown (DeepSeek-OCR via native llama.cpp)."""

from .pipeline import image_to_markdown
from .vlm import get_vision, is_available

__all__ = ["image_to_markdown", "get_vision", "is_available"]
