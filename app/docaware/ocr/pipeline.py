"""docaware/ocr/pipeline.py — Image → Markdown via the DeepSeek-OCR engine.

A thin entry point used by both PATH A (digitize) and RAG image ingestion. The
heavy lifting lives in ``vlm.py`` (DeepSeek-OCR through native llama-mtmd-cli).
"""

from __future__ import annotations

from pathlib import Path

from ..config import CONFIG, AppConfig


def image_to_markdown(path: str | Path, cfg: AppConfig | None = None) -> str:
    """Transcribe an image to clean Markdown using the DeepSeek-OCR engine."""
    from .vlm import get_vision

    cfg = cfg or CONFIG
    return get_vision(cfg.vision).transcribe(path)
