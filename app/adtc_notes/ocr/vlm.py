"""adtc_notes/ocr/vlm.py — Vision-language digitize engine (Qwen2.5-VL via llama.cpp).

Reads a photographed/scanned page — including cursive handwriting and math — and
emits clean Markdown with LaTeX directly. This is the high-quality digitize path;
Tesseract remains a printed-text fallback. Fully offline, CPU, no PyTorch.

Memory: managed by ``adtc_notes._models`` so loading the VLM evicts the chat LLM
and embedder first (the VLM is ~3.3 GB and must run alone on an 8 GB machine).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from ..config import CONFIG, VisionConfig
from ..errors import BackendNotInstalledError, ModelNotFoundError

# One-shot instruction: faithful transcription into structured Markdown + LaTeX.
TRANSCRIBE_PROMPT = (
    "You are a meticulous document transcription engine. Transcribe the page in the "
    "image into clean, faithful GitHub-flavored Markdown.\n"
    "Rules:\n"
    "- Preserve the reading order and structure: headings, bullet/numbered lists, tables.\n"
    "- Render every mathematical expression as LaTeX: inline as $...$ and displayed "
    "equations as $$...$$.\n"
    "- Represent each diagram or figure with a concise italic caption like "
    "*(Figure: short description of what it shows)*.\n"
    "- Transcribe exactly what is on the page. Do not invent content; do not omit content.\n"
    "- Output only the Markdown, with no preamble or commentary."
)


def is_available(cfg: VisionConfig | None = None) -> bool:
    """Return True if both the VLM weights and its mmproj projector are present."""
    cfg = cfg or CONFIG.vision
    return cfg.model_path.exists() and cfg.mmproj_path.exists()


def _image_data_uri(path: str | Path, *, max_side: int = 1280) -> str:
    """Load, EXIF-orient, downscale, and base64-encode an image as a JPEG data URI.

    Bounding the longest side keeps the vision token count (and latency) sane on CPU
    without hurting legibility for document text.
    """
    try:
        from PIL import Image, ImageOps  # type: ignore
    except ImportError as exc:
        raise BackendNotInstalledError("Pillow not installed: pip install pillow") from exc
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


class VisionModel:
    """A loaded Qwen2.5-VL model that transcribes images to Markdown."""

    def __init__(self, cfg: VisionConfig | None = None):
        self.cfg = cfg or CONFIG.vision
        if not self.cfg.model_path.exists():
            raise ModelNotFoundError(
                f"Vision model not found at {self.cfg.model_path}. Run `bash download_model.sh`."
            )
        if not self.cfg.mmproj_path.exists():
            raise ModelNotFoundError(
                f"Vision projector (mmproj) not found at {self.cfg.mmproj_path}."
            )
        try:
            from llama_cpp import Llama  # type: ignore
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler  # type: ignore
        except ImportError as exc:
            raise BackendNotInstalledError(
                "llama-cpp-python is not installed: pip install llama-cpp-python"
            ) from exc

        handler = Qwen25VLChatHandler(clip_model_path=str(self.cfg.mmproj_path), verbose=False)
        self._llm = Llama(
            model_path=str(self.cfg.model_path),
            chat_handler=handler,
            n_ctx=self.cfg.n_ctx,
            n_threads=self.cfg.n_threads,
            verbose=False,
        )

    def transcribe(self, image_path: str | Path) -> str:
        """Transcribe one image into clean Markdown (with LaTeX for formulas)."""
        uri = _image_data_uri(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TRANSCRIBE_PROMPT},
                    {"type": "image_url", "image_url": {"url": uri}},
                ],
            }
        ]
        out = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        )
        return out["choices"][0]["message"]["content"].strip()


def get_vision(cfg: VisionConfig | None = None) -> VisionModel:
    """Return the active ``VisionModel``, loading it if needed.

    Evicts the heavy chat LLM and embedder first to free RAM for the ~3.3 GB VLM.
    """
    from .. import _models

    existing = _models.get("vision")
    if existing is not None:
        return existing
    return _models.register("vision", VisionModel(cfg or CONFIG.vision), evict=("llm", "embed"))
