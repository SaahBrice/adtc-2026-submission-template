"""docaware/ocr/vlm.py — Digitize engine: DeepSeek-OCR via native llama-mtmd-cli.

DeepSeek-OCR ("optical compression": few vision tokens) is fast on CPU (~40 s/page
vs minutes for a general VLM) and robust to real-world scans/photos. It is run
through the native ``llama-mtmd-cli`` binary as a subprocess, using the model's own
chat template (``--jinja``) — the reliable path for these OCR VLMs (the Python
bindings' generic handler mis-formats them and loops).

App-side only (not the ADTC-benchmarked model). The subprocess loads the weights
transiently and frees them on exit; before launching it we evict the in-process
chat/embedding models so peak RAM stays within the 8 GB budget. Fully offline.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..config import CONFIG, VisionConfig
from ..errors import BackendNotInstalledError, ModelNotFoundError


def is_available(cfg: VisionConfig | None = None) -> bool:
    """True if the DeepSeek-OCR weights, mmproj, and llama-mtmd-cli are all present."""
    cfg = cfg or CONFIG.vision
    return bool(cfg.mtmd_cli) and cfg.model_path.exists() and cfg.mmproj_path.exists()


class VisionEngine:
    """Stateless wrapper that shells out to llama-mtmd-cli for one image."""

    def __init__(self, cfg: VisionConfig | None = None):
        self.cfg = cfg or CONFIG.vision
        if not self.cfg.model_path.exists() or not self.cfg.mmproj_path.exists():
            raise ModelNotFoundError(
                f"DeepSeek-OCR weights missing ({self.cfg.model_path.name} / "
                f"{self.cfg.mmproj_path.name}). Run `bash download_model.sh`."
            )
        if not self.cfg.mtmd_cli:
            raise BackendNotInstalledError(
                "llama-mtmd-cli not found. Install llama.cpp (it provides llama-mtmd-cli)\n"
                "and put it on PATH, or set ADTC_MTMD_CLI to its full path.\n"
                "Prebuilt binaries: https://github.com/ggml-org/llama.cpp/releases"
            )

    def transcribe(self, image_path: str | Path) -> str:
        """Transcribe one image to Markdown via llama-mtmd-cli. Returns the Markdown."""
        # Free the in-process chat/embedder so the OCR subprocess has RAM headroom.
        from .. import _models

        _models.drop("llm")
        _models.drop("embed")

        cmd = [
            self.cfg.mtmd_cli,
            "-m",
            str(self.cfg.model_path),
            "--mmproj",
            str(self.cfg.mmproj_path),
            "--image",
            str(image_path),
            "-p",
            self.cfg.prompt,
            "-c",
            str(self.cfg.n_ctx),
            "-n",
            str(self.cfg.max_tokens),
            "--temp",
            str(self.cfg.temperature),
            "--top-p",
            "0.9",
            "--top-k",
            "0",
            "--repeat-penalty",
            "1.0",
            "-t",
            str(self.cfg.n_threads),
            "-ngl",
            "0",  # CPU-only, matching the target hardware
            "--jinja",  # use the model's own chat template (critical for OCR VLMs)
        ]
        proc = subprocess.run(cmd, capture_output=True)  # binary output → decode utf-8
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", "replace")[-800:]
            raise BackendNotInstalledError(f"llama-mtmd-cli failed:\n{err}")
        return proc.stdout.decode("utf-8", "replace").strip()


def get_vision(cfg: VisionConfig | None = None) -> VisionEngine:
    """Return a vision engine (cheap; the heavy model runs in a subprocess per call)."""
    return VisionEngine(cfg or CONFIG.vision)
