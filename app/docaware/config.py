"""docaware/config.py — Central configuration (offline, RAM-budgeted).

Every tunable lives here so we can tighten the memory/speed envelope for the
ADTC Standard Laptop (8 GB RAM, 4 vCPU, CPU-only) from one place. Values may be
overridden via environment variables (prefix ``ADTC_``) without code changes.

Constraint: keep total resident memory well under the 7 GB scoring budget. The
LLM is the dominant consumer; OCR/embedding models are loaded lazily and freed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# Repository root = two levels up from this file (app/docaware/config.py → repo/).
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"
DATA_DIR = REPO_ROOT / "app" / "data"  # local index + working files (gitignored)
OUTPUT_DIR = REPO_ROOT / "app" / "output"  # generated downloadable documents


def _env(name: str, default: str) -> str:
    """Read ``ADTC_<NAME>`` from the environment, falling back to ``default``."""
    return os.environ.get(f"ADTC_{name}", default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class LLMConfig:
    """Settings for the local GGUF chat model (the ADTC-scored artifact).

    Defaults target the 8 GB laptop: a modest context window and thread count
    sized for 4 vCPU. ``n_ctx`` is the single biggest RAM lever after model size.
    """

    # Scored model — chosen for throughput + low RAM on a 4-core/8 GB CPU (see
    # docs/SELF_ASSESSMENT.md benchmarks). RAG grounds answers, so a small model
    # keeps accuracy while winning Sperf (30%) and Seff (20%).
    model_filename: str = field(
        default_factory=lambda: _env("MODEL_FILE", "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
    )
    # PERF: n_ctx is the biggest RAM lever after model size; 2048 fits RAG context
    # (top-k chunks + short history) with room to spare and shrinks the KV cache.
    n_ctx: int = field(default_factory=lambda: _env_int("N_CTX", 2048))
    n_threads: int = field(default_factory=lambda: _env_int("N_THREADS", 4))  # 4 vCPU target
    n_batch: int = field(default_factory=lambda: _env_int("N_BATCH", 512))
    max_tokens: int = field(default_factory=lambda: _env_int("MAX_TOKENS", 1024))
    temperature: float = 0.3  # low: factual drafting/summarization, not creative
    # PERF: flash attention is faster and uses less memory; q8_0 KV cache halves
    # KV RAM vs f16 with negligible quality loss. Both validated by the model bench.
    flash_attn: bool = field(default_factory=lambda: _env("FLASH_ATTN", "1") == "1")
    kv_quant_q8: bool = field(default_factory=lambda: _env("KV_Q8", "1") == "1")
    # mmap keeps weights pageable (lower RSS); mlock would pin them (risks OOM on 8 GB).
    use_mmap: bool = True
    use_mlock: bool = False

    @property
    def model_path(self) -> Path:
        return MODEL_DIR / self.model_filename


@dataclass
class EmbeddingConfig:
    """Settings for the local embedding model used by RAG.

    Also a GGUF model run through llama.cpp — keeps the whole stack on ggml with
    no PyTorch dependency. Small (~30–90 MB), loaded only during ingestion/query.
    """

    model_filename: str = field(
        default_factory=lambda: _env("EMBED_FILE", "bge-small-en-v1.5-q4_k_m.gguf")
    )
    n_ctx: int = 512
    n_threads: int = field(default_factory=lambda: _env_int("N_THREADS", 4))
    dim: int = 384  # bge-small-en-v1.5 embedding dimension

    @property
    def model_path(self) -> Path:
        return MODEL_DIR / self.model_filename


def _detect_mtmd_cli() -> str:
    """Find the native ``llama-mtmd-cli`` binary used for vision OCR.

    Honors ``ADTC_MTMD_CLI``; else searches PATH; else common local build dirs.
    Returns "" if not found (digitize then reports a clear, actionable error).
    """
    import shutil

    explicit = os.environ.get("ADTC_MTMD_CLI")
    if explicit:
        return explicit
    found = shutil.which("llama-mtmd-cli") or shutil.which("llama-mtmd-cli.exe")
    if found:
        return found
    for c in (REPO_ROOT / "bin" / "llama-mtmd-cli.exe", REPO_ROOT / "bin" / "llama-mtmd-cli"):
        if c.exists():
            return str(c)
    return ""


@dataclass
class VisionConfig:
    """Settings for the OCR/digitize vision model (DeepSeek-OCR via llama.cpp).

    DeepSeek-OCR uses "optical compression" (few vision tokens) → fast on CPU and
    robust to real-world scans/photos. Run through the native ``llama-mtmd-cli`` (a
    subprocess) with the model's own chat template (`--jinja`) — the reliable path
    for these OCR VLMs. App-side only (not the ADTC-benchmarked model).
    """

    model_filename: str = field(default_factory=lambda: _env("VLM_FILE", "DeepSeek-OCR-Q8_0.gguf"))
    mmproj_filename: str = field(
        default_factory=lambda: _env("VLM_MMPROJ", "mmproj-DeepSeek-OCR-Q8_0.gguf")
    )
    mtmd_cli: str = field(default_factory=_detect_mtmd_cli)
    prompt: str = field(
        default_factory=lambda: _env("VLM_PROMPT", "Convert this document to markdown.")
    )
    n_ctx: int = field(default_factory=lambda: _env_int("VLM_N_CTX", 8192))
    n_threads: int = field(default_factory=lambda: _env_int("VLM_THREADS", os.cpu_count() or 4))
    max_tokens: int = field(default_factory=lambda: _env_int("VLM_MAX_TOKENS", 2048))
    temperature: float = 0.2  # faithful transcription; matches DeepSeek-OCR guidance

    @property
    def model_path(self) -> Path:
        return MODEL_DIR / self.model_filename

    @property
    def mmproj_path(self) -> Path:
        return MODEL_DIR / self.mmproj_filename


@dataclass
class RAGConfig:
    """Chunking + retrieval parameters for document Q&A."""

    chunk_chars: int = field(default_factory=lambda: _env_int("CHUNK_CHARS", 1200))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 200))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 4))
    # Conversation memory: how many recent messages (user+assistant) to keep in context.
    max_history_messages: int = field(default_factory=lambda: _env_int("MAX_HISTORY", 8))
    sessions_dir: Path = DATA_DIR / "sessions"


def _detect_tesseract() -> str:
    """Return a usable tesseract command/path.

    Honors ``ADTC_TESSERACT_CMD``; otherwise tries the bare name (PATH) and the
    standard Windows install location so the app works without PATH changes.
    """
    explicit = os.environ.get("ADTC_TESSERACT_CMD")
    if explicit:
        return explicit
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return "tesseract"  # assume on PATH (Linux/macOS default)


@dataclass
class OCRConfig:
    """OCR pipeline settings. Math/formula OCR is optional ("added advantage")."""

    # Digitize engine: "vlm" (vision model, best for handwriting) or "tesseract"
    # (printed text only). "auto" uses the VLM when its weights are present.
    engine: str = field(default_factory=lambda: _env("OCR_ENGINE", "auto"))
    tesseract_lang: str = field(default_factory=lambda: _env("OCR_LANG", "eng"))
    tesseract_cmd: str = field(default_factory=_detect_tesseract)
    enable_formula_ocr: bool = field(default_factory=lambda: _env("ENABLE_FORMULA_OCR", "1") == "1")
    dpi: int = 300


@dataclass
class AppConfig:
    """Top-level config bundle. Construct once and pass down."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)

    def ensure_dirs(self) -> None:
        """Create working directories if missing (safe to call repeatedly)."""
        for d in (DATA_DIR, OUTPUT_DIR, self.rag.sessions_dir):
            d.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view (paths rendered as strings)."""
        d = asdict(self)
        for section in d.values():
            for k, v in list(section.items()):
                if isinstance(v, Path):
                    section[k] = str(v)
        return d


# Module-level singleton for convenience; callers may build their own.
CONFIG = AppConfig()
