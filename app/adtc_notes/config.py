"""adtc_notes/config.py — Central configuration (offline, RAM-budgeted).

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

# Repository root = two levels up from this file (app/adtc_notes/config.py → repo/).
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

    # Provisional default — see REPORT.md "Model selection". MIT-licensed, strong
    # at summarization/drafting, ~2.2 GB at Q4_K_M. Final pick confirmed via profiler.
    model_filename: str = field(
        default_factory=lambda: _env("MODEL_FILE", "Phi-3.5-mini-instruct-Q4_K_M.gguf")
    )
    n_ctx: int = field(default_factory=lambda: _env_int("N_CTX", 4096))
    n_threads: int = field(default_factory=lambda: _env_int("N_THREADS", 4))  # 4 vCPU target
    n_batch: int = field(default_factory=lambda: _env_int("N_BATCH", 256))
    max_tokens: int = field(default_factory=lambda: _env_int("MAX_TOKENS", 1024))
    temperature: float = 0.3  # low: factual drafting/summarization, not creative
    # PERF: mmap keeps weights on disk and pages in on demand (lower RSS); mlock would
    # pin them in RAM (faster, but risks OOM on 8 GB) — leave mlock off by default.
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


@dataclass
class RAGConfig:
    """Chunking + retrieval parameters for document Q&A."""

    chunk_chars: int = field(default_factory=lambda: _env_int("CHUNK_CHARS", 1200))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 200))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 4))
    index_dir: Path = DATA_DIR / "index"


@dataclass
class OCRConfig:
    """OCR pipeline settings. Math/formula OCR is optional ("added advantage")."""

    tesseract_lang: str = field(default_factory=lambda: _env("OCR_LANG", "eng"))
    enable_formula_ocr: bool = field(default_factory=lambda: _env("ENABLE_FORMULA_OCR", "1") == "1")
    dpi: int = 300


@dataclass
class AppConfig:
    """Top-level config bundle. Construct once and pass down."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)

    def ensure_dirs(self) -> None:
        """Create working directories if missing (safe to call repeatedly)."""
        for d in (DATA_DIR, OUTPUT_DIR, self.rag.index_dir):
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
