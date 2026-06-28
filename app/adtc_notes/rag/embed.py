"""adtc_notes/rag/embed.py — Local text embeddings via a GGUF model (llama.cpp).

Reuses the llama.cpp backend for embeddings so RAG adds no PyTorch dependency —
critical for the 8 GB target. The embedding model is small and loaded lazily,
only while ingesting or querying, then eligible for GC.

Constraint: keep the embedder out of memory during pure LLM benchmarking.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import CONFIG, EmbeddingConfig
from ..errors import BackendNotInstalledError, ModelNotFoundError


class Embedder:
    """Loads a GGUF embedding model and turns text into vectors."""

    def __init__(self, cfg: EmbeddingConfig | None = None):
        self.cfg = cfg or CONFIG.embedding
        if not self.cfg.model_path.exists():
            raise ModelNotFoundError(
                f"Embedding model not found at {self.cfg.model_path}.\n"
                f"Run `bash download_model.sh` (it fetches the embedder too)."
            )
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise BackendNotInstalledError(
                "llama-cpp-python is not installed: pip install llama-cpp-python"
            ) from exc
        self._llm = Llama(
            model_path=str(self.cfg.model_path),
            n_ctx=self.cfg.n_ctx,
            n_threads=self.cfg.n_threads,
            embedding=True,
            verbose=False,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input string.

        Args:
            texts: Strings to embed.

        Returns:
            List of float vectors (each length == ``cfg.dim``).
        """
        out = self._llm.create_embedding(texts)
        return [row["embedding"] for row in out["data"]]

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string (convenience)."""
        return self.embed([text])[0]


@lru_cache(maxsize=1)
def get_embedder(cfg: EmbeddingConfig | None = None) -> Embedder:
    """Process-wide singleton embedder (loads weights once)."""
    return Embedder(cfg or CONFIG.embedding)
