"""docaware/llm/client.py — Thin wrapper over a local GGUF model (llama.cpp).

Wraps ``llama-cpp-python`` (which embeds llama.cpp) so the rest of the app speaks
a small, stable interface and never imports the backend directly. CPU-only and
fully offline: weights are loaded from disk, no network calls are ever made.

Constraint: a single model instance is cached process-wide — loading two copies
would blow the 7 GB RAM budget. Use ``get_llm()`` rather than constructing many.
"""

from __future__ import annotations

from typing import Iterator

from ..config import CONFIG, AppConfig, LLMConfig
from ..errors import BackendNotInstalledError, ModelNotFoundError


def _import_llama():
    """Import llama_cpp lazily so the package is usable without the backend.

    Returns:
        The ``llama_cpp.Llama`` class.

    Raises:
        BackendNotInstalledError: If llama-cpp-python is not installed.
    """
    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise BackendNotInstalledError(
            "llama-cpp-python is not installed. Install it with:\n"
            "  pip install llama-cpp-python\n"
            "(CPU build; see app/README.md for offline/build notes)."
        ) from exc
    return Llama


class LLMClient:
    """A loaded local chat model exposing ``chat`` and ``complete``."""

    def __init__(self, cfg: LLMConfig | None = None):
        self.cfg = cfg or CONFIG.llm
        if not self.cfg.model_path.exists():
            raise ModelNotFoundError(
                f"Model weight not found at {self.cfg.model_path}.\n"
                f"Run `bash download_model.sh` first."
            )
        Llama = _import_llama()
        # PERF: n_ctx/n_batch dominate scratch memory; flash attention speeds up and
        # shrinks attention memory; q8_0 KV cache halves KV RAM vs f16. GGML q8_0 == 8.
        kwargs = dict(
            model_path=str(self.cfg.model_path),
            n_ctx=self.cfg.n_ctx,
            n_threads=self.cfg.n_threads,
            n_batch=self.cfg.n_batch,
            use_mmap=self.cfg.use_mmap,
            use_mlock=self.cfg.use_mlock,
            embedding=False,
            verbose=False,
        )
        if self.cfg.flash_attn:
            kwargs["flash_attn"] = True
        if self.cfg.kv_quant_q8:
            kwargs["type_k"] = 8
            kwargs["type_v"] = 8
        self._llm = Llama(**kwargs)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Run a chat completion over OpenAI-style messages.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            max_tokens: Override the configured generation cap.
            temperature: Override the configured sampling temperature.
            stream: If True, yield token chunks instead of a single string.

        Returns:
            The assistant's reply (str), or an iterator of text chunks if streaming.
        """
        kwargs = dict(
            messages=messages,
            max_tokens=max_tokens or self.cfg.max_tokens,
            temperature=self.cfg.temperature if temperature is None else temperature,
            stream=stream,
        )
        if stream:
            return self._stream_chat(kwargs)
        out = self._llm.create_chat_completion(**kwargs)
        return out["choices"][0]["message"]["content"].strip()

    def _stream_chat(self, kwargs: dict) -> Iterator[str]:
        """Yield incremental content deltas from a streaming chat completion."""
        for chunk in self._llm.create_chat_completion(**kwargs):
            delta = chunk["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                yield delta["content"]

    def complete(self, prompt: str, **kwargs) -> str:
        """Convenience single-turn completion using a user message."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)  # type: ignore[return-value]


def get_llm(cfg: AppConfig | None = None) -> LLMClient:
    """Return the active chat ``LLMClient``, loading it if needed.

    Managed by ``docaware._models`` so loading the chat model evicts the heavy
    vision model first (they must not be co-resident on an 8 GB machine). The
    tiny embedder is allowed to stay loaded for RAG.
    """
    from .. import _models

    existing = _models.get("llm")
    if existing is not None:
        return existing
    cfg = cfg or CONFIG
    return _models.register("llm", LLMClient(cfg.llm), evict=("vision",))
