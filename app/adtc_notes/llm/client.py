"""adtc_notes/llm/client.py — Thin wrapper over a local GGUF model (llama.cpp).

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
        # PERF: n_ctx and n_batch dominate scratch memory; keep modest for 8 GB.
        self._llm = Llama(
            model_path=str(self.cfg.model_path),
            n_ctx=self.cfg.n_ctx,
            n_threads=self.cfg.n_threads,
            n_batch=self.cfg.n_batch,
            use_mmap=self.cfg.use_mmap,
            use_mlock=self.cfg.use_mlock,
            embedding=False,
            verbose=False,
        )

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


_LLM: LLMClient | None = None


def get_llm(cfg: AppConfig | None = None) -> LLMClient:
    """Return a process-wide singleton ``LLMClient`` (loads weights once).

    A plain module global rather than ``lru_cache`` because the config is a
    mutable dataclass (unhashable). First call wins; later ``cfg`` args are ignored.
    """
    global _LLM
    if _LLM is None:
        cfg = cfg or CONFIG
        _LLM = LLMClient(cfg.llm)
    return _LLM
