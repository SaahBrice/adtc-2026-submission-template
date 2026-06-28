"""docaware/_models.py — Single-active-model manager (8 GB RAM discipline).

The app uses up to three GGUF models: the chat LLM (~2.4 GB), the vision VLM
(~3.3 GB), and a tiny embedder (~25 MB). Loading the big ones together would
exceed the 8 GB budget, so heavy models are mutually evicted: digitize loads the
VLM (and frees the chat LLM); Q&A loads the chat LLM (and frees the VLM). The
tiny embedder may stay resident alongside the chat LLM for RAG.

Registry keys: "llm", "vision", "embed".
"""

from __future__ import annotations

import gc
from typing import Any

_ACTIVE: dict[str, Any] = {}


def get(key: str) -> Any | None:
    """Return the live instance for ``key`` if loaded, else None."""
    return _ACTIVE.get(key)


def register(key: str, obj: Any, *, evict: tuple[str, ...] = ()) -> Any:
    """Store ``obj`` under ``key`` after evicting the named conflicting models.

    Args:
        key: Registry key for this model.
        obj: The loaded model instance.
        evict: Keys of heavy models to free first to reclaim RAM.

    Returns:
        ``obj`` (for convenient chaining).
    """
    for k in evict:
        drop(k)
    _ACTIVE[key] = obj
    return obj


def drop(key: str) -> None:
    """Free and forget the model under ``key`` (no-op if absent)."""
    obj = _ACTIVE.pop(key, None)
    if obj is None:
        return
    # llama_cpp models expose .close(); fall back to plain deref + GC otherwise.
    inner = getattr(obj, "_llm", obj)
    close = getattr(inner, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
    del obj
    gc.collect()
