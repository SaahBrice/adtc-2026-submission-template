"""adtc_notes/rag/chunk.py — Text chunking (pure, dependency-free, unit-tested).

Splits long documents into overlapping, paragraph-aware chunks suitable for
embedding. No I/O and no heavy deps, so this is fast to test and reason about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Split on blank lines first to keep paragraphs intact where possible.
_PARA_SPLIT = re.compile(r"\n\s*\n")


@dataclass
class Chunk:
    """A retrievable unit of text plus provenance metadata.

    Attributes:
        text: The chunk content.
        source: Originating document name/path.
        index: Position of this chunk within its source (0-based).
        metadata: Free-form extra fields (page number, section, …).
    """

    text: str
    source: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize(text: str) -> str:
    """Collapse Windows newlines and trailing whitespace; keep paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


def chunk_text(
    text: str,
    source: str,
    *,
    chunk_chars: int = 1200,
    overlap: int = 200,
) -> list[Chunk]:
    """Split ``text`` into overlapping character-bounded chunks.

    Paragraphs are accumulated until adding the next would exceed ``chunk_chars``;
    the tail ``overlap`` characters are carried into the next chunk to preserve
    context across boundaries. Oversized single paragraphs are hard-split.

    Args:
        text: Full document text.
        source: Label stored on every produced chunk.
        chunk_chars: Soft maximum characters per chunk.
        overlap: Characters of trailing context repeated into the next chunk.

    Returns:
        Ordered list of ``Chunk`` (possibly empty for blank input).
    """
    text = _normalize(text)
    if not text:
        return []
    if overlap >= chunk_chars:  # guard against pathological config
        overlap = chunk_chars // 4

    paragraphs = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    chunks: list[Chunk] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            chunks.append(Chunk(text=buf.strip(), source=source, index=len(chunks)))
        buf = ""

    for para in paragraphs:
        # Hard-split paragraphs longer than the chunk budget.
        while len(para) > chunk_chars:
            flush()
            chunks.append(Chunk(text=para[:chunk_chars], source=source, index=len(chunks)))
            para = para[chunk_chars - overlap :]
        if len(buf) + len(para) + 2 > chunk_chars:
            tail = buf[-overlap:] if overlap else ""
            flush()
            buf = (tail + "\n\n" + para).strip()
        else:
            buf = (buf + "\n\n" + para).strip() if buf else para
    flush()
    return chunks
