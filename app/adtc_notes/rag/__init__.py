"""adtc_notes.rag — Offline retrieval-augmented generation over local documents."""

from .chunk import Chunk, chunk_text
from .store import VectorStore
from .retriever import Retriever

__all__ = ["Chunk", "chunk_text", "VectorStore", "Retriever"]
