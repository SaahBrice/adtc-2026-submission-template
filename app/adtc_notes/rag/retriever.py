"""adtc_notes/rag/retriever.py — Orchestrates ingest → chunk → embed → store → query.

This is the public RAG entry point used by the CLI/UI. It ties together the
dependency-light pieces (chunking, NumPy store) with the lazy GGUF backends
(embedder, LLM) so the heavy models load only when actually needed.
"""

from __future__ import annotations

from pathlib import Path

from ..config import CONFIG, AppConfig
from ..llm import get_llm, prompts
from .chunk import chunk_text
from .embed import get_embedder
from .ingest import load_text
from .store import VectorStore


class Retriever:
    """Builds and queries a local document index for grounded Q&A."""

    def __init__(self, cfg: AppConfig | None = None):
        self.cfg = cfg or CONFIG
        self.cfg.ensure_dirs()
        self.store = VectorStore.load(self.cfg.rag.index_dir)

    def add_document(self, path: str | Path) -> int:
        """Ingest, chunk, embed, and index a single document.

        Args:
            path: Path to a supported document or image.

        Returns:
            Number of chunks added.
        """
        text = load_text(path)
        chunks = chunk_text(
            text,
            source=Path(path).name,
            chunk_chars=self.cfg.rag.chunk_chars,
            overlap=self.cfg.rag.chunk_overlap,
        )
        if not chunks:
            return 0
        embedder = get_embedder(self.cfg.embedding)
        # Initialize store dim from the embedder on first use.
        if self.store.dim == 0:
            self.store = VectorStore(dim=self.cfg.embedding.dim)
        vectors = embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        return len(chunks)

    def add_documents(self, paths: list[str | Path]) -> int:
        """Index several documents; returns total chunks added."""
        total = sum(self.add_document(p) for p in paths)
        self.persist()
        return total

    def persist(self) -> None:
        """Save the current index to disk."""
        self.store.save(self.cfg.rag.index_dir)

    def retrieve(self, question: str, top_k: int | None = None) -> list[tuple]:
        """Return ``(chunk, score)`` results most relevant to ``question``."""
        embedder = get_embedder(self.cfg.embedding)
        qv = embedder.embed_one(question)
        return self.store.search(qv, top_k or self.cfg.rag.top_k)

    def ask(self, question: str, top_k: int | None = None) -> dict:
        """Answer a question grounded in retrieved context.

        Returns:
            ``{"answer": str, "sources": list[str], "contexts": list[str]}``.
        """
        results = self.retrieve(question, top_k)
        contexts = [c.text for c, _ in results]
        sources = [f"{c.source}#{c.index}" for c, _ in results]
        if not contexts:
            return {
                "answer": "No documents indexed yet — add some first.",
                "sources": [],
                "contexts": [],
            }
        llm = get_llm(self.cfg)
        answer = llm.chat(prompts.qa_messages(question, contexts))
        return {"answer": answer, "sources": sources, "contexts": contexts}
