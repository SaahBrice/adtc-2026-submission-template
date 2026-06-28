"""docaware/rag/store.py — Minimal persistent vector store (NumPy-backed).

A dependency-light cosine-similarity index. For the SME-scale corpora we target
(thousands of chunks), brute-force NumPy search is fast enough and avoids pulling
in FAISS. The index persists to disk as a .npy matrix + JSON sidecar so it is
fully offline and survives restarts.

If a corpus ever outgrows this, swap in FAISS behind the same interface.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunk import Chunk


class VectorStore:
    """Cosine-similarity store over normalized embedding vectors."""

    def __init__(self, dim: int):
        self.dim = dim
        self._vectors = np.empty((0, dim), dtype=np.float32)
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        """L2-normalize rows so dot product == cosine similarity."""
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Append chunks and their embedding vectors to the index.

        Args:
            chunks: Chunks being indexed.
            vectors: Parallel list of embedding vectors (same length as chunks).
        """
        if not chunks:
            return
        mat = self._normalize(np.asarray(vectors, dtype=np.float32))
        self._vectors = np.vstack([self._vectors, mat]) if len(self) else mat
        self._chunks.extend(chunks)

    def search(self, query_vector: list[float], top_k: int = 4) -> list[tuple[Chunk, float]]:
        """Return the ``top_k`` most similar chunks to a query vector.

        Args:
            query_vector: Embedding of the query.
            top_k: Number of results to return.

        Returns:
            List of ``(chunk, score)`` pairs sorted by descending similarity.
        """
        if len(self) == 0:
            return []
        q = self._normalize(np.asarray([query_vector], dtype=np.float32))[0]
        scores = self._vectors @ q
        k = min(top_k, len(self))
        # argpartition for the top-k, then sort just those — O(n) vs full sort.
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self._chunks[i], float(scores[i])) for i in idx]

    # --- persistence ---------------------------------------------------------

    def save(self, index_dir: str | Path) -> None:
        """Persist vectors (.npy) and chunk metadata (.json) to ``index_dir``."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "vectors.npy", self._vectors)
        meta = {"dim": self.dim, "chunks": [asdict(c) for c in self._chunks]}
        (index_dir / "chunks.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: str | Path) -> "VectorStore":
        """Load a previously saved index. Returns an empty store if none exists."""
        index_dir = Path(index_dir)
        meta_path = index_dir / "chunks.json"
        vec_path = index_dir / "vectors.npy"
        if not meta_path.exists() or not vec_path.exists():
            return cls(dim=0)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        store = cls(dim=meta["dim"])
        store._vectors = np.load(vec_path)
        store._chunks = [Chunk(**c) for c in meta["chunks"]]
        return store
