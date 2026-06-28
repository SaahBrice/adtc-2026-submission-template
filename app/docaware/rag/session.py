"""docaware/rag/session.py — Persistent chat sessions (enterprise conversational RAG).

Each chat is an isolated session with its own documents, vector index, and
conversation history, persisted under ``data/sessions/<id>/``. This gives true
multi-chat behaviour ("new chat with different documents") and survives restarts.

Answering follows the standard conversational-RAG pipeline:
  1. history-aware retrieval — rewrite the follow-up into a standalone query using
     recent history, then embed/search with that (so "what about Q4?" resolves);
  2. retrieve top-k chunks (each carrying its source file + page);
  3. answer grounded in those chunks, citing ``[file p.N]``, with recent history
     in context for continuity.

Offline and CPU-only; reuses the GGUF embedder/LLM and the NumPy vector store.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from ..config import CONFIG, AppConfig
from ..llm import get_llm, prompts
from .chunk import chunk_text
from .embed import get_embedder
from .ingest import load_pages
from .store import VectorStore


def _sessions_root(cfg: AppConfig) -> Path:
    cfg.rag.sessions_dir.mkdir(parents=True, exist_ok=True)
    return cfg.rag.sessions_dir


class ChatSession:
    """One chat: its documents, vector index, and conversation history."""

    def __init__(self, session_id: str, cfg: AppConfig | None = None):
        self.cfg = cfg or CONFIG
        self.id = session_id
        self.dir = _sessions_root(self.cfg) / session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.store = VectorStore.load(self.dir / "index")
        self.meta = self._load_json(
            "meta.json",
            {
                "id": session_id,
                "title": "New chat",
                "created": time.time(),
                "documents": [],
                "summary": "",  # running summary of older turns (long-chat memory)
                "summarized_upto": 0,  # history index already folded into summary
            },
        )
        self.history: list[dict[str, str]] = self._load_json("history.json", [])

    # --- persistence ---------------------------------------------------------

    def _load_json(self, name: str, default):
        p = self.dir / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return default

    def _save_json(self, name: str, obj) -> None:
        (self.dir / name).write_text(json.dumps(obj), encoding="utf-8")

    def save(self) -> None:
        self.store.save(self.dir / "index")
        self._save_json("meta.json", self.meta)
        self._save_json("history.json", self.history)

    @property
    def title(self) -> str:
        return self.meta.get("title", "New chat")

    def summary(self) -> dict:
        """Compact descriptor for listing sessions in the UI."""
        return {
            "id": self.id,
            "title": self.title,
            "created": self.meta.get("created", 0),
            "documents": self.meta.get("documents", []),
            "indexed_chunks": len(self.store),
        }

    # --- ingestion -----------------------------------------------------------

    def add_documents(self, paths: list[str | Path]) -> int:
        """Ingest, chunk (page-tagged), embed, and index documents into this chat."""
        embedder = get_embedder(self.cfg.embedding)
        if self.store.dim == 0:
            self.store = VectorStore(dim=self.cfg.embedding.dim)
        added = 0
        for path in paths:
            name = Path(path).name
            chunks = []
            for page, text in load_pages(path):
                for ch in chunk_text(
                    text,
                    source=name,
                    chunk_chars=self.cfg.rag.chunk_chars,
                    overlap=self.cfg.rag.chunk_overlap,
                ):
                    ch.metadata["page"] = page  # carry page for citations
                    chunks.append(ch)
            if not chunks:
                continue
            vectors = embedder.embed([c.text for c in chunks])
            self.store.add(chunks, vectors)
            added += len(chunks)
            if name not in self.meta["documents"]:
                self.meta["documents"].append(name)
        # Name the chat after its first document until the user renames it.
        if self.meta.get("title") == "New chat" and self.meta["documents"]:
            self.meta["title"] = self.meta["documents"][0]
        self.save()
        return added

    def remove_document(self, name: str) -> int:
        """Remove a document and its chunks from this chat. Returns chunks removed."""
        removed = self.store.remove_source(name)
        self.meta["documents"] = [d for d in self.meta.get("documents", []) if d != name]
        self.save()
        return removed

    # --- conversational QA ---------------------------------------------------

    @staticmethod
    def _label(chunk) -> str:
        page = chunk.metadata.get("page")
        return f"{chunk.source} p.{page}" if page else chunk.source

    def _retrieval_query(self, question, history, summary):
        """History-aware query: rewrite a follow-up into a standalone question."""
        if not history and not summary:
            return question
        rewritten = get_llm(self.cfg).chat(
            prompts.condense_question_messages(question, history, summary), max_tokens=96
        )
        return (rewritten or question).strip() or question

    _NO_DOCS = "No documents in this chat yet — add some first."

    def _prepare(self, question: str, top_k: int | None = None):
        """History-aware retrieval. Returns ``(window, labeled, sources, summary)`` or None.

        ``window`` is the recent raw turns; ``summary`` is the running summary of older
        turns (long-chat memory). Both feed the condense step and the answer prompt.
        """
        if len(self.store) == 0:
            return None
        window = self.history[-self.cfg.rag.max_history_messages :]
        summary = self.meta.get("summary") or None
        query = self._retrieval_query(question, window, summary)
        embedder = get_embedder(self.cfg.embedding)
        results = self.store.search(embedder.embed_one(query), top_k or self.cfg.rag.top_k)
        labeled = [(self._label(c), c.text) for c, _ in results]
        sources = list(dict.fromkeys(label for label, _ in labeled))  # unique, ordered
        return window, labeled, sources, summary

    def _commit(self, question: str, answer: str) -> None:
        """Persist a completed turn; fold older turns into the running summary."""
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        if self.meta.get("title") == "New chat":
            self.meta["title"] = question[:48]
        self._maybe_summarize()
        self.save()

    def _maybe_summarize(self) -> None:
        """Fold turns older than the recent window into a compact running summary.

        Keeps LLM context bounded (summary + recent window) while remembering an
        arbitrarily long chat — the summary-buffer pattern.
        """
        window = self.cfg.rag.max_history_messages
        start = self.meta.get("summarized_upto", 0)
        end = len(self.history) - window  # messages older than the window
        if end - start < 4:  # only summarize in batches to limit LLM calls
            return
        older = self.history[start:end]
        self.meta["summary"] = (
            get_llm(self.cfg)
            .chat(
                prompts.update_summary_messages(self.meta.get("summary", ""), older),
                max_tokens=320,
            )
            .strip()
        )
        self.meta["summarized_upto"] = end

    def ask(self, question: str, top_k: int | None = None) -> dict:
        """Answer a question (non-streaming). Returns ``{answer, sources, contexts}``."""
        prep = self._prepare(question, top_k)
        if prep is None:
            return {"answer": self._NO_DOCS, "sources": [], "contexts": []}
        window, labeled, sources, summary = prep
        answer = get_llm(self.cfg).chat(prompts.qa_messages(question, labeled, window, summary))
        self._commit(question, answer)
        return {"answer": answer, "sources": sources, "contexts": [t for _, t in labeled]}

    def ask_stream(self, question: str, top_k: int | None = None):
        """Stream an answer token-by-token for a responsive UI.

        Yields ``("meta", sources)`` first, then ``("delta", text)`` chunks. Persists
        the full turn once streaming finishes.
        """
        prep = self._prepare(question, top_k)
        if prep is None:
            yield ("meta", [])
            yield ("delta", self._NO_DOCS)
            return
        window, labeled, sources, summary = prep
        yield ("meta", sources)
        pieces: list[str] = []
        for piece in get_llm(self.cfg).chat(
            prompts.qa_messages(question, labeled, window, summary), stream=True
        ):
            pieces.append(piece)
            yield ("delta", piece)
        self._commit(question, "".join(pieces))

    def reset(self) -> None:
        """Clear this chat's conversation history + summary (keeps documents/index)."""
        self.history = []
        self.meta["summary"] = ""
        self.meta["summarized_upto"] = 0
        self.save()


# --- session management ------------------------------------------------------


def list_sessions(cfg: AppConfig | None = None) -> list[dict]:
    """Return summaries of all saved chats, newest first."""
    cfg = cfg or CONFIG
    root = _sessions_root(cfg)
    out = [ChatSession(p.name, cfg).summary() for p in root.iterdir() if p.is_dir()]
    return sorted(out, key=lambda s: s["created"], reverse=True)


def create_session(cfg: AppConfig | None = None) -> ChatSession:
    """Create and return a fresh empty chat."""
    return ChatSession(uuid.uuid4().hex[:12], cfg or CONFIG)


def get_session(session_id: str, cfg: AppConfig | None = None) -> ChatSession:
    """Load an existing chat by id (creates the dir if missing)."""
    return ChatSession(session_id, cfg or CONFIG)


def delete_session(session_id: str, cfg: AppConfig | None = None) -> None:
    """Delete a chat and all its data."""
    import shutil

    cfg = cfg or CONFIG
    target = _sessions_root(cfg) / session_id
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
