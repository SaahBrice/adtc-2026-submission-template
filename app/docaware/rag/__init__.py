"""docaware.rag — Offline conversational RAG over local document chats."""

from .chunk import Chunk, chunk_text
from .store import VectorStore
from .session import (
    ChatSession,
    create_session,
    delete_session,
    get_session,
    list_sessions,
)

__all__ = [
    "Chunk",
    "chunk_text",
    "VectorStore",
    "ChatSession",
    "create_session",
    "get_session",
    "list_sessions",
    "delete_session",
]
