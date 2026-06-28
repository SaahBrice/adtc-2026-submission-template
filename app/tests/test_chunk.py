"""Tests for docaware.rag.chunk (pure, no heavy deps)."""

from docaware.rag.chunk import chunk_text


def test_empty_input_returns_no_chunks():
    assert chunk_text("", "doc") == []
    assert chunk_text("   \n\n  ", "doc") == []


def test_short_text_is_one_chunk():
    chunks = chunk_text("Hello world.", "doc")
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert chunks[0].source == "doc"
    assert chunks[0].index == 0


def test_paragraphs_split_within_budget():
    text = "\n\n".join(f"Paragraph number {i} with some words." for i in range(20))
    chunks = chunk_text(text, "doc", chunk_chars=120, overlap=20)
    assert len(chunks) > 1
    # Indices are sequential.
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # No chunk grossly exceeds the soft budget (allow overlap slack).
    assert all(len(c.text) <= 120 + 40 for c in chunks)


def test_oversized_paragraph_is_hard_split():
    big = "x" * 5000
    chunks = chunk_text(big, "doc", chunk_chars=1000, overlap=100)
    assert len(chunks) >= 5
    assert all(len(c.text) <= 1000 for c in chunks)


def test_overlap_guard_against_bad_config():
    # overlap >= chunk_chars must not crash or loop forever.
    chunks = chunk_text("a b c d e " * 50, "doc", chunk_chars=50, overlap=80)
    assert len(chunks) >= 1
