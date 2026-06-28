"""Tests for adtc_notes.llm.prompts (pure functions)."""

from adtc_notes.llm import prompts


def test_format_document_messages_structure():
    msgs = prompts.format_document_messages("raw ocr text")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "raw ocr text" in msgs[1]["content"]


def test_qa_messages_numbers_contexts():
    msgs = prompts.qa_messages("What is X?", ["first ctx", "second ctx"])
    user = msgs[1]["content"]
    assert "[1] first ctx" in user
    assert "[2] second ctx" in user
    assert "What is X?" in user


def test_qa_system_demands_grounding():
    msgs = prompts.qa_messages("q", ["c"])
    assert "ONLY" in msgs[0]["content"]


def test_summary_messages_structure():
    msgs = prompts.summary_messages("a long document")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == "a long document"
