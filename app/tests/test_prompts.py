"""Tests for docaware.llm.prompts (pure functions)."""

from docaware.llm import prompts


def test_qa_messages_labels_contexts_with_sources():
    msgs = prompts.qa_messages(
        "What is X?", [("report.pdf p.1", "first ctx"), ("report.pdf p.2", "second ctx")]
    )
    user = msgs[1]["content"]
    assert "[report.pdf p.1] first ctx" in user
    assert "[report.pdf p.2] second ctx" in user
    assert "What is X?" in user


def test_qa_messages_includes_history():
    history = [
        {"role": "user", "content": "earlier q"},
        {"role": "assistant", "content": "earlier a"},
    ]
    msgs = prompts.qa_messages("follow up", [("d p.1", "c")], history=history)
    assert msgs[1]["content"] == "earlier q"  # history spliced between system and final user
    assert msgs[-1]["content"].endswith("Question: follow up")


def test_qa_system_demands_grounding():
    msgs = prompts.qa_messages("q", [("d p.1", "c")])
    assert "ONLY" in msgs[0]["content"]


def test_condense_question_messages():
    history = [{"role": "user", "content": "Tell me about Q3"}]
    msgs = prompts.condense_question_messages("what about Q4?", history)
    assert msgs[0]["role"] == "system" and "standalone" in msgs[0]["content"].lower()
    assert "what about Q4?" in msgs[1]["content"]


def test_update_summary_messages():
    msgs = prompts.update_summary_messages(
        "prev summary", [{"role": "user", "content": "new thing"}]
    )
    assert msgs[0]["role"] == "system"
    assert "prev summary" in msgs[1]["content"] and "new thing" in msgs[1]["content"]
