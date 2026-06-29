"""Tests for the web server wiring (no models required for these endpoints)."""

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("httpx") is None,
    reason="fastapi + httpx (TestClient) required",
)


def _client():
    from fastapi.testclient import TestClient

    from docaware.web.server import create_app

    return TestClient(create_app())


@pytest.fixture
def isolated_sessions(tmp_path, monkeypatch):
    """Point session storage at a temp dir so tests don't touch real chats."""
    from docaware.config import CONFIG

    monkeypatch.setattr(CONFIG.rag, "sessions_dir", tmp_path / "sessions")
    return tmp_path


def test_index_page_served():
    resp = _client().get("/")
    assert resp.status_code == 200
    assert "Docaware" in resp.text


def test_status_reports_model_presence():
    body = _client().get("/api/status").json()
    assert {"llm_present", "embed_present", "vision_present"} <= set(body)
    assert isinstance(body["vision_present"], bool)


def test_create_and_list_session(isolated_sessions):
    c = _client()
    created = c.post("/api/sessions").json()
    assert created["id"] and created["title"] == "New chat"
    listed = c.get("/api/sessions").json()
    assert any(s["id"] == created["id"] for s in listed)


def test_ask_empty_question_rejected(isolated_sessions):
    c = _client()
    sid = c.post("/api/sessions").json()["id"]
    assert c.post(f"/api/sessions/{sid}/ask", json={"question": "  "}).status_code == 400


def test_ask_with_no_documents_is_graceful(isolated_sessions):
    c = _client()
    sid = c.post("/api/sessions").json()["id"]
    out = c.post(f"/api/sessions/{sid}/ask", json={"question": "hello?"}).json()
    assert "No documents" in out["answer"]  # no model load needed when index is empty


def test_download_missing_file_404():
    assert _client().get("/download/nope.pdf").status_code == 404
