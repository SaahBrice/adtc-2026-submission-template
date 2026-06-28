"""Tests for the web server wiring (no models required for these endpoints)."""

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("httpx") is None,
    reason="fastapi + httpx (TestClient) required",
)


def _client():
    from fastapi.testclient import TestClient

    from adtc_notes.web.server import create_app

    return TestClient(create_app())


def test_index_page_served():
    resp = _client().get("/")
    assert resp.status_code == 200
    assert "adtc_notes" in resp.text


def test_status_reports_model_presence():
    resp = _client().get("/api/status")
    assert resp.status_code == 200
    body = resp.json()
    assert {"llm_present", "embed_present", "vision_present", "engine", "indexed_chunks"} <= set(
        body
    )
    assert isinstance(body["indexed_chunks"], int)
    assert body["engine"] in {"vlm", "tesseract"}


def test_ask_empty_question_rejected():
    resp = _client().post("/api/ask", json={"question": "   "})
    assert resp.status_code == 400


def test_download_missing_file_404():
    resp = _client().get("/download/nope.pdf")
    assert resp.status_code == 404
