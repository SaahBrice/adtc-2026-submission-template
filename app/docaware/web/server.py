"""docaware/web/server.py — Offline FastAPI server for the document assistant.

Serves one self-contained offline page plus a small JSON API:

* /api/digitize                      — image → formatted, downloadable document.
* /api/sessions (+ /{id}/…)          — multi-chat: each chat has its own documents,
                                       conversation memory, and page-cited answers.

Chats are persisted on disk (see rag/session.py), so they survive restarts and a
request simply loads the session it needs. Heavy models load lazily and are managed
to stay within the 8 GB budget. Launch with: ``python -m docaware serve``.
"""

# NOTE: deliberately NOT using ``from __future__ import annotations`` — FastAPI
# needs real type objects (UploadFile, Form, …) at decoration time, not strings.

from pathlib import Path
from typing import List

from ..config import CONFIG, OUTPUT_DIR
from ..errors import ADTCError

_STATIC = Path(__file__).parent / "static"


def _save_upload(upload, dest_dir: Path) -> Path:
    """Persist a Starlette ``UploadFile`` to ``dest_dir`` and return its path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(upload.filename).name
    with dest.open("wb") as fh:
        fh.write(upload.file.read())
    return dest


def create_app():
    """Build and return the FastAPI application (FastAPI imported lazily)."""
    try:
        from fastapi import FastAPI, UploadFile, File, Form, Body
        from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        from ..errors import BackendNotInstalledError

        raise BackendNotInstalledError(
            "Web UI deps not installed: pip install fastapi uvicorn python-multipart"
        ) from exc

    CONFIG.ensure_dirs()
    uploads = CONFIG.rag.sessions_dir.parent / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="docaware", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/status")
    def status():
        from ..ocr.pipeline import resolve_engine

        return {
            "llm_present": CONFIG.llm.model_path.exists(),
            "embed_present": CONFIG.embedding.model_path.exists(),
            "vision_present": CONFIG.vision.model_path.exists()
            and CONFIG.vision.mmproj_path.exists(),
            "engine": resolve_engine(CONFIG),
        }

    # --- digitize (not chat-scoped) -----------------------------------------

    @app.post("/api/digitize")
    def digitize(file: UploadFile = File(...), fmt: str = Form("md")):
        from ..pipeline import digitize_to_document

        src = _save_upload(file, uploads)
        try:
            result = digitize_to_document(src, fmt=fmt)
        except ADTCError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(
            {
                "markdown": result.markdown,
                "download": f"/download/{result.output_path.name}",
                "warnings": result.warnings,
            }
        )

    @app.get("/download/{name}")
    def download(name: str):
        path = OUTPUT_DIR / Path(name).name  # prevent path traversal
        if not path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(str(path), filename=path.name)

    # --- chats (sessions) ----------------------------------------------------

    @app.get("/api/sessions")
    def sessions_list():
        from ..rag import list_sessions

        return JSONResponse(list_sessions(CONFIG))

    @app.post("/api/sessions")
    def sessions_create():
        from ..rag import create_session

        s = create_session(CONFIG)
        s.save()
        return JSONResponse(s.summary())

    @app.get("/api/sessions/{sid}")
    def session_get(sid: str):
        from ..rag import get_session

        s = get_session(sid, CONFIG)
        return JSONResponse({**s.summary(), "history": s.history})

    @app.post("/api/sessions/{sid}/documents")
    def session_documents(sid: str, files: List[UploadFile] = File(...)):
        from ..rag import get_session

        s = get_session(sid, CONFIG)
        paths = [_save_upload(f, uploads) for f in files]
        try:
            added = s.add_documents(paths)
        except ADTCError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse({"added": added, **s.summary()})

    @app.post("/api/sessions/{sid}/ask")
    def session_ask(sid: str, payload: dict = Body(...)):
        from ..rag import get_session

        question = (payload or {}).get("question", "").strip()
        if not question:
            return JSONResponse({"error": "empty question"}, status_code=400)
        s = get_session(sid, CONFIG)
        try:
            out = s.ask(question)
        except ADTCError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse({**out, "title": s.title})

    @app.post("/api/sessions/{sid}/reset")
    def session_reset(sid: str):
        from ..rag import get_session

        get_session(sid, CONFIG).reset()
        return JSONResponse({"ok": True})

    @app.post("/api/sessions/{sid}/delete")
    def session_delete(sid: str):
        from ..rag import delete_session

        delete_session(sid, CONFIG)
        return JSONResponse({"ok": True})

    if _STATIC.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
    return app


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the server with uvicorn (blocking)."""
    try:
        import uvicorn
    except ImportError as exc:
        from ..errors import BackendNotInstalledError

        raise BackendNotInstalledError("uvicorn not installed: pip install uvicorn") from exc
    uvicorn.run(create_app(), host=host, port=port)
