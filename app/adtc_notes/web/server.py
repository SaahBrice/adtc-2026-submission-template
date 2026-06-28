"""adtc_notes/web/server.py — Offline FastAPI server for the document assistant.

Serves a single self-contained HTML page (no external CDNs — fully offline) and a
small JSON API wrapping the two pipelines:

* POST /api/digitize  — image upload → formatted, downloadable document.
* POST /api/documents — index uploaded documents for Q&A.
* POST /api/ask       — grounded question answering over the index.

Heavy backends (LLM, embedder) load lazily on first use, so the server starts
instantly and the page is usable for status checks even before models are present.

Launch with: ``python -m adtc_notes serve`` (see cli.py).
"""

# NOTE: deliberately NOT using ``from __future__ import annotations`` — FastAPI
# needs real type objects (UploadFile, Form, …) at decoration time, not strings.

from pathlib import Path
from typing import List

from ..config import CONFIG, OUTPUT_DIR
from ..errors import ADTCError

_STATIC = Path(__file__).parent / "static"
_UPLOADS = None  # set on app creation


def _save_upload(upload, dest_dir: Path) -> Path:
    """Persist a Starlette ``UploadFile`` to ``dest_dir`` and return its path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(upload.filename).name
    with dest.open("wb") as fh:
        fh.write(upload.file.read())
    return dest


def create_app():
    """Build and return the FastAPI application.

    Imported lazily so the package does not hard-depend on FastAPI/uvicorn.

    Raises:
        BackendNotInstalledError: If FastAPI is not installed.
    """
    try:
        from fastapi import FastAPI, UploadFile, File, Form, Body
        from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        from ..errors import BackendNotInstalledError

        raise BackendNotInstalledError(
            "Web UI deps not installed: pip install fastapi uvicorn python-multipart"
        ) from exc

    global _UPLOADS
    CONFIG.ensure_dirs()
    _UPLOADS = CONFIG.rag.index_dir.parent / "uploads"
    _UPLOADS.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="adtc_notes", docs_url=None, redoc_url=None)

    # Lazily-created retriever shared across requests (index persists to disk).
    state: dict = {"retriever": None}

    def retriever():
        from ..rag import Retriever

        if state["retriever"] is None:
            state["retriever"] = Retriever(CONFIG)
        return state["retriever"]

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/status")
    def status():
        r = retriever()
        return {
            "llm_present": CONFIG.llm.model_path.exists(),
            "embed_present": CONFIG.embedding.model_path.exists(),
            "indexed_chunks": len(r.store),
        }

    @app.post("/api/digitize")
    def digitize(file: UploadFile = File(...), fmt: str = Form("md")):
        from ..pipeline import digitize_to_document

        src = _save_upload(file, _UPLOADS)
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

    @app.post("/api/documents")
    def documents(files: List[UploadFile] = File(...)):
        paths = [_save_upload(f, _UPLOADS) for f in files]
        try:
            added = retriever().add_documents(paths)
        except ADTCError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse({"added": added, "indexed_chunks": len(retriever().store)})

    @app.post("/api/ask")
    def ask(payload: dict = Body(...)):
        question = (payload or {}).get("question", "").strip()
        if not question:
            return JSONResponse({"error": "empty question"}, status_code=400)
        try:
            out = retriever().ask(question)
        except ADTCError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(out)

    @app.get("/download/{name}")
    def download(name: str):
        path = OUTPUT_DIR / Path(name).name  # prevent path traversal
        if not path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(str(path), filename=path.name)

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
