"""docaware/rag/ingest.py — Load local documents into page-tagged text.

Returns text split by page where the format has pages (PDF), so chunks can carry
real page numbers for citations. Supports the formats an SME has on disk: .txt,
.md, .pdf, .docx, and images routed through the OCR pipeline. Backends are
imported lazily so a missing optional dep only fails for the format that needs it.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import BackendNotInstalledError, UnsupportedFileError

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# A page of extracted text: (page_number_or_None, text).
Page = tuple[int | None, str]


def load_pages(path: str | Path) -> list[Page]:
    """Extract text from a document as a list of ``(page_number, text)``.

    Page numbers are 1-based for PDFs; ``None`` for formats without stable pages
    (text, Markdown, DOCX) and ``1`` for single images.

    Raises:
        UnsupportedFileError: For unknown extensions.
        FileNotFoundError: If the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()

    if suffix in TEXT_SUFFIXES:
        return [(None, path.read_text(encoding="utf-8", errors="replace"))]
    if suffix == ".pdf":
        return _load_pdf_pages(path)
    if suffix == ".docx":
        return [(None, _load_docx(path))]
    if suffix in IMAGE_SUFFIXES:
        # Images become text/Markdown via the OCR engine. Lazy import avoids a cycle.
        from ..ocr.pipeline import image_to_markdown

        return [(1, image_to_markdown(path))]
    raise UnsupportedFileError(f"Unsupported file type: {suffix} ({path.name})")


def _load_pdf_pages(path: Path) -> list[Page]:
    """Extract a PDF as one ``(page_number, text)`` entry per page (pypdf)."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise BackendNotInstalledError("pypdf not installed: pip install pypdf") from exc
    reader = PdfReader(str(path))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def _load_docx(path: Path) -> str:
    """Extract text from a .docx using python-docx (lazy import)."""
    try:
        import docx  # type: ignore
    except ImportError as exc:
        raise BackendNotInstalledError(
            "python-docx not installed: pip install python-docx"
        ) from exc
    document = docx.Document(str(path))
    return "\n\n".join(p.text for p in document.paragraphs if p.text.strip())
