"""adtc_notes/rag/ingest.py — Load local documents into plain text.

Supports the formats an SME actually has on disk: .txt, .md, .pdf, .docx, and
images (.png/.jpg/...) which are routed through the OCR pipeline. Each backend is
imported lazily so missing optional deps only fail for the formats that need them.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import BackendNotInstalledError, UnsupportedFileError

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_text(path: str | Path) -> str:
    """Extract plain text from a document, dispatching on file extension.

    Args:
        path: Path to a .txt/.md/.pdf/.docx file or an image.

    Returns:
        Extracted UTF-8 text (may be empty).

    Raises:
        UnsupportedFileError: For unknown extensions.
        FileNotFoundError: If the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()

    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    if suffix in IMAGE_SUFFIXES:
        # Images become text via OCR (imported here to avoid a hard dependency cycle).
        from ..ocr.pipeline import ocr_image_to_text

        return ocr_image_to_text(path)
    raise UnsupportedFileError(f"Unsupported file type: {suffix} ({path.name})")


def _load_pdf(path: Path) -> str:
    """Extract text from a PDF using pypdf (lazy import)."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise BackendNotInstalledError("pypdf not installed: pip install pypdf") from exc
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


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
