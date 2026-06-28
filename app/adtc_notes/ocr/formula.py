"""adtc_notes/ocr/formula.py — Optional math-formula → LaTeX recognition.

This is the "added advantage" feature. pix2tex (LaTeX-OCR) is optional and pulls
in PyTorch, so it is fully isolated here: if it is not installed, the rest of the
app keeps working and formulas are simply left as recognized text.

Constraint: pix2tex is heavier than the rest of the stack — load it on demand,
run it, and let it go out of scope so it does not sit in RAM during LLM inference.
"""

from __future__ import annotations

from functools import lru_cache

from ..errors import BackendNotInstalledError


def is_available() -> bool:
    """Return True if the pix2tex backend can be imported."""
    try:
        import pix2tex.cli  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _get_model():
    """Lazily construct the pix2tex model (cached for the process)."""
    try:
        from pix2tex.cli import LatexOCR  # type: ignore
    except ImportError as exc:
        raise BackendNotInstalledError(
            "pix2tex not installed (optional). Enable formula OCR with:\n"
            "  pip install 'pix2tex[gui]'\n"
            "Note: this pulls in PyTorch (CPU). See app/README.md."
        ) from exc
    return LatexOCR()


def image_to_latex(image) -> str:
    """Convert a cropped equation image to a LaTeX string.

    Args:
        image: A ``PIL.Image`` containing a single equation.

    Returns:
        LaTeX markup (without surrounding ``$``), or "" if nothing recognized.
    """
    model = _get_model()
    return (model(image) or "").strip()
