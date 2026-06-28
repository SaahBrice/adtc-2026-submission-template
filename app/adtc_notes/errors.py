"""adtc_notes/errors.py — Project-specific exceptions.

Narrow exception types make failures explicit (and easy to handle in the UI/CLI)
instead of leaking generic ImportError/FileNotFoundError from optional backends.
"""

from __future__ import annotations


class ADTCError(Exception):
    """Base class for all adtc_notes errors."""


class ModelNotFoundError(ADTCError):
    """A required GGUF weight file is missing. Run ``bash download_model.sh``."""


class BackendNotInstalledError(ADTCError):
    """An optional dependency (llama-cpp-python, tesseract, pix2tex, …) is absent."""


class UnsupportedFileError(ADTCError):
    """An uploaded document/image type is not handled by any ingester."""
