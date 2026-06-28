"""docaware — Offline document-intelligence app for ADTC 2026 (Corporate/Enterprise).

Two capabilities, one local GGUF model (run via llama.cpp), 100% offline:

* PATH A — Digitize: image/photo of notes or documents → clean, downloadable
  formatted document (Markdown / DOCX / PDF) with typeset formulas and figures.
* PATH B — Ask (RAG): ingest many local documents → ask questions answered with
  retrieved, grounded context.

Design constraints (see ../../docs/): runs on the ADTC Standard Laptop
(4 vCPU, 8 GB RAM, integrated GPU only, Ubuntu 22.04), CPU-only, no network at runtime.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
