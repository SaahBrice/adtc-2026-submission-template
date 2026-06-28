# adtc_notes — Offline Document Intelligence (ADTC 2026, Corporate/Enterprise)

A fully **offline**, **CPU-only** document assistant for the ADTC Standard Laptop
(4 vCPU, 8 GB RAM, integrated GPU, Ubuntu 22.04). Two capabilities, one local
GGUF model run through `llama.cpp`:

- **Digitize (PATH A):** snap/upload a photo or scan → clean, **downloadable**
  document (Markdown / DOCX / PDF) with typeset formulas and figures.
- **Ask (PATH B / RAG):** index many local documents (PDF, Word, text, images)
  → ask questions answered with grounded, cited context.

> The whole AI stack (chat **and** embeddings) runs on llama.cpp — **no PyTorch**
> in the core, to respect the 8 GB budget. The only PyTorch-dependent feature is
> optional formula OCR (pix2tex), fully isolated in `adtc_notes/ocr/formula.py`.

---

## Architecture

```
adtc_notes/
├── config.py          # all tunables (RAM/speed levers), env-overridable
├── errors.py          # explicit exception types
├── llm/               # GGUF chat model wrapper + prompt templates
├── ocr/               # preprocess → Tesseract text OCR → (optional) formula OCR
├── rag/               # ingest → chunk → embed (GGUF) → NumPy vector store → retrieve
├── render/            # Markdown / DOCX / PDF writers + LaTeX→PNG (matplotlib)
├── pipeline.py        # PATH A orchestration (image → formatted document)
└── cli.py             # command-line interface
```

See `../docs/PROJECT_VISION_AND_ARCHITECTURE.md` for the full design and rationale.

---

## Setup

### 1. Python deps (small, ~quick)

```bash
cd app
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. System dependencies (offline tools)

- **Tesseract OCR** (optional — printed-text fallback only; the default digitize
  engine is the vision model below):
  - Ubuntu: `sudo apt install tesseract-ocr`
  - Windows: `winget install UB-Mannheim.TesseractOCR`
- **Pandoc + LaTeX** (best-quality PDF; optional — app falls back to fpdf2):
  - Ubuntu: `sudo apt install pandoc texlive-latex-recommended`

### 3. Models (downloaded by the submission script)

From the **repo root** (one level up):

```bash
bash download_model.sh
```

This fetches all GGUF weights into `model/` (~5.7 GB total):
- **chat model** (~2.4 GB) — the ADTC-benchmarked text model (RAG Q&A)
- **embedding model** (~25 MB) — local RAG embeddings
- **vision model + mmproj** (~3.3 GB) — Qwen2.5-VL for digitizing handwriting/images

If your connection is slow, run the script yourself and let it finish first. The
heavy models are never co-resident: the vision model loads alone at digitize time
and the chat model loads for Q&A, so peak RAM stays within the 8 GB budget.

Digitize engine selection: `ADTC_OCR_ENGINE=auto|vlm|tesseract` (default `auto` —
uses the vision model when present). Tune VLM speed/accuracy with `ADTC_VLM_MAX_SIDE`
(default 1280).

### 4. (Optional) pix2tex formula OCR

```bash
pip install -r requirements-optional.txt   # pulls in PyTorch (CPU); heavier
```

---

## Usage (CLI)

```bash
# Show config + whether models are present
python -m adtc_notes info

# Digitize an image into a downloadable Word document
python -m adtc_notes digitize path/to/photo.jpg --format docx

# Index documents, then ask questions across all of them
python -m adtc_notes add report1.pdf notes.docx scan.png
python -m adtc_notes ask "What were the Q3 action items?"
```

Outputs are written to `app/output/`; the RAG index lives in `app/data/index/`
(both are gitignored).

---

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest          # pure-core + renderer tests (no models needed)
black . && ruff check .   # format + lint
```

Code style: Google-style docstrings, type hints, lightweight file headers — see
`../docs/CODING_STANDARDS.md`.

---

## Offline guarantee

At runtime the app makes **zero network calls**. Models are loaded from local disk
via llama.cpp; OCR and rendering are local. Network is used only once, at setup, to
download dependencies and weights.
