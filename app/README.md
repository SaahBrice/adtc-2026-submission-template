# Docaware — Offline Document Intelligence (ADTC 2026, Corporate/Enterprise)

A fully **offline**, **CPU-only** document assistant for the ADTC Standard Laptop
(4 vCPU, 8 GB RAM, integrated GPU, Ubuntu 22.04). Two capabilities, one local
GGUF model run through `llama.cpp`:

- **Digitize (PATH A):** snap/upload a photo or scan → clean, **downloadable**
  document (Markdown / DOCX / PDF) with typeset formulas and figures.
- **Ask (PATH B / RAG):** index many local documents (PDF, Word, text, images)
  → ask questions answered with grounded, cited context.

> The entire stack runs on **llama.cpp — no PyTorch at all**: the chat model and
> embeddings via `llama-cpp-python`, and OCR via the native `llama-mtmd-cli`
> (DeepSeek-OCR). Small footprint, fully offline, fits the 8 GB budget.

---

## Architecture

```
docaware/
├── config.py          # all tunables (RAM/speed levers), env-overridable
├── errors.py          # explicit exception types
├── llm/               # GGUF chat model wrapper + prompt templates
├── ocr/               # DeepSeek-OCR via native llama-mtmd-cli (image → Markdown)
├── rag/               # ingest → chunk → embed (GGUF) → NumPy store → sessions/memory
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

### 2. Native llama.cpp binaries (for OCR + benchmarking)

Digitize uses **DeepSeek-OCR** via `llama-mtmd-cli` (from llama.cpp). `download_model.sh`
auto-fetches the binaries into `bin/` on Linux/macOS. On Windows, download a build from
https://github.com/ggml-org/llama.cpp/releases and set `ADTC_MTMD_CLI` to `llama-mtmd-cli.exe`
(or put it on PATH). Optional: **Pandoc** for best-quality PDF (`sudo apt install pandoc`;
otherwise the app uses fpdf2).

### 3. Models (downloaded by the submission script)

From the **repo root** (one level up):

```bash
bash download_model.sh
```

Fetches into `model/` (~4.6 GB): the **chat model** (Qwen2.5-1.5B, ~1 GB — the ADTC-scored
text model), the **embedding model** (bge-small, ~25 MB), and the **OCR model** (DeepSeek-OCR
Q8 + mmproj, ~3.6 GB). The big models are never co-resident: digitize runs the OCR model in a
short-lived subprocess after evicting the chat/embedder, so peak RAM stays within 8 GB.

Tune OCR via env: `ADTC_MTMD_CLI` (binary path), `ADTC_VLM_THREADS`, `ADTC_VLM_PROMPT`.

---

## Usage (CLI)

```bash
# Show config + whether models are present
python -m docaware info

# Digitize an image into a downloadable Word document
python -m docaware digitize path/to/photo.jpg --format docx

# Index documents, then ask questions across all of them
python -m docaware add report1.pdf notes.docx scan.png
python -m docaware ask "What were the Q3 action items?"
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
