# Technical Report — adtc_notes: Offline Document Intelligence for African SMEs

**Team ID:** TODO-REGISTER-ON-ADTF-PORTAL  
**Domain:** corporate_enterprise  
**Model:** Phi-3.5-mini-instruct-Q4_K_M (GGUF, llama.cpp)

> ⚠️ This report is a living draft. Benchmark numbers below are **placeholders to be
> filled from the ADTC profiler** on target-class hardware. Model selection is
> provisional pending profiler results (see "Design Decisions").

---

## Problem

Small and medium enterprises across Africa run on documents — invoices, contracts,
receipts, meeting notes, reports — much of it on paper or in scattered files. The
people who need answers from these documents (a shop owner in Dakar, an office
administrator in Arusha) typically work on **low-cost 8 GB laptops with unreliable
internet and electricity**. Cloud AI tools are blocked by API fees, connectivity,
and data-privacy concerns over sensitive business records.

**adtc_notes** is a fully **offline** assistant that does two things on commodity
hardware:

1. **Digitize** — turn a photo/scan of a document into a clean, **downloadable**
   Word/PDF/Markdown file (with formulas and figures preserved).
2. **Ask** — index a local library of documents and answer natural-language
   questions with **grounded, cited** responses (retrieval-augmented generation).

The target user keeps full control of their data (nothing leaves the device) and
gets useful productivity gains (summarizing, drafting, document Q&A) without
internet — directly addressing the **Corporate/Enterprise** domain and the
**African Use Case** bonus.

---

## Design Decisions

- **Runtime:** `llama.cpp` (via `llama-cpp-python`), as required. The *entire* AI
  stack — chat **and** embeddings — runs on llama.cpp/ggml, so the app needs **no
  PyTorch** at its core. This keeps the dependency footprint and RAM small.
- **Base model (provisional):** **Phi-3.5-mini-instruct (3.8B)**, MIT-licensed,
  strong at summarization/drafting/analysis — the core Corporate/Enterprise tasks.
- **Quantization:** **Q4_K_M** — the established speed/quality/RAM sweet spot on
  memory-constrained CPUs (~2.4 GB weights).
- **Embeddings:** `bge-small-en-v1.5` GGUF (~25 MB) for offline RAG, loaded only
  during ingestion/query so it does not occupy RAM during chat inference.
- **Retrieval:** a dependency-light NumPy cosine index (brute force is ample for
  SME-scale corpora; swappable for FAISS later).
- **Document rendering:** Markdown always; DOCX via `python-docx`; PDF via Pandoc
  if present, else an `fpdf2` fallback. Formulas render to images with matplotlib
  **mathtext** — no system TeX install required.
- **OCR:** Tesseract (lightest reliable CPU engine). Optional handwritten/printed
  **formula → LaTeX** via pix2tex is fully isolated (it pulls PyTorch), so it is an
  opt-in "added advantage", not a core dependency.

### Alternatives considered

| Option | Verdict |
|---|---|
| Qwen2.5-3B-Instruct | Strong, but 3B variant carries a non-Apache "Qwen" license — chose MIT Phi-3.5-mini to avoid ambiguity. **To re-benchmark.** |
| Llama-3.2-3B-Instruct | Good; community license. Candidate fallback if Phi is too slow on CPU. |
| Qwen2.5-1.5B (Apache) / SmolLM2-1.7B | Lighter/faster, higher efficiency score, but weaker analysis quality. Fallback if RAM/throughput require it. |
| sentence-transformers embeddings | Rejected: pulls PyTorch. GGUF embeddings keep the stack unified and light. |
| FAISS vector store | Deferred: unnecessary at SME scale; NumPy avoids an extra dependency. |
| TeX Live for PDF math | Rejected as default: too heavy for the target; matplotlib mathtext covers it. |

> **Model selection is not final.** We will profile Phi-3.5-mini, Llama-3.2-3B, and
> a 1.5B option on the ADTC profiler and pick the best **accuracy ÷ (RAM, latency)**
> trade-off, since efficiency (20%) and throughput (30%) both reward smaller models.

---

## Constraints

- **Hardware:** ADTC Standard Laptop — 4 vCPU, **8 GB RAM**, integrated GPU only,
  Ubuntu 22.04. CPU-only inference.
- **Memory:** efficiency is scored against a 7 GB budget; **OOM = disqualification**.
  Levers used: Q4_K_M weights, modest `n_ctx` (4096), mmap (no mlock by default),
  and never co-resident embedding + chat models.
- **Thermal:** must stay ≤ 85 °C (−10 penalty). Thread count capped at 4; to be
  validated under sustained load with `lm-sensors`.
- **Connectivity:** **zero network calls at runtime.** Weights download once via
  `download_model.sh`; everything else is local.
- **Data:** sensitive business documents never leave the device.

---

## Benchmarks

> Placeholders — to be replaced with real `adtc-profiler` output on target hardware.

| Metric | Value |
|---|---|
| Machine | TBD (dev) / ADTC Standard Laptop (official) |
| Model | Phi-3.5-mini-instruct-Q4_K_M |
| RAM at peak | _TBD_ |
| Time to first token | _TBD_ |
| Generation speed (t/s) | _TBD_ |
| Thermal throttling | _TBD_ |

Reproduce locally:

```bash
pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"
bash download_model.sh
adtc-profiler run --submission . --mode participant --output submission.json --skip-accuracy
```

Official scores are measured by the ADTC profiler on the standard evaluation machine.

---

## Application

The product built around the model lives in [`app/`](app/) (see its README). It
demonstrates the Corporate/Enterprise use case end-to-end: digitize → format →
download, and index → ask. The same scored GGUF model powers both flows.
