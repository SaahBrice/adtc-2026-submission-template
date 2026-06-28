# Technical Report — Docaware: Offline Document Intelligence for African SMEs

**Team ID:** TODO-REGISTER-ON-ADTF-PORTAL  
**Domain:** corporate_enterprise  
**Model:** Qwen2.5-1.5B-Instruct-Q4_K_M (GGUF, llama.cpp)

> Note: benchmark numbers below are **dev-machine measurements** (Intel i7-8700, 4
> threads) used to select the model; official figures come from the ADTC profiler on
> the 8 GB Ubuntu target.

---

## Problem

Small and medium enterprises across Africa run on documents — invoices, contracts,
receipts, meeting notes, reports — much of it on paper or in scattered files. The
people who need answers from these documents (a shop owner in Dakar, an office
administrator in Arusha) typically work on **low-cost 8 GB laptops with unreliable
internet and electricity**. Cloud AI tools are blocked by API fees, connectivity,
and data-privacy concerns over sensitive business records.

**Docaware** is a fully **offline** assistant that does two things on commodity
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
- **Base model:** **Qwen2.5-1.5B-Instruct**, Apache-2.0, strong instruction-following
  for summarization/drafting/QA. Chosen by benchmark (below): on a 4-core CPU it hits
  ~15 t/s at ~1.76 GB — far better throughput and memory than larger models, while RAG
  grounding keeps answers accurate. This directly maximizes Sperf (30%) and Seff (20%).
- **Quantization:** **Q4_K_M** — the established speed/quality/RAM sweet spot on
  memory-constrained CPUs (~0.99 GB weights).
- **Inference tuning:** flash-attention + **q8_0 KV cache** + `n_ctx=2048` + 4 threads —
  measured faster and lower-RAM than defaults; KV quantization roughly halves KV memory.
- **Embeddings:** `bge-small-en-v1.5` GGUF (~25 MB) for offline RAG, loaded only
  during ingestion/query so it does not occupy RAM during chat inference.
- **Retrieval:** a dependency-light NumPy cosine index (brute force is ample for
  SME-scale corpora; swappable for FAISS later).
- **Document rendering:** Markdown always; DOCX via `python-docx`; PDF via Pandoc
  if present, else an `fpdf2` fallback. Formulas render to images with matplotlib
  **mathtext** — no system TeX install required.
- **Digitize / OCR:** the primary engine is **Qwen2.5-VL-3B** (GGUF + mmproj) run
  through llama.cpp — a vision-language model that reads handwriting, layout, and
  math directly into clean Markdown with LaTeX. Tesseract was evaluated and rejected
  for handwriting (unusable on cursive); it remains a fast fallback for printed text.
  The VLM is app-side only (not the ADTC-benchmarked model) and is loaded alone at
  digitize time via a single-active-model manager so peak RAM stays within 8 GB.
  Resolution is the main latency lever (`VLM_MAX_SIDE`, default 1280 for accuracy).

### Alternatives considered (measured on dev, 4 threads, flash-attn + q8 KV)

| Option | t/s | Peak RSS | Verdict |
|---|---|---|---|
| **Qwen2.5-1.5B-Instruct** (chosen) | **14.7** | **1.76 GB** | Best balance: near-reference throughput, low RAM, accurate via RAG. Apache-2.0. |
| Phi-3.5-mini (3.8B) | 6.2 | 4.15 GB | Rejected: too slow (Sperf) and too heavy (Seff ~41, near-OOM). |
| Qwen2.5-0.5B | 31.5 | 0.55 GB | Fastest/lightest, but weaker reasoning — kept as an extreme-efficiency option. |
| sentence-transformers embeddings | — | — | Rejected: pulls PyTorch. GGUF embeddings keep the stack unified and light. |
| FAISS vector store | — | — | Deferred: unnecessary at SME scale; NumPy avoids a dependency. |
| TeX Live for PDF math | — | — | Rejected as default: too heavy; matplotlib mathtext covers it. |

> Final confirmation comes from the ADTC profiler on the 8 GB Ubuntu target; if more
> accuracy is needed we will fine-tune the 1.5B on Corporate/Enterprise data (the
> challenge's GPU credits exist for exactly this).

---

## Constraints

- **Hardware:** ADTC Standard Laptop — 4 vCPU, **8 GB RAM**, integrated GPU only,
  Ubuntu 22.04. CPU-only inference.
- **Memory:** efficiency is scored against a 7 GB budget; **OOM = disqualification**.
  Levers used: a 1.5B Q4_K_M model (~1.76 GB peak), `n_ctx=2048`, **q8_0 KV cache**,
  flash-attention, mmap (no mlock), and a single-active-model manager so the digitize
  VLM and chat model are never co-resident.
- **Thermal:** must stay ≤ 85 °C (−10 penalty). Thread count capped at 4; to be
  validated under sustained load with `lm-sensors`.
- **Connectivity:** **zero network calls at runtime.** Weights download once via
  `download_model.sh`; everything else is local.
- **Data:** sensitive business documents never leave the device.

---

## Benchmarks

Measured with the **official `adtc-profiler`** (participant mode), pinned to **4 cores**
to mimic the ADTC Standard Laptop (full Linux profiler run on the target laptop pending).

| Profiler metric | Qwen2.5-1.5B-Instruct-Q4_K_M | Score impact |
|---|---|---|
| `tokens_per_second_generation` | **17.03 t/s** | vs 15.0 reference → **Sperf ≈ 100** (capped) |
| `memory.peak_rss_mb` | **1716 MB (1.72 GB)** | `Seff = 100·(7−1.72)/7` ≈ **75** |
| `cpu_thermal.throttled` | **false** | **Pthermal = 0** |
| `cpu_thermal.cpu_percent_p99` | 55.7% | headroom on 4 cores |
| `model_info.params_match` | **true** | claimed 1.5B == GGUF (fraud check passes) |
| `model_info.architecture` | qwen2 | valid GGUF via llama.cpp |

Raw `llama-bench` (4 threads) confirms: **pp512 = 66.6 t/s, tg128 = 18.6 t/s**. App-side RAG
answers **stream** (first tokens ~1–2 s warm). The previously-provisional Phi-3.5-mini (3.8B)
measured 6.2 t/s / 4.15 GB — the switch to 1.5B is ~2.7× throughput and ~2.4× less RAM.

Digitize (Qwen2.5-VL, app-side, not the scored model): see "Design Decisions"; an OCR-
specialized engine is under evaluation to cut page latency.

Reproduce:

```bash
pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"   # needs llama-bench on PATH
bash download_model.sh
adtc-profiler run --submission . --mode participant --output submission.json --skip-accuracy
```

`measured_on` is `participant_laptop`; official scores come from the profiler on the
standard evaluation machine. (Accuracy/`Sacc` requires the hidden validation set.)

---

## Application

The product built around the model lives in [`app/`](app/) (see its README). It
demonstrates the Corporate/Enterprise use case end-to-end: digitize → format →
download, and index → ask. The same scored GGUF model powers both flows.
