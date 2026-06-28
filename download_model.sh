#!/usr/bin/env bash
# Download model weights for the ADTC 2026 submission.
#
# Rules (per the template):
#   - Idempotent (safe to run multiple times).
#   - No credentials required (public URLs only).
#   - The scored model's output path must match `_runtime.model_path` in metadata.json.
#
# This script fetches TWO GGUF files:
#   1. The chat model  — the ADTC-scored artifact (run by the profiler via llama.cpp).
#   2. The embedder    — small model used by the app's offline RAG (not benchmarked).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"
mkdir -p "$MODEL_DIR"

# ── Chat model (scored): Phi-3.5-mini-instruct, Q4_K_M, ~2.39 GB, MIT license ──
CHAT_FILE="$MODEL_DIR/Phi-3.5-mini-instruct-Q4_K_M.gguf"
CHAT_URL="https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf"

# ── Embedding model (RAG): bge-small-en-v1.5, Q4_K_M, ~25 MB, MIT license ──────
EMBED_FILE="$MODEL_DIR/bge-small-en-v1.5-q4_k_m.gguf"
EMBED_URL="https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q4_k_m.gguf"

# ── Vision model (digitize/OCR): Qwen2.5-VL-3B, Q4_K_M + mmproj, ~3.3 GB, Apache-2.0 ──
# Reads handwriting/printed pages + formulas directly to Markdown. App-side only
# (not the benchmarked model); loaded alone at digitize time to respect 8 GB RAM.
VLM_FILE="$MODEL_DIR/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
VLM_URL="https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
VLM_MMPROJ_FILE="$MODEL_DIR/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
VLM_MMPROJ_URL="https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"

# Download $2 → $1 only if missing. Uses curl or wget, whichever is available.
download() {
  local dest="$1" url="$2" label="$3"
  if [[ -f "$dest" ]]; then
    echo "[skip] $label already present at $dest"
    return 0
  fi
  echo "[download] $label"
  echo "  from: $url"
  echo "  to:   $dest"
  if command -v curl > /dev/null 2>&1; then
    curl -L --fail --progress-bar -o "$dest.partial" "$url"
  elif command -v wget > /dev/null 2>&1; then
    wget --show-progress -O "$dest.partial" "$url"
  else
    echo "error: neither curl nor wget found" >&2
    exit 1
  fi
  mv "$dest.partial" "$dest"
  echo "[done] $dest"
}

download "$CHAT_FILE"        "$CHAT_URL"        "chat model (Phi-3.5-mini Q4_K_M, ~2.39 GB)"
download "$EMBED_FILE"       "$EMBED_URL"       "embedding model (bge-small-en-v1.5, ~25 MB)"
download "$VLM_FILE"         "$VLM_URL"         "vision model (Qwen2.5-VL-3B Q4_K_M, ~1.93 GB)"
download "$VLM_MMPROJ_FILE"  "$VLM_MMPROJ_URL"  "vision projector (mmproj f16, ~1.34 GB)"

echo "all model weights ready in $MODEL_DIR"
