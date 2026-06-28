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

# Download $2 → $1 only if missing. Uses curl or wget, whichever is available.
download() {
  local dest="$1" url="$2" label="$3"
  if [[ -f "$dest" ]]; then
    echo "✓ $label already present at $dest — skipping"
    return 0
  fi
  echo "↓ downloading $label"
  echo "  $url"
  echo "  → $dest"
  if command -v curl > /dev/null 2>&1; then
    curl -L --fail --progress-bar -o "$dest.partial" "$url"
  elif command -v wget > /dev/null 2>&1; then
    wget --show-progress -O "$dest.partial" "$url"
  else
    echo "error: neither curl nor wget found" >&2
    exit 1
  fi
  mv "$dest.partial" "$dest"
  echo "✓ done: $dest"
}

download "$CHAT_FILE"  "$CHAT_URL"  "chat model (Phi-3.5-mini Q4_K_M, ~2.39 GB)"
download "$EMBED_FILE" "$EMBED_URL" "embedding model (bge-small-en-v1.5, ~25 MB)"

echo "all model weights ready in $MODEL_DIR"
