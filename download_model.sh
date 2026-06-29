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

# ── Chat model (scored): Qwen2.5-1.5B-Instruct, Q4_K_M, ~0.99 GB, Apache-2.0 ──
# Chosen by benchmark: ~15 t/s and ~1.76 GB RSS on a 4-core CPU (flash-attn + q8 KV).
CHAT_FILE="$MODEL_DIR/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
CHAT_URL="https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"

# ── Embedding model (RAG): bge-small-en-v1.5, Q4_K_M, ~25 MB, MIT license ──────
EMBED_FILE="$MODEL_DIR/bge-small-en-v1.5-q4_k_m.gguf"
EMBED_URL="https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q4_k_m.gguf"

# ── Vision/OCR model (digitize): DeepSeek-OCR, Q8_0 + mmproj, ~3.6 GB ──────────
# Optical compression → few vision tokens → ~40 s/page on CPU and robust to scans/
# photos. App-side only (not the benchmarked model); run via native llama-mtmd-cli.
VLM_FILE="$MODEL_DIR/DeepSeek-OCR-Q8_0.gguf"
VLM_URL="https://huggingface.co/ggml-org/DeepSeek-OCR-GGUF/resolve/main/DeepSeek-OCR-Q8_0.gguf"
VLM_MMPROJ_FILE="$MODEL_DIR/mmproj-DeepSeek-OCR-Q8_0.gguf"
VLM_MMPROJ_URL="https://huggingface.co/ggml-org/DeepSeek-OCR-GGUF/resolve/main/mmproj-DeepSeek-OCR-Q8_0.gguf"

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

download "$CHAT_FILE"        "$CHAT_URL"        "chat model (Qwen2.5-1.5B Q4_K_M, ~0.99 GB)"
download "$EMBED_FILE"       "$EMBED_URL"       "embedding model (bge-small-en-v1.5, ~25 MB)"
download "$VLM_FILE"         "$VLM_URL"         "OCR model (DeepSeek-OCR Q8_0, ~3.1 GB)"
download "$VLM_MMPROJ_FILE"  "$VLM_MMPROJ_URL"  "OCR projector (mmproj Q8_0, ~0.45 GB)"

echo "all model weights ready in $MODEL_DIR"

# ── Native llama.cpp binaries (provide llama-mtmd-cli for OCR, llama-bench for the
# profiler). Fetched into bin/ on Linux/macOS so the app is self-contained offline.
# On Windows, install llama.cpp manually and set ADTC_MTMD_CLI (see app/README).
BIN_DIR="$HERE/bin"
if [[ ! -x "$BIN_DIR/llama-mtmd-cli" ]]; then
  OS="$(uname -s 2>/dev/null || echo unknown)"
  ASSET=""
  case "$OS" in
    Linux)  ASSET="llama-b9835-bin-ubuntu-x64.zip" ;;
    Darwin) ASSET="llama-b9835-bin-macos-arm64.zip" ;;
  esac
  if [[ -n "$ASSET" ]] && command -v curl > /dev/null 2>&1; then
    echo "[download] native llama.cpp binaries ($ASSET)"
    mkdir -p "$BIN_DIR"
    if curl -L --fail -o "$BIN_DIR/llama.zip" \
        "https://github.com/ggml-org/llama.cpp/releases/download/b9835/$ASSET"; then
      (cd "$BIN_DIR" && unzip -o -j llama.zip '*/llama-mtmd-cli' '*/llama-bench' \
        'llama-mtmd-cli' 'llama-bench' 2>/dev/null; chmod +x llama-mtmd-cli llama-bench 2>/dev/null)
      rm -f "$BIN_DIR/llama.zip"
      echo "[done] native binaries in $BIN_DIR"
    fi
  else
    echo "[note] auto-fetch of llama.cpp binaries skipped on $OS — see app/README for setup."
  fi
fi
