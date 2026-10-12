#!/bin/bash
# Launch llama.cpp's llama-server for LLaMA 4 Maverick (GGUF) with its vision projector,
# so that frame-based strategies receive the images.
#
# Required:
#   LLAMA_SERVER_BIN  path to the llama-server binary (built from llama.cpp)
#   MODEL_PATH        path to the first shard of the Llama-4-Maverick GGUF
#   MMPROJ_PATH       path to the matching mmproj GGUF (vision projector)
# Optional:
#   HOST, PORT        where the server listens
#   EXTRA_ARGS        any further llama-server flags (context size, threads,
#                     GPU layers, tensor offloading, ...)
set -euo pipefail

: "${LLAMA_SERVER_BIN:?set LLAMA_SERVER_BIN}"
: "${MODEL_PATH:?set MODEL_PATH}"
: "${MMPROJ_PATH:?set MMPROJ_PATH}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

# shellcheck disable=SC2086
exec "$LLAMA_SERVER_BIN" \
  --model "$MODEL_PATH" \
  --mmproj "$MMPROJ_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  ${EXTRA_ARGS:-}
