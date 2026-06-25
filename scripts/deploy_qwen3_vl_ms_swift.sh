#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}"
ADAPTERS="${ADAPTERS:-$ROOT_DIR/output/qwen3-vl-screen-parser-lora/checkpoint-xxx}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVE_NAME="${SERVE_NAME:-qwen3-vl-screen-parser}"
INFER_BACKEND="${INFER_BACKEND:-transformers}"
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export IMAGE_MAX_TOKEN_NUM
export VIDEO_MAX_TOKEN_NUM
export FPS_MAX_FRAMES
export PYTORCH_CUDA_ALLOC_CONF

if [[ ! -e "$ADAPTERS" ]]; then
  echo "Missing adapter path: $ADAPTERS" >&2
  echo "Edit ADAPTERS to point at your checkpoint directory." >&2
  exit 1
fi

echo "Serving model: $SERVE_NAME"
echo "Base model: $MODEL_ID"
echo "Adapters: $ADAPTERS"
echo "Endpoint: http://$HOST:$PORT/v1/chat/completions"
exec swift deploy \
  --model "$MODEL_ID" \
  --adapters "$ADAPTERS" \
  --infer_backend "$INFER_BACKEND" \
  --host "$HOST" \
  --port "$PORT" \
  --served_model_name "$SERVE_NAME"
