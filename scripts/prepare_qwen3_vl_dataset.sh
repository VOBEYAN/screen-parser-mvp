#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT_JSONL="${INPUT_JSONL:-$ROOT_DIR/data/finetune/qwen_vl_component_recognition_jpg/data.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-$ROOT_DIR/data/finetune/qwen_vl_component_recognition_jpg/images}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/data/finetune/hf_qwen_component_recognition}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SEED="${SEED:-42}"
COPY_MODE="${COPY_MODE:-hardlink}"

if [[ ! -f "$INPUT_JSONL" ]]; then
  echo "Missing input JSONL: $INPUT_JSONL" >&2
  exit 1
fi

if [[ ! -d "$IMAGE_ROOT" ]]; then
  echo "Missing image root: $IMAGE_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" "$ROOT_DIR/scripts/convert_qwen_vl_to_hf.py" \
  --input "$INPUT_JSONL" \
  --image-root "$IMAGE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --val-ratio "$VAL_RATIO" \
  --seed "$SEED" \
  --copy-mode "$COPY_MODE"

