#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/models/qwen3-vl-2b-instruct-mlx-bf16-hfkeyed"
OUTPUT="$MODEL_DIR/model.safetensors"
EXPECTED_SHA256="7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"

if ! compgen -G "$MODEL_DIR/model.safetensors.part-*" >/dev/null; then
  echo "No model shards found in $MODEL_DIR" >&2
  echo "Run 'git lfs pull' first." >&2
  exit 1
fi

if [[ -f "$OUTPUT" && "${1:-}" != "--force" ]]; then
  actual_sha256="$(shasum -a 256 "$OUTPUT" | awk '{print $1}')"
  if [[ "$actual_sha256" == "$EXPECTED_SHA256" ]]; then
    echo "Model already exists and checksum is valid: $OUTPUT"
    exit 0
  fi
  echo "Existing model checksum does not match." >&2
  echo "Run '$0 --force' to rebuild it from shards." >&2
  exit 1
fi

cat "$MODEL_DIR"/model.safetensors.part-* >"$OUTPUT"
actual_sha256="$(shasum -a 256 "$OUTPUT" | awk '{print $1}')"

if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
  echo "Checksum mismatch after reconstruction." >&2
  echo "Expected: $EXPECTED_SHA256" >&2
  echo "Actual:   $actual_sha256" >&2
  exit 1
fi

echo "Reconstructed model: $OUTPUT"
