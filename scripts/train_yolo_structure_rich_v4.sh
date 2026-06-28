#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET="${DATASET:-data/screen-structure-rich-v4}"
PRESET="${PRESET:-balanced}"
BASE_MODEL="${BASE_MODEL:-models/yolo_screen_structure_chart_hard_v3.pt}"
RUN_NAME="${RUN_NAME:-yolo_screen_structure_rich_v4}"
EPOCHS="${EPOCHS:-60}"
IMGSZ="${IMGSZ:-1280}"
BATCH="${BATCH:-4}"
DEVICE="${DEVICE:-mps}"
WORKERS="${WORKERS:-2}"
PATIENCE="${PATIENCE:-18}"
CLOSE_MOSAIC="${CLOSE_MOSAIC:-10}"
FRACTION="${FRACTION:-1.0}"
BUILD_DATASET="${BUILD_DATASET:-1}"
PROJECT="${PROJECT:-$ROOT/runs/detect}"

if [[ "$BUILD_DATASET" == "1" || ! -f "$DATASET/data.yaml" ]]; then
  python scripts/build_yolo_rich_dataset.py \
    --out "$DATASET" \
    --preset "$PRESET" \
    --clean
fi

python scripts/train_yolo.py \
  --data "$DATASET/data.yaml" \
  --base-model "$BASE_MODEL" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --patience "$PATIENCE" \
  --fraction "$FRACTION" \
  --close-mosaic "$CLOSE_MOSAIC" \
  --project "$PROJECT" \
  --name "$RUN_NAME" \
  --exist-ok

if [[ -f "$PROJECT/$RUN_NAME/weights/best.pt" ]]; then
  cp "$PROJECT/$RUN_NAME/weights/best.pt" "models/${RUN_NAME}.pt"
  cp "$PROJECT/$RUN_NAME/weights/last.pt" "models/${RUN_NAME}_last.pt"
  echo "copied models/${RUN_NAME}.pt"
fi
