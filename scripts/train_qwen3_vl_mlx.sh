#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-mlx-community/Qwen3-VL-2B-Instruct-bf16}"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/data/finetune/mlx_qwen_component_recognition}"
SPLIT="${SPLIT:-train}"
DATASET_CONFIG="${DATASET_CONFIG:-}"
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/output/qwen3-vl-mlx-lora/adapters.safetensors}"
ADAPTER_PATH="${ADAPTER_PATH:-}"

BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS_VALUE="${EPOCHS:-}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
ITERS="${ITERS:-20}"
STEPS_PER_REPORT="${STEPS_PER_REPORT:-10}"
STEPS_PER_EVAL="${STEPS_PER_EVAL:-200}"
STEPS_PER_SAVE="${STEPS_PER_SAVE:-100}"
VAL_BATCHES="${VAL_BATCHES:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
GRAD_CHECKPOINT="${GRAD_CHECKPOINT:-true}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TRAIN_ON_COMPLETIONS="${TRAIN_ON_COMPLETIONS:-true}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
ASSISTANT_ID="${ASSISTANT_ID:-77091}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
IMAGE_RESIZE_W="${IMAGE_RESIZE_W:-512}"
IMAGE_RESIZE_H="${IMAGE_RESIZE_H:-512}"
TRAIN_VISION="${TRAIN_VISION:-false}"
FULL_FINETUNE="${FULL_FINETUNE:-false}"

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Missing dataset dir: $DATASET_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

cmd=(
  "$PYTHON_BIN" -m mlx_vlm.lora
  --model-path "$MODEL_PATH"
  --dataset "$DATASET_DIR"
  --split "$SPLIT"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --iters "$ITERS"
  --steps-per-report "$STEPS_PER_REPORT"
  --steps-per-eval "$STEPS_PER_EVAL"
  --steps-per-save "$STEPS_PER_SAVE"
  --val-batches "$VAL_BATCHES"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --grad-clip "$GRAD_CLIP"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --assistant-id "$ASSISTANT_ID"
  --lora-rank "$LORA_RANK"
  --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT"
  --output-path "$OUTPUT_PATH"
  --image-resize-shape "$IMAGE_RESIZE_W" "$IMAGE_RESIZE_H"
)

if [[ -n "$DATASET_CONFIG" ]]; then
  cmd+=(--dataset-config "$DATASET_CONFIG")
fi

if [[ -n "$EPOCHS_VALUE" ]]; then
  cmd+=(--epochs "$EPOCHS_VALUE")
fi

if [[ -n "$ADAPTER_PATH" ]]; then
  cmd+=(--adapter-path "$ADAPTER_PATH")
fi

if [[ "$GRAD_CHECKPOINT" == "true" ]]; then
  cmd+=(--grad-checkpoint)
fi

if [[ "$TRAIN_ON_COMPLETIONS" == "true" ]]; then
  cmd+=(--train-on-completions)
fi

if [[ "$FULL_FINETUNE" == "true" ]]; then
  cmd+=(--full-finetune)
fi

if [[ "$TRAIN_VISION" == "true" ]]; then
  cmd+=(--train-vision)
fi

echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_DIR"
echo "Split: $SPLIT"
echo "Output: $OUTPUT_PATH"
exec "${cmd[@]}"
