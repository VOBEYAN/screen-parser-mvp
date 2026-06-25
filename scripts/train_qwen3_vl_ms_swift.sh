#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/finetune/hf_qwen_component_recognition}"
TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/train_swift.jsonl}"
VAL_DATA="${VAL_DATA:-$DATA_DIR/val_swift.jsonl}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output/qwen3-vl-screen-parser-lora}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
FREEZE_VIT="${FREEZE_VIT:-true}"
FREEZE_ALIGNER="${FREEZE_ALIGNER:-true}"
PACKING="${PACKING:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
USE_HF="${USE_HF:-false}"
CHECK_MODEL="${CHECK_MODEL:-true}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-true}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "$TRAIN_DATA" ]]; then
  if [[ -f "$ROOT_DIR/data/finetune/qwen_vl_component_recognition_jpg/data.jsonl" ]]; then
    echo "Dataset not found, preparing it first..."
    bash "$ROOT_DIR/scripts/prepare_qwen3_vl_dataset.sh"
  fi
fi

if [[ ! -f "$TRAIN_DATA" ]]; then
  echo "Missing training dataset: $TRAIN_DATA" >&2
  echo "Run scripts/prepare_qwen3_vl_dataset.sh first." >&2
  exit 1
fi

if [[ ! -f "$VAL_DATA" ]]; then
  echo "Missing validation dataset: $VAL_DATA" >&2
  echo "Run scripts/prepare_qwen3_vl_dataset.sh first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES
export IMAGE_MAX_TOKEN_NUM
export VIDEO_MAX_TOKEN_NUM
export FPS_MAX_FRAMES
export PYTORCH_CUDA_ALLOC_CONF

cmd=(
  swift sft
  --model "$MODEL_ID"
  --dataset "$TRAIN_DATA"
  --val_dataset "$VAL_DATA"
  --tuner_type lora
  --torch_dtype bfloat16
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE"
  --lora_rank "$LORA_RANK"
  --lora_alpha "$LORA_ALPHA"
  --target_modules "$TARGET_MODULES"
  --freeze_vit "$FREEZE_VIT"
  --freeze_aligner "$FREEZE_ALIGNER"
  --packing "$PACKING"
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING"
  --max_length "$MAX_LENGTH"
  --output_dir "$OUTPUT_DIR"
  --warmup_ratio "$WARMUP_RATIO"
  --save_strategy steps
  --save_steps "$SAVE_STEPS"
  --save_total_limit "$SAVE_TOTAL_LIMIT"
  --dataset_num_proc "$DATASET_NUM_PROC"
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
  --attn_impl "$ATTN_IMPL"
  --load_from_cache_file "$LOAD_FROM_CACHE_FILE"
)

if [[ "$USE_HF" == "true" ]]; then
  cmd+=(--use_hf true)
fi

if [[ "$CHECK_MODEL" == "false" ]]; then
  cmd+=(--check_model false)
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  cmd+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

if [[ "$NPROC_PER_NODE" != "1" ]]; then
  export NPROC_PER_NODE
fi

mkdir -p "$OUTPUT_DIR"
echo "Training model: $MODEL_ID"
echo "Train data: $TRAIN_DATA"
echo "Val data: $VAL_DATA"
echo "Output dir: $OUTPUT_DIR"
exec "${cmd[@]}"
