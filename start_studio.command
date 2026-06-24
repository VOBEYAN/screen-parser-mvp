#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="/Users/wbl/miniconda3/bin/python3"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3)"
fi

URL="http://127.0.0.1:8765/"
echo "Screen Parser Studio: $URL"

export SCREEN_PARSER_VLM_BASE_URL="${SCREEN_PARSER_VLM_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export SCREEN_PARSER_VLM_MODEL="${SCREEN_PARSER_VLM_MODEL:-qwen3-vl-flash}"
export SCREEN_PARSER_VLM_FORCE="${SCREEN_PARSER_VLM_FORCE:-true}"
export SCREEN_PARSER_VLM_TIMEOUT="${SCREEN_PARSER_VLM_TIMEOUT:-25}"
export SCREEN_PARSER_VLM_MAX_NODES="${SCREEN_PARSER_VLM_MAX_NODES:-18}"
export SCREEN_PARSER_VLM_CANDIDATE_K="${SCREEN_PARSER_VLM_CANDIDATE_K:-95}"

if command -v open >/dev/null 2>&1; then
  (sleep 2 && open "$URL") >/dev/null 2>&1 &
fi

SERVER_ARGS=(
  -m app.server
  --port 8765
  --yolo-model models/yolo_screen_structure_chart_hard_v3.pt
  --yolo-conf 0.05
  --graph-model models/graph_transformer_structure_local_v1.pt
  --reference-library data/component-reference
  --multimodal-classifier
  --multimodal-base-url "$SCREEN_PARSER_VLM_BASE_URL"
  --multimodal-model "$SCREEN_PARSER_VLM_MODEL"
)

if [ -n "${SCREEN_PARSER_VLM_API_KEY:-}" ]; then
  SERVER_ARGS+=(--multimodal-api-key "$SCREEN_PARSER_VLM_API_KEY")
else
  echo "SCREEN_PARSER_VLM_API_KEY is not set; VLM calls will be skipped until you export it."
fi

exec "$PYTHON_BIN" "${SERVER_ARGS[@]}"
