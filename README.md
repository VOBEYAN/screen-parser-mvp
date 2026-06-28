# screen-parser-mvp

大屏设计图解析工作台。它负责上传设计图、检测组件区域、结合 OCR / 视觉大模型 / 组件参考库识别组件，并输出可被 `ai-schema-view` 运行时渲染的大屏组件数据。

配套前端组件库仓库：

```text
https://github.com/VOBEYAN/ai-schema-view
```

## 功能

- 上传大屏设计图，输出检测框、层级树、组件候选、识别报告和结构化 JSON。
- 支持 YOLO 结构检测模型；未放置权重时会走本地 OpenCV fallback，方便先跑通流程。
- 支持 PaddleOCR 文字识别。
- 支持本地 Qwen3-VL LoRA 组件识别模型，也支持 OpenAI-compatible 视觉大模型接口，例如 DashScope Qwen VL。
- 内置 `data/component-reference` 组件参考库，用于把识别结果匹配到 `ai-schema-view` 的组件 ID。
- 输出 `aiSchemaComponents`，可直接被 `ai-schema-view` 的 `/schema-render` 页面读取并运行时渲染。
- 提供纠错样本保存和 Qwen VL 微调数据导出接口，自动生成 `data.jsonl` 和训练 zip 包。

## 快速启动

```bash
git clone https://github.com/VOBEYAN/screen-parser-mvp.git
cd screen-parser-mvp

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m app.server \
  --port 8765 \
  --reference-library data/component-reference \
  --multimodal-classifier
```

浏览器打开：

```text
http://127.0.0.1:8765/
```

也可以使用一键脚本：

```bash
chmod +x start_studio.command
./start_studio.command
```

## 视觉大模型配置

## 使用本地 Qwen3-VL LoRA

本仓库现在可以直接加载本地微调后的 Qwen3-VL LoRA，用于把检测出来的组件 crop 识别成 `ai-schema-view` 的 `componentId`。

默认路径：

```text
models/qwen3-vl-2b-instruct-mlx-bf16-hfkeyed
output/qwen3-vl-mps-peft-component-lora-2000steps
```

GitHub 上的基础模型使用 Git LFS 分片保存。克隆仓库后先拉取 LFS 文件并还原权重：

```bash
git lfs pull
bash scripts/reconstruct_qwen3_vl_model.sh
```

启动：

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m app.server \
  --port 8765 \
  --reference-library data/component-reference \
  --multimodal-classifier \
  --local-qwen \
  --local-qwen-model models/qwen3-vl-2b-instruct-mlx-bf16-hfkeyed \
  --local-qwen-adapter output/qwen3-vl-mps-peft-component-lora-2000steps
```

`start_studio.command` 已默认优先使用 `mlx-vlm-qwen` conda 环境和这个本地 LoRA。模型会懒加载：第一次解析图片时才加载到 MPS/GPU/CPU。

## 远程视觉大模型配置

不要把真实 API Key 写进代码或提交到 Git。启动前用环境变量配置：

```bash
export SCREEN_PARSER_VLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export SCREEN_PARSER_VLM_MODEL="qwen3-vl-flash"
export SCREEN_PARSER_VLM_API_KEY="你的 API Key"
export SCREEN_PARSER_VLM_FORCE="true"
export SCREEN_PARSER_VLM_MAX_NODES="18"
export SCREEN_PARSER_VLM_CANDIDATE_K="95"

python -m app.server \
  --port 8765 \
  --reference-library data/component-reference \
  --multimodal-classifier
```

说明：

- 未配置 `SCREEN_PARSER_VLM_API_KEY` 时，不会真正调用大模型，只会使用本地 OCR / 视觉规则 / 组件参考库。
- `SCREEN_PARSER_VLM_FORCE=true` 会提高大模型在最终组件判断中的参与度。
- 页面和结果 JSON 里会显示本次大模型调用次数。

## 使用训练好的检测模型

模型权重体积较大，默认不提交到 GitHub。拿到权重后放到：

```text
models/yolo_screen_structure_chart_hard_v3.pt
models/graph_transformer_structure_local_v1.pt
```

然后这样启动：

```bash
python -m app.server \
  --port 8765 \
  --yolo-model models/yolo_screen_structure_chart_hard_v3.pt \
  --yolo-conf 0.05 \
  --graph-model models/graph_transformer_structure_local_v1.pt \
  --reference-library data/component-reference \
  --multimodal-classifier
```

没有这些模型也可以先启动，只是检测准确率会低于训练权重版本。

## 命令行解析图片

```bash
python scripts/parse_image.py /path/to/design.png \
  --reference-library data/component-reference \
  --multimodal-classifier \
  --top-k 5
```

输出目录：

```text
artifacts/run_*/result.json
artifacts/run_*/evidence.png
artifacts/run_*/report.md
artifacts/run_*/report.html
```

## 对接 ai-schema-view

1. 启动本仓库后端，端口默认 `8765`。
2. 启动 `ai-schema-view`。
3. 在工作台上传图片并解析。
4. 拿到结果里的 `runId` 后打开：

```text
http://127.0.0.1:3020/ai-schema/#/schema-render?api=http%3A%2F%2F127.0.0.1%3A8765&runId=run_xxx&fit=1
```

`schema-render` 会从后端读取 `aiSchemaComponents`，再从组件库里运行时渲染对应组件。

## 纠错与微调数据

后端提供两个接口：

```text
POST /api/labels
POST /api/finetune/export
```

纠错样本会保存在 `artifacts/` 下，导出的 Qwen VL 微调数据默认在：

```text
data/finetune/qwen_vl_component_recognition/data.jsonl
data/finetune/qwen_vl_component_recognition/qwen_vl_component_recognition.zip
```

导出命令：

```bash
python scripts/export_qwen_vl_finetune_dataset.py \
  --output data/finetune/qwen_vl_component_recognition/data.jsonl \
  --reference-variants 20
```

上传到百炼/Qwen VL 微调任务时，使用生成的：

```text
data/finetune/qwen_vl_component_recognition/qwen_vl_component_recognition.zip
```

zip 根目录里包含 `data.jsonl` 和对应图片文件。`data/finetune/` 是生成产物，默认不提交。微调完成后，把后端环境变量里的模型 ID 换成你的微调模型即可。

## 训练数据与模型

合成训练数据、运行报告、adapter 输出和临时模型默认不提交。当前本地 Qwen3-VL 基础模型以 `model.safetensors.part-*` 的 Git LFS 分片提交，克隆后用 `scripts/reconstruct_qwen3_vl_model.sh` 还原。

```text
data/finetune/
data/screen-structure*/
artifacts/
logs/
runs/
models/*.pt
output/
```

需要重新生成训练数据时，可使用：

```bash
python scripts/generate_composited_training_data.py \
  --out data/screen-structure-local-v1 \
  --label-mode coarse \
  --train-count 480 \
  --val-count 96 \
  --components-per-screen 10 \
  --layout-mode mixed \
  --title-placement-mode diverse \
  --overlay-rate 0.35 \
  --include-sketch \
  --clean
```

训练 YOLO：

```bash
python scripts/train_yolo.py \
  --data data/screen-structure-local-v1/data.yaml \
  --epochs 20 \
  --imgsz 640 \
  --batch 8 \
  --base-model yolo26n.pt \
  --project runs/detect \
  --name yolo_screen_structure_local_v1_640ft \
  --device mps \
  --no-amp
```

## 项目结构

```text
app/
  server.py                 # HTTP 工作台和 API
  pipeline.py               # 端到端解析流程
  detectors.py              # YOLO / OpenCV 检测
  multimodal_classifier.py  # OCR + 视觉大模型 + 本地特征判断
  visual_matcher.py         # 组件参考图匹配
  schema_hydrator.py        # 生成 ai-schema-view 可渲染参数
  finetune_data.py          # 纠错样本与微调数据导出
  web/                      # 内置工作台前端
data/component-reference/   # 95 类组件参考图和特征
scripts/                    # 解析、训练、参考库构建、微调导出脚本
```

## 常见问题

如果页面显示“大模型调用 0 次”，先确认：

```bash
echo $SCREEN_PARSER_VLM_API_KEY
echo $SCREEN_PARSER_VLM_MODEL
```

如果 PaddleOCR 安装较慢，可以先跑通后端，再按机器环境单独安装 PaddlePaddle / PaddleOCR。

如果 `schema-render` 页面没有内容，先确认后端解析结果 `result.json` 里存在 `aiSchemaComponents`，并确认 URL 的 `api` 和 `runId` 正确。
