# Qwen3-VL 服务器微调流程

这份流程用于在服务器上直接跑 Qwen3-VL 的 LoRA 微调。

## 适合的模型

- 首选：`Qwen/Qwen3-VL-4B-Instruct`
- 如果显存很紧张：先把 `IMAGE_MAX_TOKEN_NUM` 调低，再把 `per_device_train_batch_size` 保持为 `1`
- 如果服务器上已经有本地模型目录，也可以把 `MODEL_ID` 改成本地路径

## 1. 准备环境

```bash
conda create -n qwen3-vl-ft python=3.10 -y
conda activate qwen3-vl-ft

# 先按你的 CUDA 版本安装 PyTorch。
# 下面是 CUDA 12.1 示例。
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 ms-swift
pip install -U "ms-swift[all]"
```

如果 `flash-attn` 安装失败，先别卡住，训练脚本里把 `ATTN_IMPL=sdpa` 即可。

## 2. 准备数据

把这个仓库和原始数据目录放到服务器上，至少要有：

- `scripts/convert_qwen_vl_to_hf.py`
- `data/finetune/qwen_vl_component_recognition_jpg/`

然后执行：

```bash
bash scripts/prepare_qwen3_vl_dataset.sh
```

生成结果会在：

- `data/finetune/hf_qwen_component_recognition/train_swift.jsonl`
- `data/finetune/hf_qwen_component_recognition/val_swift.jsonl`

这两个文件就是训练要用的。

## 3. 开始训练

单卡先跑这个：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_qwen3_vl_ms_swift.sh
```

多卡可以这样：

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash scripts/train_qwen3_vl_ms_swift.sh
```

常用可调参数：

- `MODEL_ID`：基础模型，默认 `Qwen/Qwen3-VL-4B-Instruct`
- `USE_HF=true`：强制从 HuggingFace 下载；不设时走 ms-swift 默认来源
- `OUTPUT_DIR`：LoRA 输出目录
- `IMAGE_MAX_TOKEN_NUM`：图片 token 上限，显存不够时先降这个
- `MAX_LENGTH`：上下文长度，默认 `4096`
- `NUM_TRAIN_EPOCHS`：轮数，默认 `3`
- `RESUME_FROM_CHECKPOINT`：断点续训时指定 checkpoint 目录

## 4. 断点续训

如果训练中断了，继续跑同一个脚本，并加上：

```bash
RESUME_FROM_CHECKPOINT=output/qwen3-vl-screen-parser-lora/checkpoint-xxx \
bash scripts/train_qwen3_vl_ms_swift.sh
```

这会恢复优化器状态和随机种子。

如果你只是想加载 LoRA 权重继续跑，不恢复优化器状态，就用 `--adapters` 那类方式。

## 5. 快速验证

训练完以后，先做一次交互式推理：

```bash
CUDA_VISIBLE_DEVICES=0 IMAGE_MAX_TOKEN_NUM=1024 swift infer \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --adapters output/qwen3-vl-screen-parser-lora/checkpoint-xxx \
  --stream true \
  --max_new_tokens 256
```

## 6. 导出和部署

### 方案 A：直接带 LoRA 部署

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/deploy_qwen3_vl_ms_swift.sh
```

默认会启动一个 OpenAI 兼容服务：

- `http://SERVER_IP:8000/v1/chat/completions`
- 模型名：`qwen3-vl-screen-parser`

### 方案 B：先合并再部署

```bash
CUDA_VISIBLE_DEVICES=0 swift export \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --adapters output/qwen3-vl-screen-parser-lora/checkpoint-xxx \
  --merge_lora true \
  --output_dir output/qwen3-vl-screen-parser-merged

CUDA_VISIBLE_DEVICES=0 swift deploy \
  --model output/qwen3-vl-screen-parser-merged \
  --host 0.0.0.0 \
  --port 8000
```

## 7. 接回你这个项目

把下面这些环境变量指向服务器服务：

```bash
export SCREEN_PARSER_VLM_BASE_URL="http://SERVER_IP:8000/v1"
export SCREEN_PARSER_VLM_MODEL="qwen3-vl-screen-parser"
export SCREEN_PARSER_VLM_API_KEY="EMPTY"
```

然后再启动本项目：

```bash
./start_studio.command
```

## 8. 常见问题

- 显存不够：先把 `IMAGE_MAX_TOKEN_NUM` 调低，再把 `MAX_LENGTH` 调到 `2048`
- 没网：把 `MODEL_ID` 改成本地模型目录，并设置 `CHECK_MODEL=false`
- 训练很慢：先跑 `1` 轮 smoke test，确认流程通了再正式跑
- 部署报 `Missing adapter path`：把 `ADAPTERS` 改成真实 checkpoint 目录，例如 `ADAPTERS=output/qwen3-vl-screen-parser-lora/checkpoint-500`
