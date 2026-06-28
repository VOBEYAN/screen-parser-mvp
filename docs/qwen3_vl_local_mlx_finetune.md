# Qwen3-VL 本地 Mac 微调

这份流程记录的是已在本机跑通的 Apple Silicon 本地 LoRA 微调路径。

## 结论

- 可用路线：PyTorch MPS + PEFT LoRA。
- 已验证模型：`mlx-community/Qwen3-VL-2B-Instruct-bf16` 转换后的 HF-keyed 本地目录。
- 当前 MLX-VLM 训练路线会在 Qwen3-VL 上触发 `CustomKernel` 反向传播错误，暂不作为推荐路线。
- 本机已跑通 2000 步 LoRA 续训，输出 adapter 约 44MB。

## 环境

```bash
conda activate mlx-vlm-qwen
python -m pip install -U torch torchvision transformers peft accelerate qwen-vl-utils safetensors
```

确认 MPS 可用：

```bash
python - <<'PY'
import torch
print(torch.backends.mps.is_available())
PY
```

## 数据

训练数据：

```bash
data/finetune/hf_qwen_component_recognition/train_messages.jsonl
```

验证数据：

```bash
data/finetune/hf_qwen_component_recognition/val_messages.jsonl
```

当前数据规模：

- train: 1733
- val: 190
- components: 95

## 权重转换

如果已经下载了 `mlx-community/Qwen3-VL-2B-Instruct-bf16`，先把它转换成 Transformers 能完整加载的 key：

```bash
python scripts/convert_qwen3_vl_mlx_to_hf.py \
  --source /Users/wbl/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-2B-Instruct-bf16/snapshots/c8a67a84327484ba87f5ec4f8fb927cdafd791aa \
  --output models/qwen3-vl-2b-instruct-mlx-bf16-hfkeyed
```

转换内容：

- `language_model.model.*` -> `model.language_model.*`
- `vision_tower.*` -> `model.visual.*`
- `model.visual.patch_embed.proj.weight` 做维度转置

## 冒烟测试

先跑 5 步确认 MPS 训练链路：

```bash
PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
python scripts/train_qwen3_vl_mps_peft.py \
  --model models/qwen3-vl-2b-instruct-mlx-bf16-hfkeyed \
  --max-steps 5 \
  --max-samples 5 \
  --image-size 224 \
  --output-dir output/qwen3-vl-mps-peft-smoke-5steps
```

## 本机微调

推荐先跑 1000 步：

```bash
PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
python scripts/train_qwen3_vl_mps_peft.py \
  --model models/qwen3-vl-2b-instruct-mlx-bf16-hfkeyed \
  --max-steps 1000 \
  --max-samples 1733 \
  --image-size 224 \
  --lr 1e-4 \
  --log-every 25 \
  --output-dir output/qwen3-vl-mps-peft-component-lora-1000steps
```

继续微调：

```bash
PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
python scripts/train_qwen3_vl_mps_peft.py \
  --model models/qwen3-vl-2b-instruct-mlx-bf16-hfkeyed \
  --adapter output/qwen3-vl-mps-peft-component-lora-1000steps \
  --max-steps 1000 \
  --max-samples 1733 \
  --image-size 224 \
  --lr 5e-5 \
  --seed 123 \
  --log-every 25 \
  --output-dir output/qwen3-vl-mps-peft-component-lora-2000steps
```

## 本机结果

已生成：

```bash
output/qwen3-vl-mps-peft-component-lora-2000steps
```

快速验证结果：

- 前 20 条 validation：JSON 可解析 20/20
- 前 20 条 validation：componentId 命中 20/20

## 接入服务

本地 LoRA 已接入 `app.multimodal_classifier.MultimodalComponentClassifier`。
启动服务时加 `--multimodal-classifier --local-qwen`，会在组件匹配阶段把每个检测节点 crop 送入本地 Qwen3-VL LoRA，再把返回的 `componentId` 合并回节点候选。

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m app.server \
  --port 8765 \
  --reference-library data/component-reference \
  --multimodal-classifier \
  --local-qwen \
  --local-qwen-model models/qwen3-vl-2b-instruct-mlx-bf16-hfkeyed \
  --local-qwen-adapter output/qwen3-vl-mps-peft-component-lora-2000steps
```

## 注意

- `models/` 和 `output/` 是本地大文件/训练输出，已被 `.gitignore` 忽略。
- 当前 Qwen3-VL 基础模型以 `model.safetensors.part-*` 的 Git LFS 分片提交；clone 后运行 `git lfs pull && bash scripts/reconstruct_qwen3_vl_model.sh` 还原。
- GitHub 不提交 LoRA adapter 输出目录。
- 本地 18GB M3 Pro 可跑 LoRA 小步数训练；完整更大规模训练仍建议用云端 GPU。
