#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import unquote, urlparse

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "finetune" / "hf_qwen_component_recognition" / "train_messages.jsonl"
DEFAULT_OUTPUT = ROOT / "output" / "qwen3-vl-mps-peft-smoke"


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL LoRA training loop for Apple MPS.")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--adapter", default=None, help="Optional existing LoRA adapter to continue training.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--train-on-full-text", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    device = resolve_device(args.device)
    records = list(load_records(Path(args.dataset)))
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(records)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    if not records:
        raise SystemExit(f"No records loaded from {args.dataset}")

    print(f"Loading processor: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)

    print(f"Loading model: {args.model}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "mps" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.to(device)
    model.gradient_checkpointing_enable()

    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = min(args.max_steps, len(records))
    recent_losses: List[float] = []
    for step, record in enumerate(records[:total_steps], start=1):
        messages = normalize_messages(record["messages"])
        prompt_messages = strip_assistant_messages(messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        if args.image_size and image_inputs:
            image_inputs = [image.resize((args.image_size, args.image_size)) for image in image_inputs]

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()
        if not args.train_on_full_text:
            prompt_inputs = processor(
                text=[prompt_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            prompt_len = min(prompt_inputs["input_ids"].shape[1], labels.shape[1])
            labels[:, :prompt_len] = -100
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        inputs["labels"] = labels
        inputs = move_to_device(inputs, device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        recent_losses.append(loss.detach().float().cpu().item())
        if step % args.log_every == 0 or step == total_steps:
            avg_loss = sum(recent_losses) / len(recent_losses)
            print(f"step {step}/{total_steps} loss={recent_losses[-1]:.4f} avg_loss={avg_loss:.4f}")
            recent_losses.clear()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved adapter to {output_dir}")


def resolve_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                yield json.loads(line)


def normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            normalized.append({"role": message["role"], "content": content})
            continue
        parts: List[Dict[str, Any]] = []
        for part in content:
            if part.get("type") == "image":
                image = part.get("image") or ""
                if image.startswith("file://"):
                    image = unquote(urlparse(image).path)
                parts.append({"type": "image", "image": image})
            elif part.get("type") == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
            else:
                parts.append(part)
        normalized.append({"role": message["role"], "content": parts})
    return normalized


def strip_assistant_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [message for message in messages if message.get("role") != "assistant"]


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


if __name__ == "__main__":
    main()
