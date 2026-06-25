#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


SYSTEM_PROMPT = (
    "你是 ai-schema-view 大屏组件识别助手。"
    "你只根据图片判断最匹配的组件，并严格输出 JSON。"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Bailian Qwen-VL JSONL into HuggingFace/Qwen multimodal chat datasets."
    )
    parser.add_argument(
        "--input",
        default="data/finetune/qwen_vl_component_recognition_jpg/data.jsonl",
        help="Input Bailian-style JSONL.",
    )
    parser.add_argument(
        "--image-root",
        default="data/finetune/qwen_vl_component_recognition_jpg/images",
        help="Directory containing input images referenced by the JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/finetune/hf_qwen_component_recognition",
        help="Output dataset directory.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-mode",
        choices=["hardlink", "copy", "none"],
        default="hardlink",
        help="How to place images in the output dataset directory.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_image_root = output_dir / "images"

    records = load_records(input_path, image_root)
    split = stratified_split(records, val_ratio=args.val_ratio, seed=args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_mode != "none":
        output_image_root.mkdir(parents=True, exist_ok=True)
        materialize_images(records, output_image_root, args.copy_mode)
    for record in records:
        record["dataset_image_abs"] = output_image_root / record["image_name"]

    datasets = {
        "all": records,
        "train": split["train"],
        "val": split["val"],
    }
    for name, items in datasets.items():
        write_jsonl(output_dir / f"{name}_messages.jsonl", (to_hf_qwen(item) for item in items))
        write_jsonl(output_dir / f"{name}_swift.jsonl", (to_ms_swift(item) for item in items))
        write_json(output_dir / f"{name}_llava.json", [to_llava(item) for item in items])

    write_dataset_info(output_dir)
    manifest = build_manifest(input_path, image_root, output_dir, records, split, args)
    write_json(output_dir / "manifest.json", manifest)
    write_readme(output_dir, manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_records(input_path: Path, image_root: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise SystemExit(f"Input JSONL not found: {input_path}")
    if not image_root.exists():
        raise SystemExit(f"Image root not found: {image_root}")

    records: List[Dict[str, Any]] = []
    for index, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        messages = item.get("messages") or []
        if len(messages) < 2:
            raise ValueError(f"Line {index}: expected user and assistant messages")

        user_content = messages[0].get("content") or []
        assistant_content = messages[1].get("content") or []
        prompt = first_text(user_content)
        image_name = first_image(user_content)
        answer_text = first_text(assistant_content)
        if not prompt or not image_name or not answer_text:
            raise ValueError(f"Line {index}: missing prompt, image, or answer")

        image_path = image_root / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Line {index}: image not found: {image_path}")
        answer = json.loads(answer_text)
        component_id = str(answer.get("componentId") or "")
        if not component_id:
            raise ValueError(f"Line {index}: assistant answer missing componentId")

        records.append(
            {
                "id": Path(image_name).stem,
                "source_line": index,
                "prompt": prompt,
                "image_name": image_name,
                "image_path": str(image_path),
                "answer": answer,
                "answer_text": json.dumps(answer, ensure_ascii=False, separators=(",", ":")),
                "component_id": component_id,
            }
        )
    return records


def first_text(content: Iterable[Dict[str, Any]]) -> str:
    for item in content:
        text = item.get("text")
        if text:
            return str(text)
    return ""


def first_image(content: Iterable[Dict[str, Any]]) -> str:
    for item in content:
        image = item.get("image")
        if image:
            return str(image)
    return ""


def stratified_split(records: List[Dict[str, Any]], val_ratio: float, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    by_component: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_component[record["component_id"]].append(record)

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for items in by_component.values():
        shuffled = list(items)
        rng.shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
        val.extend(shuffled[:val_count])
        train.extend(shuffled[val_count:])

    train.sort(key=lambda item: item["source_line"])
    val.sort(key=lambda item: item["source_line"])
    return {"train": train, "val": val}


def materialize_images(records: List[Dict[str, Any]], output_image_root: Path, mode: str) -> None:
    for record in records:
        src = Path(record["image_path"])
        dst = output_image_root / record["image_name"]
        if dst.exists():
            continue
        if mode == "hardlink":
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        elif mode == "copy":
            shutil.copy2(src, dst)


def to_hf_qwen(record: Dict[str, Any]) -> Dict[str, Any]:
    image = path_to_file_uri(Path(record["dataset_image_abs"]))
    return {
        "id": record["id"],
        "images": [image],
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": record["prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": record["answer_text"]}],
            },
        ],
    }


def to_ms_swift(record: Dict[str, Any]) -> Dict[str, Any]:
    image = str(Path(record["dataset_image_abs"]).resolve())
    return {
        "id": record["id"],
        "images": [image],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>\n{record['prompt']}"},
            {"role": "assistant", "content": record["answer_text"]},
        ],
    }


def to_llava(record: Dict[str, Any]) -> Dict[str, Any]:
    image = path_to_file_uri(Path(record["dataset_image_abs"]))
    return {
        "id": record["id"],
        "image": image,
        "conversations": [
            {"from": "human", "value": f"<image>\n{record['prompt']}"},
            {"from": "gpt", "value": record["answer_text"]},
        ],
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def path_to_file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def write_dataset_info(output_dir: Path) -> None:
    dataset_info = {
        "screen_parser_qwen_vl_train": {
            "file_name": "train_messages.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
        "screen_parser_qwen_vl_val": {
            "file_name": "val_messages.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
    }
    write_json(output_dir / "dataset_info.json", dataset_info)


def build_manifest(
    input_path: Path,
    image_root: Path,
    output_dir: Path,
    records: List[Dict[str, Any]],
    split: Dict[str, List[Dict[str, Any]]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    counts = Counter(record["component_id"] for record in records)
    return {
        "source": str(input_path),
        "sourceImageRoot": str(image_root),
        "outputDir": str(output_dir),
        "format": "hf_qwen_multimodal_messages",
        "recordCount": len(records),
        "trainCount": len(split["train"]),
        "valCount": len(split["val"]),
        "componentCount": len(counts),
        "minSamplesPerComponent": min(counts.values()) if counts else 0,
        "maxSamplesPerComponent": max(counts.values()) if counts else 0,
        "valRatio": args.val_ratio,
        "seed": args.seed,
        "copyMode": args.copy_mode,
        "files": {
            "hfAll": "all_messages.jsonl",
            "hfTrain": "train_messages.jsonl",
            "hfVal": "val_messages.jsonl",
            "swiftAll": "all_swift.jsonl",
            "swiftTrain": "train_swift.jsonl",
            "swiftVal": "val_swift.jsonl",
            "llavaAll": "all_llava.json",
            "llavaTrain": "train_llava.json",
            "llavaVal": "val_llava.json",
            "datasetInfo": "dataset_info.json",
            "images": "images/",
        },
    }


def write_readme(output_dir: Path, manifest: Dict[str, Any]) -> None:
    text = f"""# HF/Qwen Component Recognition Dataset

Converted from `{manifest['source']}`.

## Files

- `train_swift.jsonl`, `val_swift.jsonl`, `all_swift.jsonl`: ms-swift/Qwen3-VL custom multimodal records.
- `train_messages.jsonl`, `val_messages.jsonl`, `all_messages.jsonl`: HuggingFace/Qwen multimodal chat records.
- `train_llava.json`, `val_llava.json`, `all_llava.json`: LLaVA-style conversation records for Qwen-VL fine-tuning scripts that expect that layout.
- `images/`: image files referenced by the dataset records.
- `dataset_info.json`: LLaMA-Factory-style dataset registration helper.

## Counts

- Records: {manifest['recordCount']}
- Train: {manifest['trainCount']}
- Val: {manifest['valCount']}
- Components: {manifest['componentCount']}

Each HuggingFace/Qwen record uses this shape:

```json
{{
  "images": ["file:///absolute/path/to/example.jpg"],
  "messages": [
    {{"role": "system", "content": [{{"type": "text", "text": "..."}}]}},
    {{"role": "user", "content": [{{"type": "image", "image": "file:///absolute/path/to/example.jpg"}}, {{"type": "text", "text": "..."}}]}},
    {{"role": "assistant", "content": [{{"type": "text", "text": "{{...json...}}"}}]}}
  ]
}}
```
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
