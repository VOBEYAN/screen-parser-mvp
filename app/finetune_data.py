from __future__ import annotations

import json
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image, ImageEnhance, ImageOps

from .schemas import BBox


DATASET_ROOT = Path(__file__).resolve().parents[1] / "data" / "finetune"
REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "data" / "component-reference"
PROMPT = (
    "请从 ai-schema-view 的 95 个组件库中识别图片里的组件，只返回 JSON。"
    "JSON 字段包含 componentId、visualForm、confidence、nearMisses。"
)


def save_correction_sample(
    artifact_root: Path,
    run_id: str,
    node_id: str,
    correct_component_id: str,
    visual_form: str = "",
    note: str = "",
) -> Dict[str, Any]:
    run_dir = artifact_root / run_id
    result_path = run_dir / "result.json"
    if not result_path.exists():
        raise ValueError(f"Run not found: {run_id}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    node = next((item for item in result.get("nodes", []) if item.get("node_id") == node_id), None)
    if not node:
        raise ValueError(f"Node not found: {node_id}")
    if not correct_component_id:
        raise ValueError("Missing correct componentId")

    image_path = Path(str(result.get("imageMeta", {}).get("path") or ""))
    if not image_path.exists():
        raise ValueError(f"Source image not found: {image_path}")

    bbox = BBox(**node["bbox"])
    crop = crop_image(image_path, bbox)
    sample_dir = DATASET_ROOT / "corrections" / "images"
    sample_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_filename(f"{run_id}_{node_id}_{correct_component_id}.png")
    image_out = sample_dir / safe_name
    crop.save(image_out)

    classifier = (node.get("features") or {}).get("contentClassifier") or {}
    wrong_component = node.get("component_id") or ((node.get("candidates") or [{}])[0] or {}).get("componentId") or ""
    sample = {
        "id": safe_name.removesuffix(".png"),
        "source": "manual_correction",
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "runId": run_id,
        "nodeId": node_id,
        "image": str(image_out.relative_to(DATASET_ROOT)),
        "fullImage": str(image_path),
        "bbox": node.get("bbox"),
        "ocrText": classifier.get("paddleOcrText") or "",
        "modelText": classifier.get("text") or "",
        "wrongComponentId": wrong_component,
        "correctComponentId": correct_component_id,
        "visualForm": visual_form or classifier.get("localVisualForm") or classifier.get("llmVisualForm") or "",
        "note": note,
    }
    append_jsonl(DATASET_ROOT / "corrections" / "labels.jsonl", sample)
    return sample


def export_qwen_vl_dataset(
    output_path: Path,
    reference_variants: int = 8,
    include_corrections: bool = True,
    limit_components: Optional[int] = None,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_output_dir = output_path.parent / "images"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    records.extend(build_reference_records(image_output_dir, reference_variants, limit_components=limit_components))
    if include_corrections:
        records.extend(build_correction_records())

    with output_path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(to_qwen_message(record), ensure_ascii=False) + "\n")

    package_path = output_path.parent / "qwen_vl_component_recognition.zip"
    package_qwen_vl_dataset(output_path, records, package_path)

    manifest = {
        "output": str(output_path),
        "package": str(package_path),
        "recordCount": len(records),
        "referenceVariants": reference_variants,
        "includeCorrections": include_corrections,
        "createdAt": datetime.now().isoformat(timespec="seconds"),
    }
    (output_path.parent / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_reference_records(image_output_dir: Path, variants: int, limit_components: Optional[int] = None) -> List[Dict[str, Any]]:
    features_path = REFERENCE_ROOT / "reference_features.json"
    profiles_path = REFERENCE_ROOT / "component_vlm_profiles.json"
    features_payload = json.loads(features_path.read_text(encoding="utf-8"))
    profiles_payload = json.loads(profiles_path.read_text(encoding="utf-8"))
    profiles = {
        str(item.get("componentId")): item
        for item in profiles_payload.get("components", [])
        if isinstance(item, dict) and item.get("componentId")
    }
    components = features_payload.get("components", [])
    if limit_components:
        components = components[:limit_components]

    records: List[Dict[str, Any]] = []
    for component in components:
        component_id = str(component.get("componentId") or "")
        image_path = Path(str(component.get("imagePath") or ""))
        if not component_id or not image_path.exists():
            continue
        profile = profiles.get(component_id, {})
        visual_form = str(profile.get("visualForm") or ((component.get("features") or {}).get("structural") or {}).get("profileVisualForm") or "")
        for index, variant in enumerate(generate_image_variants(Image.open(image_path).convert("RGB"), variants)):
            variant_name = safe_filename(f"ref_{component_id}_{index:03d}.png")
            variant_path = image_output_dir / variant_name
            variant.save(variant_path)
            records.append(
                {
                    "id": variant_name.removesuffix(".png"),
                    "source": "component_reference",
                    "image": str(variant_path),
                    "componentId": component_id,
                    "visualForm": visual_form,
                    "ocrText": "",
                    "nearMisses": near_misses_for_component(component_id, profiles),
                }
            )
    return records


def build_correction_records() -> List[Dict[str, Any]]:
    label_path = DATASET_ROOT / "corrections" / "labels.jsonl"
    if not label_path.exists():
        return []
    records: List[Dict[str, Any]] = []
    for item in read_jsonl(label_path):
        image_path = DATASET_ROOT / str(item.get("image") or "")
        if not image_path.exists():
            continue
        records.append(
            {
                "id": item.get("id"),
                "source": item.get("source", "manual_correction"),
                "image": str(image_path),
                "componentId": item.get("correctComponentId"),
                "visualForm": item.get("visualForm") or "",
                "ocrText": item.get("ocrText") or item.get("modelText") or "",
                "nearMisses": [item.get("wrongComponentId")] if item.get("wrongComponentId") else [],
            }
        )
    return records


def to_qwen_message(record: Dict[str, Any]) -> Dict[str, Any]:
    answer = {
        "componentId": record.get("componentId"),
        "visualForm": record.get("visualForm") or "",
        "confidence": 0.98,
        "nearMisses": [item for item in record.get("nearMisses", []) if item],
    }
    ocr_text = str(record.get("ocrText") or "")
    prompt = PROMPT + (f"OCR文本：{ocr_text}" if ocr_text else "")
    image_name = Path(str(record.get("image") or "")).name
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": prompt},
                    {"image": image_name, "resized_width": 640, "resized_height": 420},
                ],
            },
            {"role": "assistant", "content": [{"text": json.dumps(answer, ensure_ascii=False)}]},
        ]
    }


def package_qwen_vl_dataset(data_path: Path, records: List[Dict[str, Any]], package_path: Path) -> None:
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(data_path, "data.jsonl")
        used_names = {"data.jsonl"}
        for record in records:
            image_path = Path(str(record.get("image") or ""))
            if not image_path.exists():
                continue
            image_name = image_path.name
            if image_name in used_names:
                continue
            used_names.add(image_name)
            archive.write(image_path, image_name)


def generate_image_variants(image: Image.Image, count: int) -> Iterable[Image.Image]:
    count = max(1, int(count))
    base = ImageOps.contain(image, (640, 420), Image.Resampling.LANCZOS)
    for index in range(count):
        variant = base.copy()
        brightness = 0.86 + (index % 5) * 0.07
        contrast = 0.9 + ((index // 5) % 4) * 0.06
        color = 0.92 + ((index // 3) % 5) * 0.04
        variant = ImageEnhance.Brightness(variant).enhance(brightness)
        variant = ImageEnhance.Contrast(variant).enhance(contrast)
        variant = ImageEnhance.Color(variant).enhance(color)
        pad = 10 + (index % 6) * 6
        background = (8 + (index * 7) % 26, 16 + (index * 5) % 28, 28 + (index * 3) % 34)
        canvas = Image.new("RGB", (variant.width + pad * 2, variant.height + pad * 2), background)
        canvas.paste(variant, (pad, pad))
        yield canvas


def crop_image(image_path: Path, bbox: BBox) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    x1 = max(0, int(round(bbox.x)))
    y1 = max(0, int(round(bbox.y)))
    x2 = min(w, int(round(bbox.right)))
    y2 = min(h, int(round(bbox.bottom)))
    return image.crop((x1, y1, x2, y2))


def near_misses_for_component(component_id: str, profiles: Dict[str, Dict[str, Any]], limit: int = 4) -> List[str]:
    profile = profiles.get(component_id, {})
    content_type = profile.get("contentType")
    visual_form = profile.get("visualForm")
    misses = []
    for other_id, other in profiles.items():
        if other_id == component_id:
            continue
        if other.get("contentType") == content_type or other.get("visualForm") == visual_form:
            misses.append(other_id)
        if len(misses) >= limit:
            break
    return misses


def append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        yield json.loads(line)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample"
