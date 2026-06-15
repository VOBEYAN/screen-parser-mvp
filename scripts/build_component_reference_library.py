#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.component_library import ComponentLibrary
from app.visual_matcher import REFERENCE_FEATURES_FILE, extract_image_features_from_path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build visual reference library from ai-schema-view assets.")
    parser.add_argument("--catalog", default=str(ROOT.parent / "ai-schema-view" / "schema-md" / "catalog.md"))
    parser.add_argument(
        "--description-features",
        default=str(ROOT.parent / "ai-schema-view" / "schema-md" / "component-description-features.json"),
    )
    parser.add_argument("--image-root", default=str(ROOT.parent / "ai-schema-view" / "src" / "assets" / "images" / "chart"))
    parser.add_argument("--output", default=str(ROOT / "data" / "component-reference"))
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    feature_path = Path(args.description_features)
    image_root = Path(args.image_root)
    output_dir = Path(args.output)
    output_images = output_dir / "images"
    output_images.mkdir(parents=True, exist_ok=True)

    library = ComponentLibrary.from_catalog(catalog_path)
    description_features = load_json(feature_path)
    image_index = build_image_index(image_root)

    components: List[Dict[str, object]] = []
    missing: List[Dict[str, str]] = []

    for record in library.records:
        meta = description_features.get(record.key, {})
        image_name = str(meta.get("image", "")).strip()
        source_path = resolve_image_path(record.key, image_name, image_index)
        if not source_path:
            missing.append({"componentId": record.key, "title": record.title, "expectedImage": image_name})
            continue

        suffix = source_path.suffix.lower()
        target_path = output_images / f"{record.key}{suffix}"
        shutil.copy2(source_path, target_path)

        features = extract_image_features_from_path(str(target_path))
        components.append(
            {
                "componentId": record.key,
                "title": record.title,
                "category": record.category,
                "categoryName": record.category_name,
                "schema": record.schema,
                "description": record.description,
                "imagePath": str(target_path.resolve()),
                "sourceImagePath": str(source_path.resolve()),
                "sourceImageName": source_path.name,
                "features": features,
            }
        )

    payload = {
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "catalogPath": str(catalog_path.resolve()),
        "imageRoot": str(image_root.resolve()),
        "componentCount": len(components),
        "missingCount": len(missing),
        "missing": missing,
        "components": components,
    }
    output_json = output_dir / REFERENCE_FEATURES_FILE
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(output_json), "componentCount": len(components), "missingCount": len(missing)}, ensure_ascii=False, indent=2))


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_image_index(image_root: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    if not image_root.exists():
        return index
    for path in image_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        index.setdefault(path.name.lower(), []).append(path)
    return index


def resolve_image_path(component_id: str, image_name: str, image_index: Dict[str, List[Path]]) -> Optional[Path]:
    candidates = []
    if image_name:
        candidates.append(image_name)
    candidates.extend(
        [
            f"{component_id}.png",
            f"{component_id.lower()}.png",
            f"{snake_case(component_id)}.png",
            f"{kebab_case(component_id)}.png",
        ]
    )

    for candidate in candidates:
        paths = image_index.get(candidate.lower(), [])
        if paths:
            return sorted(paths, key=lambda item: len(str(item)))[0]
    return None


def snake_case(value: str) -> str:
    chars: List[str] = []
    for index, char in enumerate(value):
        if char.isupper() and index > 0 and value[index - 1].islower():
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


def kebab_case(value: str) -> str:
    return snake_case(value).replace("_", "-")


if __name__ == "__main__":
    main()
