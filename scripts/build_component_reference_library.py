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
    parser.add_argument(
        "--vlm-profiles",
        default=str(ROOT / "data" / "component-reference" / "component_vlm_profiles.json"),
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
    vlm_profiles = load_component_profiles(Path(args.vlm_profiles))
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
        apply_reference_profile_features(features, vlm_profiles.get(record.key, {}))
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


def load_component_profiles(path: Path) -> Dict[str, Dict[str, object]]:
    payload = load_json(path)
    components = payload.get("components") if isinstance(payload, dict) else None
    if not isinstance(components, list):
        return {}
    profiles: Dict[str, Dict[str, object]] = {}
    for item in components:
        if not isinstance(item, dict):
            continue
        component_id = str(item.get("componentId") or "")
        if component_id:
            profiles[component_id] = item
    return profiles


def apply_reference_profile_features(features: Dict[str, object], profile: Dict[str, object]) -> None:
    visual_form = normalize_profile_form(str(profile.get("visualForm") or ""))
    content_type = str(profile.get("contentType") or "")
    if not visual_form and not content_type:
        return
    structural = dict(features.get("structural") or {})
    forms = [str(item) for item in structural.get("forms", []) if item]
    if visual_form and visual_form not in forms:
        forms.insert(0, visual_form)
    if content_type == "table" and "table_grid" not in forms:
        forms.insert(0, "table_grid")
    elif content_type == "pie_chart" and visual_form and visual_form not in forms:
        forms.insert(0, visual_form)
    elif content_type == "scatter_chart" and "scatter_plot" not in forms:
        forms.insert(0, "scatter_plot")

    structural["primaryForm"] = visual_form or content_type_to_form(content_type) or structural.get("primaryForm", "")
    structural["forms"] = forms
    structural["profileContentType"] = content_type
    structural["profileVisualForm"] = visual_form
    features["structural"] = structural


def normalize_profile_form(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "prism_bar": "isometric_prism_bar",
        "color_prism_bar": "isometric_prism_bar",
        "cylinder_bar": "cylinder_vertical_bar",
        "3d_cylinder_bar": "gradient_cylinder_bar",
        "liquid_bar": "liquid_vertical_bar",
        "pie_3d_exploded": "pie3d_exploded",
        "pie_3d_ring": "pie3d_ring",
        "scatter": "scatter_plot",
    }
    return aliases.get(normalized, normalized)


def content_type_to_form(content_type: str) -> str:
    return {
        "table": "table_grid",
        "line_chart": "line_chart",
        "area_chart": "line_gradient_area",
        "bar_chart": "vertical_bar",
        "pie_chart": "pie",
        "scatter_chart": "scatter_plot",
        "map": "map",
        "metric_card": "metric_card",
        "filter": "filter",
        "title": "title_text",
        "border": "border_frame",
        "panel": "panel",
    }.get(content_type, "")


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
