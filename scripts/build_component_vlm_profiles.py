#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.component_library import ComponentLibrary
from app.component_profiles import PROFILE_FILE, infer_component_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-scan ai-schema-view component previews into reusable VLM profiles.")
    parser.add_argument("--catalog", default=str(ROOT.parent / "ai-schema-view" / "schema-md" / "catalog.md"))
    parser.add_argument("--reference", default=str(ROOT / "data" / "component-reference"))
    parser.add_argument("--model", default=os.getenv("SCREEN_PARSER_VLM_MODEL") or os.getenv("OPENAI_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("SCREEN_PARSER_VLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1")
    parser.add_argument("--api-key", default=os.getenv("SCREEN_PARSER_VLM_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=float(os.getenv("SCREEN_PARSER_VLM_TIMEOUT", "25")))
    args = parser.parse_args()

    library = ComponentLibrary.from_catalog(args.catalog)
    reference_dir = Path(args.reference)
    output_path = reference_dir / PROFILE_FILE
    existing = load_existing(output_path)
    use_vlm = bool(args.model and args.api_key)

    components = []
    for index, record in enumerate(library.records):
        if args.limit and index >= args.limit:
            break
        cached = existing.get(record.key)
        if cached and cached.get("source") == "vlm":
            components.append(cached)
            print(f"[{index + 1}/{len(library.records)}] cached {record.key}", flush=True)
            continue

        image_path = resolve_reference_image(reference_dir, record.key)
        profile = infer_component_profile(record)
        if use_vlm and image_path:
            try:
                print(f"[{index + 1}/{len(library.records)}] vlm {record.key}", flush=True)
                profile = {**profile, **scan_with_vlm(record, image_path, args.model, args.base_url, args.api_key, args.timeout)}
                profile["source"] = "vlm"
            except Exception as exc:
                profile["source"] = "heuristic_after_vlm_error"
                profile["vlmError"] = str(exc)[:240]
                print(f"[{index + 1}/{len(library.records)}] fallback {record.key}: {profile['vlmError']}", flush=True)
        else:
            print(f"[{index + 1}/{len(library.records)}] heuristic {record.key}", flush=True)
        components.append(profile)
        write_payload(output_path, args.catalog, reference_dir, "vlm" if use_vlm else "heuristic", components)

    payload = write_payload(output_path, args.catalog, reference_dir, "vlm" if use_vlm else "heuristic", components)
    print(json.dumps({"output": str(output_path), "mode": payload["mode"], "componentCount": len(components)}, ensure_ascii=False, indent=2))


def write_payload(
    output_path: Path,
    catalog_path: str,
    reference_dir: Path,
    mode: str,
    components: list[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = {
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "catalogPath": str(Path(catalog_path).resolve()),
        "referencePath": str(reference_dir.resolve()),
        "mode": mode,
        "componentCount": len(components),
        "components": components,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_existing(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("components") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return {}
    return {str(item.get("componentId") or ""): item for item in items if isinstance(item, dict)}


def resolve_reference_image(reference_dir: Path, component_id: str) -> Optional[Path]:
    image_dir = reference_dir / "images"
    for suffix in [".png", ".jpg", ".jpeg", ".webp"]:
        path = image_dir / f"{component_id}{suffix}"
        if path.exists():
            return path
    return None


def scan_with_vlm(record, image_path: Path, model: str, base_url: str, api_key: str, timeout: float) -> Dict[str, Any]:
    prompt = {
        "task": "Analyze this ai-schema-view dashboard component preview and produce a reusable matching profile.",
        "componentId": record.key,
        "title": record.title,
        "category": record.category,
        "description": record.description,
        "allowedContentTypes": [
            "title",
            "table",
            "map",
            "metric_card",
            "filter",
            "bar_chart",
            "line_chart",
            "area_chart",
            "pie_chart",
            "scatter_chart",
            "funnel_chart",
            "wordcloud",
            "chart",
            "decorate",
            "panel",
            "border",
        ],
        "returnJsonSchema": {
            "componentId": record.key,
            "contentType": "best allowedContentTypes value",
            "visualForm": "specific visual form, e.g. liquid_vertical_bar, donut_pie, table_grid, border_frame",
            "layout": "horizontal|vertical|centered|grid|balanced|freeform",
            "semanticKeywords": "array of 5-12 Chinese/English matching keywords",
            "distinguishingFeatures": "array of visual traits that separate it from similar components",
            "negativeMatches": "array of component forms this should not match",
        },
    }
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "You create concise visual-semantic profiles for dashboard component library matching. Return only JSON.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(prompt, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
                ],
            },
        ],
    }
    endpoint = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"VLM profile API failed: HTTP {exc.code} {body[:240]}") from exc

    content = response_payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    parsed["componentId"] = record.key
    return parsed


def encode_image(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


if __name__ == "__main__":
    main()
