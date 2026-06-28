#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Panel", "Title", "Chart", "Table", "Map", "MetricCard", "Border", "Decorate", "Filter", "Image"]
COLORS = {
    "Panel": (76, 201, 240, 230),
    "Title": (255, 206, 86, 235),
    "Chart": (88, 214, 141, 235),
    "Table": (255, 159, 64, 235),
    "Map": (153, 102, 255, 235),
    "MetricCard": (255, 99, 132, 235),
    "Border": (90, 160, 255, 235),
    "Decorate": (201, 203, 207, 220),
    "Filter": (75, 215, 200, 230),
    "Image": (14, 165, 233, 230),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export detector nodes from a parser run as a YOLO hard-case draft for human review.")
    parser.add_argument("--result", required=True, help="Path to result.json from a bad or interesting run.")
    parser.add_argument("--out", default=str(ROOT / "data" / "yolo-hardcase-drafts"))
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--min-conf", type=float, default=0.12)
    parser.add_argument("--include-low-conf", action="store_true")
    parser.add_argument(
        "--manual-box",
        action="append",
        default=[],
        metavar="TYPE:X,Y,W,H",
        help="Add a reviewed manual YOLO box, for example Image:610,95,700,650.",
    )
    parser.add_argument("--manual-only", action="store_true", help="Export only --manual-box labels.")
    args = parser.parse_args()

    result_path = Path(args.result)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    image_path = Path(str((result.get("imageMeta") or {}).get("path") or ""))
    if not image_path.exists():
        image_path = result_path.parent / image_path.name
    if not image_path.exists():
        raise SystemExit(f"Cannot find source image for {result_path}: {image_path}")

    out = Path(args.out)
    for kind in ["images", "labels", "preview"]:
        (out / kind / args.split).mkdir(parents=True, exist_ok=True)
    write_yolo_config(out)

    stem = f"{result.get('runId') or result_path.parent.name}_draft"
    target_image = out / "images" / args.split / f"{stem}{image_path.suffix.lower()}"
    shutil.copy2(image_path, target_image)

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    labels: list[str] = []
    reviewed_nodes: list[dict[str, Any]] = []
    if not args.manual_only:
        for node in result.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            node_type = str(node.get("type") or "")
            if node_type not in CLASSES:
                continue
            if not node.get("detection_id"):
                continue
            confidence = float(node.get("confidence") or 0.0)
            if not args.include_low_conf and confidence < args.min_conf:
                continue
            box = node.get("bbox") if isinstance(node.get("bbox"), dict) else {}
            label = yolo_label(box, node_type, width, height)
            if not label:
                continue
            labels.append(label)
            reviewed_nodes.append(
                {
                    "nodeId": node.get("node_id"),
                    "type": node_type,
                    "confidence": round(confidence, 4),
                    "bbox": box,
                    "note": "DRAFT: review this box before including it in training.",
                }
            )

    for index, value in enumerate(args.manual_box):
        node_type, box = parse_manual_box(value)
        label = yolo_label(box, node_type, width, height)
        if not label:
            raise SystemExit(f"Invalid manual box: {value}")
        labels.append(label)
        reviewed_nodes.append(
            {
                "nodeId": f"manual_{index:04d}",
                "type": node_type,
                "confidence": 1.0,
                "bbox": box,
                "note": "REVIEWED: manual hard-case box.",
            }
        )

    (out / "labels" / args.split / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    (out / "preview" / args.split / f"{stem}.json").write_text(
        json.dumps({"sourceResult": str(result_path), "image": str(target_image), "nodes": reviewed_nodes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    draw_preview(image, reviewed_nodes, out / "preview" / args.split / f"{stem}.png")
    print(json.dumps({"image": str(target_image), "labels": len(labels), "preview": str(out / "preview" / args.split / f"{stem}.png")}, ensure_ascii=False, indent=2))


def yolo_label(box: dict[str, Any], node_type: str, width: int, height: int) -> str:
    try:
        x = float(box.get("x") or 0)
        y = float(box.get("y") or 0)
        w = float(box.get("w") or 0)
        h = float(box.get("h") or 0)
    except (TypeError, ValueError):
        return ""
    if w <= 1 or h <= 1:
        return ""
    x = max(0.0, min(float(width - 1), x))
    y = max(0.0, min(float(height - 1), y))
    w = max(1.0, min(float(width) - x, w))
    h = max(1.0, min(float(height) - y, h))
    class_id = CLASSES.index(node_type)
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}"


def parse_manual_box(value: str) -> tuple[str, dict[str, float]]:
    separator = ":" if ":" in value else ","
    if separator == ":":
        node_type, raw_numbers = value.split(":", 1)
        parts = raw_numbers.split(",")
    else:
        parts = value.split(",")
        node_type, parts = parts[0], parts[1:]
    node_type = node_type.strip()
    if node_type not in CLASSES:
        raise SystemExit(f"Unknown manual box type {node_type!r}; expected one of {', '.join(CLASSES)}")
    if len(parts) != 4:
        raise SystemExit(f"Manual box must be TYPE:x,y,w,h, got: {value}")
    try:
        x, y, w, h = [float(part.strip()) for part in parts]
    except ValueError as exc:
        raise SystemExit(f"Manual box has non-numeric values: {value}") from exc
    return node_type, {"x": x, "y": y, "w": w, "h": h}


def draw_preview(image: Image.Image, nodes: list[dict[str, Any]], out: Path) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    for item in nodes:
        box = item.get("bbox") if isinstance(item.get("bbox"), dict) else {}
        node_type = str(item.get("type") or "")
        color = COLORS.get(node_type, (255, 255, 255, 230))
        x = float(box.get("x") or 0)
        y = float(box.get("y") or 0)
        w = float(box.get("w") or 0)
        h = float(box.get("h") or 0)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=4)
        label = f"{item.get('nodeId')} {node_type} {item.get('confidence')}"
        draw.rectangle([x, max(0, y - 20), x + min(360, len(label) * 8), y], fill=(0, 0, 0, 180))
        draw.text((x + 4, max(0, y - 18)), label, fill=(255, 255, 255, 255))
    image.save(out)


def write_yolo_config(out: Path) -> None:
    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASSES))
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
