#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import List, Tuple

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.schemas import BBox


CLASSES = ["Panel", "Title", "Chart", "Table", "Map", "MetricCard", "Border", "Decorate"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic screen images and YOLO labels.")
    parser.add_argument("--out", default=str(ROOT / "data" / "synthetic"))
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--sketch", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    image_train_dir = out / "images" / "train"
    image_val_dir = out / "images" / "val"
    label_train_dir = out / "labels" / "train"
    label_val_dir = out / "labels" / "val"
    meta_train_dir = out / "meta" / "train"
    meta_val_dir = out / "meta" / "val"
    for directory in [image_train_dir, image_val_dir, label_train_dir, label_val_dir, meta_train_dir, meta_val_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    for index in range(args.count):
        image, labels, meta = generate_one(args.width, args.height, sketch=args.sketch)
        stem = f"synthetic_{index:05d}"
        is_val = random.random() < args.val_ratio
        image_dir = image_val_dir if is_val else image_train_dir
        label_dir = label_val_dir if is_val else label_train_dir
        meta_dir = meta_val_dir if is_val else meta_train_dir
        image.save(image_dir / f"{stem}.png")
        (label_dir / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        (meta_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "\n".join([f"  {idx}: {name}" for idx, name in enumerate(CLASSES)])
        + "\n",
        encoding="utf-8",
    )
    print(f"Synthetic dataset generated: {out}")


def generate_one(width: int, height: int, sketch: bool = False) -> Tuple[Image.Image, List[str], dict]:
    bg = (9, 18, 36) if not sketch else (245, 247, 250)
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    labels: List[str] = []
    nodes = []

    title_box = BBox(width * 0.28, 24, width * 0.44, 56)
    draw_component(draw, title_box, "Title", sketch)
    add_label(labels, title_box, "Title", width, height)
    nodes.append(node_meta("title_000", None, title_box, "Title", 3))

    columns = random.choice([3, 4])
    rows = random.choice([2, 3])
    margin_x = 48
    margin_y = 110
    gap = 26
    cell_w = (width - margin_x * 2 - gap * (columns - 1)) / columns
    cell_h = (height - margin_y - 40 - gap * (rows - 1)) / rows

    panel_index = 0
    for row in range(rows):
        for col in range(columns):
            if random.random() < 0.12:
                continue
            x = margin_x + col * (cell_w + gap) + random.randint(-10, 10)
            y = margin_y + row * (cell_h + gap) + random.randint(-8, 8)
            w = cell_w + random.randint(-18, 18)
            h = cell_h + random.randint(-18, 18)
            panel = BBox(x, y, w, h)
            panel_id = f"panel_{panel_index:03d}"
            draw_component(draw, panel, "Panel", sketch)
            add_label(labels, panel, "Panel", width, height)
            nodes.append(node_meta(panel_id, "screen_0000", panel, "Panel", 2))

            inner_title = BBox(x + 18, y + 10, max(120, w * 0.42), 34)
            draw_component(draw, inner_title, "Title", sketch)
            add_label(labels, inner_title, "Title", width, height)
            nodes.append(node_meta(f"{panel_id}_title", panel_id, inner_title, "Title", 3))

            comp_type = random.choice(["Chart", "Table", "Map", "MetricCard"])
            content = BBox(x + 22, y + 58, w - 44, h - 78)
            draw_component(draw, content, comp_type, sketch)
            add_label(labels, content, comp_type, width, height)
            nodes.append(node_meta(f"{panel_id}_content", panel_id, content, comp_type, 4))
            panel_index += 1

    return image, labels, {"width": width, "height": height, "nodes": nodes}


def draw_component(draw: ImageDraw.ImageDraw, bbox: BBox, comp_type: str, sketch: bool) -> None:
    x1, y1, x2, y2 = bbox.x, bbox.y, bbox.right, bbox.bottom
    if sketch:
        line = (35, 48, 64)
        draw.rectangle([x1, y1, x2, y2], outline=line, width=3)
    else:
        draw.rectangle([x1, y1, x2, y2], outline=(39, 166, 220), fill=(12, 30, 58), width=2)

    if comp_type == "Chart":
        for i in range(6):
            bar_w = bbox.w / 10
            bx = x1 + 24 + i * bar_w * 1.35
            bh = random.uniform(0.25, 0.78) * bbox.h
            draw.rectangle([bx, y2 - 18 - bh, bx + bar_w, y2 - 18], fill=(59, 130, 246) if not sketch else None, outline=(59, 130, 246))
    elif comp_type == "Table":
        for i in range(1, 5):
            y = y1 + i * bbox.h / 5
            draw.line([x1 + 8, y, x2 - 8, y], fill=(96, 165, 250), width=1)
        for i in range(1, 4):
            x = x1 + i * bbox.w / 4
            draw.line([x, y1 + 8, x, y2 - 8], fill=(96, 165, 250), width=1)
    elif comp_type == "Map":
        points = [(x1 + bbox.w * random.random(), y1 + bbox.h * random.random()) for _ in range(14)]
        draw.line(points + [points[0]], fill=(45, 212, 191), width=3)
    elif comp_type == "MetricCard":
        pad = min(24, max(4, bbox.w / 8), max(4, bbox.h / 8))
        if x1 + pad < x2 - pad and y1 + pad < y2 - pad:
            draw.rectangle([x1 + pad, y1 + pad, x2 - pad, y2 - pad], outline=(250, 204, 21), width=3)


def add_label(labels: List[str], bbox: BBox, class_name: str, width: int, height: int) -> None:
    cls_id = CLASSES.index(class_name)
    cx = (bbox.x + bbox.w / 2) / width
    cy = (bbox.y + bbox.h / 2) / height
    labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bbox.w / width:.6f} {bbox.h / height:.6f}")


def node_meta(node_id: str, parent_id: str, bbox: BBox, node_type: str, level: int) -> dict:
    return {
        "nodeId": node_id,
        "parentId": parent_id,
        "type": node_type,
        "level": level,
        "bbox": bbox.to_dict(),
    }


if __name__ == "__main__":
    main()
