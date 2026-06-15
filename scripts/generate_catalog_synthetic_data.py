#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.component_library import ComponentLibrary
from app.schemas import BBox, ComponentRecord


CHART_CATEGORIES = {"Bars", "Lines", "Pies", "Scatters", "Areas", "Funnels", "WordClouds", "FlowChart"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate catalog-level synthetic screen data with componentId labels.")
    parser.add_argument("--catalog", default=str(ROOT.parent / "ai-schema-view" / "schema-md" / "catalog.md"))
    parser.add_argument("--out", default=str(ROOT / "data" / "synthetic-catalog"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--train-repeats", type=int, default=8)
    parser.add_argument("--val-repeats", type=int, default=2)
    parser.add_argument("--components-per-image", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sketch", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    library = ComponentLibrary.from_catalog(args.catalog)
    records = library.records
    if not records:
        raise SystemExit(f"No component records found: {args.catalog}")

    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    prepare_dirs(out)

    classes = [record.key for record in records]
    class_to_id = {name: index for index, name in enumerate(classes)}

    train_count = generate_split(
        out=out,
        split="train",
        records=records,
        class_to_id=class_to_id,
        width=args.width,
        height=args.height,
        repeats=args.train_repeats,
        components_per_image=args.components_per_image,
        seed=args.seed,
        sketch=args.sketch,
    )
    val_count = generate_split(
        out=out,
        split="val",
        records=records,
        class_to_id=class_to_id,
        width=args.width,
        height=args.height,
        repeats=args.val_repeats,
        components_per_image=args.components_per_image,
        seed=args.seed + 1,
        sketch=args.sketch,
    )

    (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "\n".join([f"  {index}: {name}" for index, name in enumerate(classes)])
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "catalog": str(Path(args.catalog).resolve()),
        "out": str(out.resolve()),
        "componentCount": len(records),
        "trainImages": train_count,
        "valImages": val_count,
        "trainRepeatsPerComponent": args.train_repeats,
        "valRepeatsPerComponent": args.val_repeats,
        "classes": classes,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def prepare_dirs(out: Path) -> None:
    for split in ["train", "val"]:
        for kind in ["images", "labels", "meta"]:
            (out / kind / split).mkdir(parents=True, exist_ok=True)


def generate_split(
    out: Path,
    split: str,
    records: List[ComponentRecord],
    class_to_id: Dict[str, int],
    width: int,
    height: int,
    repeats: int,
    components_per_image: int,
    seed: int,
    sketch: bool,
) -> int:
    rng = random.Random(seed)
    items: List[ComponentRecord] = []
    for _ in range(repeats):
        shuffled = list(records)
        rng.shuffle(shuffled)
        items.extend(shuffled)

    image_count = int(math.ceil(len(items) / float(components_per_image)))
    for image_index in range(image_count):
        chunk = items[image_index * components_per_image : (image_index + 1) * components_per_image]
        image, labels, meta = generate_one(chunk, class_to_id, width, height, rng, sketch)
        stem = f"catalog_{image_index:05d}"
        image.save(out / "images" / split / f"{stem}.png")
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        (out / "meta" / split / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_count


def generate_one(
    records: List[ComponentRecord],
    class_to_id: Dict[str, int],
    width: int,
    height: int,
    rng: random.Random,
    sketch: bool,
) -> Tuple[Image.Image, List[str], dict]:
    bg = (245, 247, 250) if sketch else rng.choice([(6, 14, 30), (9, 18, 36), (10, 22, 42)])
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    labels: List[str] = []
    nodes = []
    placed: List[BBox] = []

    draw_screen_chrome(draw, width, height, sketch, rng)

    for index, record in enumerate(records):
        bbox = place_bbox(record, width, height, placed, rng)
        placed.append(bbox)
        draw_component(draw, bbox, record, sketch, rng)
        add_label(labels, bbox, class_to_id[record.key], width, height)
        nodes.append(
            {
                "nodeId": f"{record.key}_{index:03d}",
                "parentId": "screen_0000",
                "type": coarse_type(record),
                "componentId": record.key,
                "level": level_for(record),
                "bbox": bbox.to_dict(),
            }
        )

    return image, labels, {"width": width, "height": height, "nodes": nodes}


def draw_screen_chrome(draw: ImageDraw.ImageDraw, width: int, height: int, sketch: bool, rng: random.Random) -> None:
    if sketch:
        draw.rectangle([12, 12, width - 12, height - 12], outline=(55, 65, 81), width=2)
        draw.line([width * 0.32, 44, width * 0.68, 44], fill=(55, 65, 81), width=2)
        return
    for y in range(0, height, 48):
        color = (12, 34, 60) if y % 96 == 0 else (10, 26, 48)
        draw.line([0, y, width, y], fill=color, width=1)
    draw.rectangle([10, 10, width - 10, height - 10], outline=(19, 86, 132), width=1)


def place_bbox(record: ComponentRecord, width: int, height: int, placed: List[BBox], rng: random.Random) -> BBox:
    for _ in range(80):
        w, h = suggested_size(record, width, height, rng)
        x = rng.uniform(24, max(25, width - w - 24))
        y = rng.uniform(54, max(55, height - h - 24))
        bbox = BBox(x, y, w, h)
        if max_overlap_ratio(bbox, placed) < 0.18:
            return bbox
    w, h = suggested_size(record, width, height, rng)
    return BBox(rng.uniform(18, max(19, width - w - 18)), rng.uniform(48, max(49, height - h - 18)), w, h)


def suggested_size(record: ComponentRecord, width: int, height: int, rng: random.Random) -> Tuple[float, float]:
    category = record.category
    key = record.key.lower()
    if category == "Borders":
        return rng.uniform(width * 0.22, width * 0.36), rng.uniform(height * 0.22, height * 0.38)
    if category in {"Title", "Texts"}:
        return rng.uniform(width * 0.16, width * 0.34), rng.uniform(28, 58)
    if category == "Tables":
        return rng.uniform(width * 0.22, width * 0.36), rng.uniform(height * 0.18, height * 0.32)
    if category in {"Maps", "Biz", "Three"}:
        return rng.uniform(width * 0.22, width * 0.42), rng.uniform(height * 0.22, height * 0.42)
    if category == "Inputs":
        return rng.uniform(width * 0.12, width * 0.26), rng.uniform(30, 76)
    if category in {"Decorates", "FlowChart"}:
        return rng.uniform(width * 0.12, width * 0.32), rng.uniform(24, 80)
    if "clock" in key or "circle" in key or "dial" in key or "water" in key:
        side = rng.uniform(64, 130)
        return side, side
    if category in CHART_CATEGORIES or category == "Mores":
        return rng.uniform(width * 0.18, width * 0.34), rng.uniform(height * 0.16, height * 0.31)
    return rng.uniform(width * 0.14, width * 0.3), rng.uniform(height * 0.12, height * 0.26)


def max_overlap_ratio(bbox: BBox, others: Iterable[BBox]) -> float:
    best = 0.0
    for other in others:
        ix1 = max(bbox.x, other.x)
        iy1 = max(bbox.y, other.y)
        ix2 = min(bbox.right, other.right)
        iy2 = min(bbox.bottom, other.bottom)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        best = max(best, inter / max(min(bbox.area, other.area), 1.0))
    return best


def draw_component(draw: ImageDraw.ImageDraw, bbox: BBox, record: ComponentRecord, sketch: bool, rng: random.Random) -> None:
    category = record.category
    key = record.key.lower()
    base = component_color(record.key, sketch)
    x1, y1, x2, y2 = bbox.x, bbox.y, bbox.right, bbox.bottom

    if category == "Borders":
        draw_border(draw, bbox, base, sketch)
    elif category in {"Title", "Texts"}:
        draw_title(draw, bbox, record, base, sketch)
    elif category == "Tables":
        draw_table(draw, bbox, record, base, sketch)
    elif category in {"Bars"} or "bar" in key or "capsule" in key or "cylinder" in key:
        draw_panel(draw, bbox, base, sketch)
        draw_bars(draw, bbox, base, rng)
    elif category in {"Lines", "Areas"} or "line" in key or "area" in key:
        draw_panel(draw, bbox, base, sketch)
        draw_line_chart(draw, bbox, base, rng, fill_area=category == "Areas" or "area" in key)
    elif category in {"Pies"} or "pie" in key:
        draw_panel(draw, bbox, base, sketch)
        draw_pie(draw, bbox, base)
    elif category in {"Maps", "Biz", "Three"} or "map" in key or "earth" in key:
        draw_panel(draw, bbox, base, sketch)
        draw_map(draw, bbox, base, rng)
    elif category == "Inputs":
        draw_input(draw, bbox, record, base, sketch)
    elif category in {"Decorates", "FlowChart"} or "decorate" in key or "pipeline" in key:
        draw_decorate(draw, bbox, base, rng, sketch)
    elif "clock" in key or "dial" in key:
        draw_clock(draw, bbox, base, sketch)
    elif "funnel" in key:
        draw_panel(draw, bbox, base, sketch)
        draw_funnel(draw, bbox, base)
    elif "number" in key or "energy" in key or "flipper" in key or "status" in key:
        draw_metric(draw, bbox, record, base, sketch)
    elif "wordcloud" in key:
        draw_panel(draw, bbox, base, sketch)
        draw_wordcloud(draw, bbox, base, rng)
    elif "heatmap" in key:
        draw_panel(draw, bbox, base, sketch)
        draw_heatmap(draw, bbox, base)
    else:
        draw_panel(draw, bbox, base, sketch)
        draw.text((x1 + 8, y1 + 8), short_label(record), fill=base)

    if not sketch and category not in {"Title", "Texts", "Inputs"}:
        draw.text((x1 + 6, max(0, y1 - 14)), short_label(record), fill=(180, 220, 255))
    elif sketch:
        draw.text((x1 + 5, max(0, y1 - 13)), short_label(record), fill=(31, 41, 55))


def draw_panel(draw: ImageDraw.ImageDraw, bbox: BBox, color: Tuple[int, int, int], sketch: bool) -> None:
    fill = None if sketch else (8, 26, 50)
    draw.rectangle([bbox.x, bbox.y, bbox.right, bbox.bottom], outline=color, fill=fill, width=2)


def draw_border(draw: ImageDraw.ImageDraw, bbox: BBox, color: Tuple[int, int, int], sketch: bool) -> None:
    fill = None if sketch else (8, 24, 45)
    draw.rectangle([bbox.x, bbox.y, bbox.right, bbox.bottom], outline=color, fill=fill, width=2)
    corner = min(bbox.w, bbox.h) * 0.18
    for sx in [bbox.x, bbox.right]:
        for sy in [bbox.y, bbox.bottom]:
            ex = sx + corner if sx == bbox.x else sx - corner
            ey = sy + corner if sy == bbox.y else sy - corner
            draw.line([sx, sy, ex, sy], fill=color, width=4)
            draw.line([sx, sy, sx, ey], fill=color, width=4)


def draw_title(draw: ImageDraw.ImageDraw, bbox: BBox, record: ComponentRecord, color: Tuple[int, int, int], sketch: bool) -> None:
    fill = None if sketch else (7, 30, 56)
    draw.rectangle([bbox.x, bbox.y, bbox.right, bbox.bottom], outline=color, fill=fill, width=2)
    text = record.title[:12] if record.title else record.key
    draw.text((bbox.x + 10, bbox.y + max(6, bbox.h * 0.28)), text, fill=color)
    draw.line([bbox.x + 8, bbox.bottom - 6, bbox.right - 8, bbox.bottom - 6], fill=color, width=2)


def draw_table(draw: ImageDraw.ImageDraw, bbox: BBox, record: ComponentRecord, color: Tuple[int, int, int], sketch: bool) -> None:
    draw_panel(draw, bbox, color, sketch)
    rows = 5
    cols = 4
    for row in range(1, rows):
        y = bbox.y + row * bbox.h / rows
        draw.line([bbox.x + 6, y, bbox.right - 6, y], fill=color, width=1)
    for col in range(1, cols):
        x = bbox.x + col * bbox.w / cols
        draw.line([x, bbox.y + 6, x, bbox.bottom - 6], fill=color, width=1)
    draw.text((bbox.x + 8, bbox.y + 8), record.title[:8], fill=color)


def draw_bars(draw: ImageDraw.ImageDraw, bbox: BBox, color: Tuple[int, int, int], rng: random.Random) -> None:
    bars = 7
    step = bbox.w / (bars + 2)
    for index in range(bars):
        bar_h = rng.uniform(0.22, 0.78) * bbox.h
        x = bbox.x + step * (index + 1)
        draw.rectangle([x, bbox.bottom - 14 - bar_h, x + step * 0.45, bbox.bottom - 14], fill=color)


def draw_line_chart(draw: ImageDraw.ImageDraw, bbox: BBox, color: Tuple[int, int, int], rng: random.Random, fill_area: bool = False) -> None:
    points = []
    for index in range(7):
        x = bbox.x + 16 + index * (bbox.w - 32) / 6
        y = bbox.y + rng.uniform(0.25, 0.75) * bbox.h
        points.append((x, y))
    if fill_area and len(points) > 1:
        poly = points + [(bbox.right - 16, bbox.bottom - 14), (bbox.x + 16, bbox.bottom - 14)]
        draw.polygon(poly, fill=soft_color(color))
    draw.line(points, fill=color, width=3)
    for point in points:
        r = 3
        draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r], fill=color)


def draw_pie(draw: ImageDraw.ImageDraw, bbox: BBox, color: Tuple[int, int, int]) -> None:
    side = min(bbox.w, bbox.h) * 0.58
    x1 = bbox.x + (bbox.w - side) / 2
    y1 = bbox.y + (bbox.h - side) / 2
    box = [x1, y1, x1 + side, y1 + side]
    draw.pieslice(box, 0, 110, fill=color)
    draw.pieslice(box, 110, 230, fill=soft_color(color))
    draw.pieslice(box, 230, 360, fill=(70, 130, 180))
    draw.ellipse([x1 + side * 0.28, y1 + side * 0.28, x1 + side * 0.72, y1 + side * 0.72], fill=(8, 26, 50))


def draw_map(draw: ImageDraw.ImageDraw, bbox: BBox, color: Tuple[int, int, int], rng: random.Random) -> None:
    points = [(bbox.x + rng.random() * bbox.w, bbox.y + rng.random() * bbox.h) for _ in range(9)]
    draw.line(points + [points[0]], fill=color, width=2)
    for point in points[:5]:
        r = rng.uniform(3, 8)
        draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r], outline=color, width=2)


def draw_input(draw: ImageDraw.ImageDraw, bbox: BBox, record: ComponentRecord, color: Tuple[int, int, int], sketch: bool) -> None:
    fill = None if sketch else (12, 31, 55)
    draw.rectangle([bbox.x, bbox.y, bbox.right, bbox.bottom], outline=color, fill=fill, width=2)
    draw.text((bbox.x + 8, bbox.y + max(6, bbox.h * 0.32)), record.title[:10], fill=color)
    draw.line([bbox.right - 22, bbox.y + bbox.h * 0.45, bbox.right - 14, bbox.y + bbox.h * 0.58], fill=color, width=2)
    draw.line([bbox.right - 14, bbox.y + bbox.h * 0.58, bbox.right - 6, bbox.y + bbox.h * 0.45], fill=color, width=2)


def draw_decorate(draw: ImageDraw.ImageDraw, bbox: BBox, color: Tuple[int, int, int], rng: random.Random, sketch: bool) -> None:
    draw.rectangle([bbox.x, bbox.y, bbox.right, bbox.bottom], outline=color, width=1)
    for index in range(5):
        y = bbox.y + (index + 1) * bbox.h / 6
        draw.line([bbox.x + 8, y, bbox.right - 8, y + rng.uniform(-8, 8)], fill=color, width=2)
    for index in range(4):
        x = bbox.x + rng.uniform(12, bbox.w - 12)
        y = bbox.y + rng.uniform(8, bbox.h - 8)
        r = 3 if sketch else 5
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def draw_clock(draw: ImageDraw.ImageDraw, bbox: BBox, color: Tuple[int, int, int], sketch: bool) -> None:
    side = min(bbox.w, bbox.h)
    cx = bbox.x + bbox.w / 2
    cy = bbox.y + bbox.h / 2
    r = side * 0.42
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)
    draw.line([cx, cy, cx, cy - r * 0.6], fill=color, width=3)
    draw.line([cx, cy, cx + r * 0.55, cy + r * 0.28], fill=color, width=3)


def draw_funnel(draw: ImageDraw.ImageDraw, bbox: BBox, color: Tuple[int, int, int]) -> None:
    levels = 4
    for index in range(levels):
        top = bbox.y + 18 + index * (bbox.h - 36) / levels
        bottom = bbox.y + 18 + (index + 0.72) * (bbox.h - 36) / levels
        ratio_top = 1.0 - index * 0.17
        ratio_bottom = 1.0 - (index + 1) * 0.17
        x_top = bbox.x + (bbox.w * (1 - ratio_top)) / 2
        x_bottom = bbox.x + (bbox.w * (1 - ratio_bottom)) / 2
        draw.polygon(
            [(x_top, top), (bbox.right - x_top + bbox.x, top), (bbox.right - x_bottom + bbox.x, bottom), (x_bottom, bottom)],
            fill=soft_color(color) if index % 2 else color,
        )


def draw_metric(draw: ImageDraw.ImageDraw, bbox: BBox, record: ComponentRecord, color: Tuple[int, int, int], sketch: bool) -> None:
    draw_panel(draw, bbox, color, sketch)
    draw.text((bbox.x + 10, bbox.y + 10), record.title[:10], fill=color)
    draw.text((bbox.x + 10, bbox.y + bbox.h * 0.45), "128,960", fill=color)
    draw.line([bbox.x + 10, bbox.bottom - 12, bbox.right - 10, bbox.bottom - 12], fill=color, width=3)


def draw_wordcloud(draw: ImageDraw.ImageDraw, bbox: BBox, color: Tuple[int, int, int], rng: random.Random) -> None:
    words = ["DATA", "AI", "MAP", "SAFE", "LINK", "FLOW"]
    for word in words:
        draw.text((bbox.x + rng.uniform(8, bbox.w - 48), bbox.y + rng.uniform(8, bbox.h - 22)), word, fill=color)


def draw_heatmap(draw: ImageDraw.ImageDraw, bbox: BBox, color: Tuple[int, int, int]) -> None:
    cols = 7
    rows = 4
    cw = (bbox.w - 22) / cols
    ch = (bbox.h - 22) / rows
    for row in range(rows):
        for col in range(cols):
            shade = tuple(min(255, int(channel * (0.45 + (row + col) / 14))) for channel in color)
            x = bbox.x + 10 + col * cw
            y = bbox.y + 10 + row * ch
            draw.rectangle([x, y, x + cw - 2, y + ch - 2], fill=shade)


def component_color(key: str, sketch: bool) -> Tuple[int, int, int]:
    if sketch:
        return (31, 41, 55)
    palette = [
        (56, 189, 248),
        (34, 211, 238),
        (96, 165, 250),
        (45, 212, 191),
        (250, 204, 21),
        (248, 113, 113),
        (167, 139, 250),
        (74, 222, 128),
    ]
    return palette[sum(ord(ch) for ch in key) % len(palette)]


def soft_color(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return tuple(max(20, int(channel * 0.48)) for channel in color)


def add_label(labels: List[str], bbox: BBox, class_id: int, width: int, height: int) -> None:
    cx = (bbox.x + bbox.w / 2) / width
    cy = (bbox.y + bbox.h / 2) / height
    labels.append(f"{class_id} {cx:.6f} {cy:.6f} {bbox.w / width:.6f} {bbox.h / height:.6f}")


def coarse_type(record: ComponentRecord) -> str:
    category = record.category
    key = record.key.lower()
    if category == "Tables":
        return "Table"
    if category in {"Maps", "Biz", "Three"} or "map" in key or "earth" in key:
        return "Map"
    if category in {"Title", "Texts"}:
        return "Title"
    if category == "Borders":
        return "Border"
    if category == "Inputs":
        return "Filter"
    if category in CHART_CATEGORIES or any(token in key for token in ["chart", "bar", "line", "pie", "radar", "sankey", "scatter", "area", "funnel", "heatmap", "wordcloud", "graph"]):
        return "Chart"
    if any(token in key for token in ["number", "energy", "flipper", "status", "process", "water"]):
        return "MetricCard"
    return "Decorate"


def level_for(record: ComponentRecord) -> int:
    node_type = coarse_type(record)
    if node_type in {"Border", "Title", "Decorate"}:
        return 3
    if node_type == "Filter":
        return 4
    return 4


def short_label(record: ComponentRecord) -> str:
    return record.key[:18]


if __name__ == "__main__":
    main()
