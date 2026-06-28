#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.component_library import ComponentLibrary
from app.schemas import BBox, ComponentRecord


DETECTION_CLASSES = ["Panel", "Title", "Chart", "Table", "Map", "MetricCard", "Border", "Decorate", "Filter", "Image"]
STRUCTURE_NODE_TYPES = ["Region", "Panel", "Title", "Border", "Content", "Chart", "Table", "Map", "MetricCard", "Decorate", "Filter", "Image"]
CHART_CATEGORIES = {"Bars", "Lines", "Pies", "Scatters", "Areas", "Funnels", "WordClouds", "FlowChart"}
TITLE_CATEGORIES = {"Title", "Texts"}
TABLE_CATEGORIES = {"Tables"}
MAP_CATEGORIES = {"Maps"}
MAP_COMPONENT_IDS = {"ChinaMap", "MapAmap", "MapBase"}
IMAGE_COMPONENT_IDS = {"AIRobot", "AIShield", "KeySecurity3D", "ThreeEarth01"}
IMAGE_CATEGORIES = {"Three"}
FILTER_CATEGORIES = {"Inputs"}
DECORATE_CATEGORIES = {"Decorates"}
SCREEN_TITLE_TEXTS = [
    "联通服务展示大屏",
    "智能监管检测系统大屏",
    "智能密码监管检测系统大屏",
    "数据可视化运营驾驶舱",
    "业务态势分析大屏",
    "智能中台",
    "智能底座",
]
PANEL_TITLE_TEXTS = [
    "服务分布",
    "平台分布",
    "实时调用",
    "调用分布指数",
    "学力分布",
    "用户调用量排行Top5",
    "安全告警",
    "攻击趋势",
    "攻击分布",
    "安全事件统计",
    "任务状态",
    "风险预警",
    "责任单位",
    "业务链梳理",
]
FONT_CANDIDATES = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]

LABEL_MODES = {"coarse", "component"}


class ComponentAsset:
    def __init__(self, record: ComponentRecord, image_path: Path):
        self.record = record
        self.image_path = image_path
        self.coarse_type = coarse_type(record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate composited screen training data from component reference images.")
    parser.add_argument("--catalog", default=str(ROOT.parent / "ai-schema-view" / "schema-md" / "catalog.md"))
    parser.add_argument("--reference-library", default=str(ROOT / "data" / "component-reference" / "reference_features.json"))
    parser.add_argument("--out", default=str(ROOT / "data" / "composited-screen"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--train-count", type=int, default=240)
    parser.add_argument("--val-count", type=int, default=60)
    parser.add_argument("--components-per-screen", type=int, default=8)
    parser.add_argument("--include-sketch", action="store_true")
    parser.add_argument("--label-mode", choices=sorted(LABEL_MODES), default="coarse")
    parser.add_argument("--layout-mode", choices=["grid", "mixed", "dense"], default="mixed")
    parser.add_argument("--title-placement-mode", choices=["center", "diverse"], default="diverse")
    parser.add_argument("--overlay-rate", type=float, default=0.35)
    parser.add_argument("--content-hints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hard-chart-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    prepare_dirs(out)

    library = ComponentLibrary.from_catalog(args.catalog)
    reference_paths = load_reference_paths(Path(args.reference_library))
    assets = build_assets(library.records, reference_paths)
    if not assets:
        raise SystemExit("No component reference assets found. Run scripts/build_component_reference_library.py first.")
    classes = build_classes(args.label_mode, assets)
    class_to_id = {name: index for index, name in enumerate(classes)}

    rng = random.Random(args.seed)
    train_summary = generate_split(
        out=out,
        split="train",
        assets=assets,
        width=args.width,
        height=args.height,
        screen_count=args.train_count,
        components_per_screen=args.components_per_screen,
        include_sketch=args.include_sketch,
        label_mode=args.label_mode,
        layout_mode=args.layout_mode,
        title_placement_mode=args.title_placement_mode,
        overlay_rate=args.overlay_rate,
        content_hints=args.content_hints,
        hard_chart_rate=args.hard_chart_rate,
        class_to_id=class_to_id,
        rng=random.Random(rng.randint(1, 10**9)),
    )
    val_summary = generate_split(
        out=out,
        split="val",
        assets=assets,
        width=args.width,
        height=args.height,
        screen_count=args.val_count,
        components_per_screen=args.components_per_screen,
        include_sketch=args.include_sketch,
        label_mode=args.label_mode,
        layout_mode=args.layout_mode,
        title_placement_mode=args.title_placement_mode,
        overlay_rate=args.overlay_rate,
        content_hints=args.content_hints,
        hard_chart_rate=args.hard_chart_rate,
        class_to_id=class_to_id,
        rng=random.Random(rng.randint(1, 10**9)),
    )

    write_yolo_config(out, classes)
    summary = {
        "out": str(out.resolve()),
        "catalog": str(Path(args.catalog).resolve()),
        "referenceLibrary": str(Path(args.reference_library).resolve()),
        "width": args.width,
        "height": args.height,
        "labelMode": args.label_mode,
        "layoutMode": args.layout_mode,
        "titlePlacementMode": args.title_placement_mode,
        "overlayRate": args.overlay_rate,
        "contentHints": args.content_hints,
        "hardChartRate": args.hard_chart_rate,
        "coarseClasses": DETECTION_CLASSES,
        "structureNodeTypes": STRUCTURE_NODE_TYPES,
        "classes": classes,
        "componentCount": len(assets),
        "train": train_summary,
        "val": val_summary,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def prepare_dirs(out: Path) -> None:
    for split in ["train", "val"]:
        for kind in ["images", "labels", "meta"]:
            (out / kind / split).mkdir(parents=True, exist_ok=True)
    (out / "preview").mkdir(parents=True, exist_ok=True)


def load_reference_paths(reference_json: Path) -> Dict[str, Path]:
    payload = json.loads(reference_json.read_text(encoding="utf-8"))
    result: Dict[str, Path] = {}
    for item in payload.get("components", []):
        component_id = item.get("componentId")
        image_path = item.get("imagePath")
        if component_id and image_path:
            path = resolve_reference_image_path(Path(str(image_path)), reference_json.parent, str(component_id))
            if path.exists():
                result[str(component_id)] = path
    return result


def resolve_reference_image_path(path: Path, reference_root: Path, component_id: str) -> Path:
    if path.exists():
        return path
    candidates = [
        reference_root / "images" / path.name,
        reference_root / "images" / f"{component_id}.png",
        reference_root / "images" / f"{component_id}.jpg",
        reference_root / "images" / f"{component_id}.jpeg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def build_assets(records: Iterable[ComponentRecord], reference_paths: Dict[str, Path]) -> List[ComponentAsset]:
    assets: List[ComponentAsset] = []
    for record in records:
        image_path = reference_paths.get(record.key)
        if image_path:
            assets.append(ComponentAsset(record, image_path))
    return assets


def build_classes(label_mode: str, assets: List[ComponentAsset]) -> List[str]:
    if label_mode == "coarse":
        return list(DETECTION_CLASSES)
    return [asset.record.key for asset in assets]


def generate_split(
    out: Path,
    split: str,
    assets: List[ComponentAsset],
    width: int,
    height: int,
    screen_count: int,
    components_per_screen: int,
    include_sketch: bool,
    label_mode: str,
    layout_mode: str,
    title_placement_mode: str,
    overlay_rate: float,
    content_hints: bool,
    hard_chart_rate: float,
    class_to_id: Dict[str, int],
    rng: random.Random,
) -> Dict[str, object]:
    class_counts = {name: 0 for name in DETECTION_CLASSES}
    node_type_counts = {name: 0 for name in STRUCTURE_NODE_TYPES}
    component_counts = {asset.record.key: 0 for asset in assets}
    stream = component_stream(assets, screen_count * components_per_screen, rng)
    overlay_stream = overlay_component_stream(stream, assets, rng)
    cursor = 0
    image_count = 0

    title_assets = [asset for asset in assets if asset.coarse_type == "Title"]
    border_assets = [asset for asset in assets if asset.coarse_type == "Border"]
    decorate_assets = [asset for asset in assets if asset.coarse_type == "Decorate"]
    chart_assets = [asset for asset in assets if asset.coarse_type == "Chart"]
    image_assets = [asset for asset in assets if asset.coarse_type == "Image"]

    for screen_index in range(screen_count):
        selected = stream[cursor : cursor + components_per_screen]
        overlay_selected = overlay_stream[cursor : cursor + components_per_screen]
        cursor += components_per_screen
        image, labels, meta = generate_one(
            selected=selected,
            overlay_selected=overlay_selected,
            title_assets=title_assets,
            border_assets=border_assets,
            decorate_assets=decorate_assets,
            chart_assets=chart_assets,
            image_assets=image_assets,
            width=width,
            height=height,
            label_mode=label_mode,
            layout_mode=layout_mode,
            title_placement_mode=title_placement_mode,
            overlay_rate=overlay_rate,
            content_hints=content_hints,
            hard_chart_rate=hard_chart_rate,
            class_to_id=class_to_id,
            rng=rng,
        )
        stem = f"{split}_{screen_index:05d}"
        save_sample(out, split, stem, image, labels, meta)
        draw_preview(image, meta, out / "preview" / f"{stem}.png")
        image_count += 1
        update_counts(meta, class_counts, component_counts, node_type_counts)

        if include_sketch:
            sketch = to_sketch(image)
            sketch_meta = copy.deepcopy(meta)
            sketch_meta["inputType"] = "sketch"
            sketch_stem = f"{stem}_sketch"
            save_sample(out, split, sketch_stem, sketch, labels, sketch_meta)
            if screen_index < 3:
                draw_preview(sketch, sketch_meta, out / "preview" / f"{sketch_stem}.png")
            image_count += 1
            update_counts(sketch_meta, class_counts, component_counts, node_type_counts)

    covered = [key for key, count in component_counts.items() if count > 0]
    missing = [key for key, count in component_counts.items() if count == 0]
    return {
        "screenCount": screen_count,
        "imageCount": image_count,
        "includeSketch": include_sketch,
        "classCounts": class_counts,
        "nodeTypeCounts": node_type_counts,
        "coveredComponentCount": len(covered),
        "missingComponentCount": len(missing),
        "missingComponents": missing,
        "componentCounts": component_counts,
    }


def component_stream(assets: List[ComponentAsset], total: int, rng: random.Random) -> List[ComponentAsset]:
    stream: List[ComponentAsset] = []
    while len(stream) < total:
        batch = list(assets)
        rng.shuffle(batch)
        stream.extend(batch)
    return stream[:total]


def overlay_component_stream(primary_stream: List[ComponentAsset], assets: List[ComponentAsset], rng: random.Random) -> List[ComponentAsset]:
    if not assets:
        return []
    shuffled = list(assets)
    rng.shuffle(shuffled)
    by_key = {asset.record.key: index for index, asset in enumerate(shuffled)}
    stream: List[ComponentAsset] = []
    for index, primary in enumerate(primary_stream):
        primary_index = by_key.get(primary.record.key, index % len(shuffled))
        offset = 1 + (index % max(1, len(shuffled) - 1))
        secondary = shuffled[(primary_index + offset) % len(shuffled)]
        if secondary.record.key == primary.record.key and len(shuffled) > 1:
            secondary = shuffled[(primary_index + offset + 1) % len(shuffled)]
        stream.append(secondary)
    return stream


def choose_bar_chart_asset(chart_assets: List[ComponentAsset], rng: random.Random) -> ComponentAsset:
    bar_assets = [
        asset
        for asset in chart_assets
        if asset.record.category == "Bars" or "bar" in asset.record.key.lower() or "柱" in asset.record.title
    ]
    return rng.choice(bar_assets or chart_assets)


def should_force_hard_chart(slot: BBox, image_width: int, hard_chart_rate: float, rng: random.Random) -> bool:
    if hard_chart_rate <= 0:
        return False
    side_slot = slot.x < image_width * 0.32 or slot.right > image_width * 0.68
    tall_enough = slot.h >= 105 and slot.w >= 150
    return side_slot and tall_enough and rng.random() < max(0.0, min(1.0, hard_chart_rate))


def should_draw_luminous_bar_chart(slot: BBox, image_width: int, hard_chart_rate: float, rng: random.Random) -> bool:
    if hard_chart_rate <= 0:
        return False
    side_slot = slot.x < image_width * 0.32 or slot.right > image_width * 0.68
    probability = hard_chart_rate if side_slot else hard_chart_rate * 0.35
    return rng.random() < max(0.0, min(1.0, probability))


def generate_one(
    selected: List[ComponentAsset],
    overlay_selected: List[ComponentAsset],
    title_assets: List[ComponentAsset],
    border_assets: List[ComponentAsset],
    decorate_assets: List[ComponentAsset],
    chart_assets: List[ComponentAsset],
    image_assets: List[ComponentAsset],
    width: int,
    height: int,
    label_mode: str,
    layout_mode: str,
    title_placement_mode: str,
    overlay_rate: float,
    content_hints: bool,
    hard_chart_rate: float,
    class_to_id: Dict[str, int],
    rng: random.Random,
) -> Tuple[Image.Image, List[str], Dict[str, object]]:
    image = create_background(width, height, rng)
    draw = ImageDraw.Draw(image, "RGBA")
    labels: List[str] = []
    nodes: List[Dict[str, object]] = []

    screen_title_style = "tech_banner" if title_placement_mode == "diverse" and rng.random() < 0.58 else "classic"
    title_box = tech_screen_title_box(width, height, rng) if screen_title_style == "tech_banner" else screen_title_box(width, height, title_placement_mode, rng)
    header_region_id = "region_header"
    add_synthetic_node(
        nodes,
        labels,
        header_region_id,
        "screen_0000",
        "Region",
        header_region_box(title_box, width, height),
        width,
        height,
        None,
        label_mode,
        class_to_id,
        role="headerRegion",
        emit_label=False,
    )
    screen_title_text = rng.choice(SCREEN_TITLE_TEXTS)
    draw_rendered_title(draw, title_box, screen_title_text, rng, prominent=True, style=screen_title_style)
    add_synthetic_node(
        nodes,
        labels,
        "node_title_0000",
        header_region_id,
        "Title",
        title_box,
        width,
        height,
        title_component_id(title_assets),
        label_mode,
        class_to_id,
        text=screen_title_text,
        role="screenTitle",
    )

    selected = list(selected)
    slots = layout_slots(len(selected), width, height, rng, mode=layout_mode)
    if image_assets and slots and layout_mode in {"mixed", "dense"} and rng.random() < 0.72:
        selected[central_visual_slot_index(slots, width, height)] = rng.choice(image_assets)
    slot_regions = build_slot_regions(slots, width, height)
    for region in slot_regions:
        add_synthetic_node(
            nodes,
            labels,
            str(region["nodeId"]),
            "screen_0000",
            "Region",
            region["bbox"],
            width,
            height,
            None,
            label_mode,
            class_to_id,
            role=str(region["role"]),
            emit_label=False,
        )

    for index, asset in enumerate(selected):
        slot = slots[index]
        if should_force_hard_chart(slot, width, hard_chart_rate, rng) and chart_assets:
            asset = choose_bar_chart_asset(chart_assets, rng)
        panel_id = f"panel_{index:04d}"
        region_id = region_for_slot(slot, slot_regions)
        panel_type = "Border" if border_assets and (label_mode == "component" or rng.random() < 0.55) else "Panel"
        if panel_type == "Border":
            border_asset = rng.choice(border_assets)
            paste_asset(image, border_asset, slot, rng, stretch=True)
            add_synthetic_node(
                nodes,
                labels,
                panel_id,
                region_id,
                "Border",
                slot,
                width,
                height,
                border_asset.record.key,
                label_mode,
                class_to_id,
            )
        else:
            draw_panel(draw, slot, rng)
            add_synthetic_node(nodes, labels, panel_id, region_id, "Panel", slot, width, height, None, label_mode, class_to_id)

        header = panel_title_box(slot, rng, title_placement_mode)
        header_style = "panel_strip" if header.w >= slot.w * 0.82 else "classic"
        header_text = rng.choice(PANEL_TITLE_TEXTS)
        draw_rendered_title(draw, header, header_text, rng, style=header_style)
        add_synthetic_node(
            nodes,
            labels,
            f"{panel_id}_title",
            panel_id,
            "Title",
            header,
            width,
            height,
            title_component_id(title_assets),
            label_mode,
            class_to_id,
            text=header_text,
            role="panelTitle",
        )

        content = content_box_for_slot(slot, header, rng)
        content_id = f"{panel_id}_content"
        add_synthetic_node(
            nodes,
            labels,
            content_id,
            panel_id,
            "Content",
            content,
            width,
            height,
            None,
            label_mode,
            class_to_id,
            emit_label=False,
        )
        pasted_content = paste_asset(image, asset, content, rng, stretch=asset.coarse_type in {"Chart", "Table", "Map", "Border", "Image"})
        add_node(nodes, labels, f"{content_id}_component", content_id, asset, pasted_content, width, height, label_mode, class_to_id)
        hard_chart_style = asset.coarse_type == "Chart" and should_draw_luminous_bar_chart(slot, width, hard_chart_rate, rng)
        if content_hints:
            if hard_chart_style:
                draw_luminous_bar_chart_hint(draw, pasted_content, rng)
            else:
                draw_content_hint(draw, pasted_content, asset, rng)

        if index < len(overlay_selected) and rng.random() < max(0.0, min(1.0, overlay_rate)):
            overlay_asset = overlay_selected[index]
            overlay_target = overlay_box(pasted_content, overlay_asset, rng)
            overlay_bbox = paste_asset(
                image,
                overlay_asset,
                overlay_target,
                rng,
                stretch=overlay_asset.coarse_type in {"Chart", "Table", "Map", "Image"},
            )
            if content_hints:
                draw_content_hint(draw, overlay_bbox, overlay_asset, rng, compact=True)
            add_node(
                nodes,
                labels,
                f"{content_id}_overlay",
                content_id,
                overlay_asset,
                overlay_bbox,
                width,
                height,
                label_mode,
                class_to_id,
            )

        if rng.random() < 0.28:
            decorate_box = small_decorate_box(slot, rng)
            if label_mode == "component" and decorate_assets:
                decorate_asset = rng.choice(decorate_assets)
                pasted_decorate = paste_asset(image, decorate_asset, decorate_box, rng, stretch=False)
                add_node(nodes, labels, f"{panel_id}_decorate", panel_id, decorate_asset, pasted_decorate, width, height, label_mode, class_to_id)
            else:
                draw_decorate(draw, decorate_box, rng)
                add_synthetic_node(
                    nodes,
                    labels,
                    f"{panel_id}_decorate",
                    panel_id,
                    "Decorate",
                    decorate_box,
                    width,
                    height,
                    None,
                    label_mode,
                    class_to_id,
                )

    if image_assets and layout_mode == "mixed" and rng.random() < 0.46:
        asset = rng.choice(image_assets)
        target = center_main_visual_box(width, height, rng)
        pasted_image = paste_asset(image, asset, target, rng, stretch=True)
        if content_hints:
            draw_image_hint(draw, pasted_image, asset, rng)
        add_node(
            nodes,
            labels,
            "node_main_image_0000",
            region_for_slot(target, slot_regions),
            asset,
            pasted_image,
            width,
            height,
            label_mode,
            class_to_id,
        )

    meta = {"width": width, "height": height, "inputType": "design", "nodes": nodes}
    return image.convert("RGB"), labels, meta


def create_background(width: int, height: int, rng: random.Random) -> Image.Image:
    top = rng.choice([(5, 12, 28), (8, 17, 36), (10, 18, 42)])
    bottom = rng.choice([(6, 28, 58), (13, 36, 68), (18, 26, 54)])
    image = Image.new("RGB", (width, height), top)
    pixels = image.load()
    for y in range(height):
        ratio = y / float(max(height - 1, 1))
        color = tuple(int(top[i] * (1 - ratio) + bottom[i] * ratio) for i in range(3))
        for x in range(width):
            pixels[x, y] = color

    draw = ImageDraw.Draw(image, "RGBA")
    for y in range(0, height, 42):
        draw.line([0, y, width, y], fill=(35, 90, 130, 45), width=1)
    for x in range(0, width, 64):
        draw.line([x, 0, x, height], fill=(35, 90, 130, 28), width=1)
    draw.rectangle([8, 8, width - 8, height - 8], outline=(45, 160, 220, 120), width=1)
    return image


def screen_title_box(width: int, height: int, mode: str, rng: random.Random) -> BBox:
    if mode == "center":
        return BBox(width * 0.28, 18.0, width * 0.44, 42.0)

    variants = [
        BBox(width * 0.24, 16.0, width * 0.52, rng.uniform(38.0, 52.0)),
        BBox(24.0, 18.0, width * rng.uniform(0.34, 0.46), rng.uniform(34.0, 48.0)),
        BBox(width * rng.uniform(0.52, 0.62), 18.0, width * rng.uniform(0.28, 0.38), rng.uniform(34.0, 48.0)),
        BBox(width * 0.31, rng.uniform(8.0, 30.0), width * 0.38, rng.uniform(32.0, 46.0)),
        BBox(width * 0.18, rng.uniform(46.0, 62.0), width * 0.64, rng.uniform(30.0, 42.0)),
    ]
    return rng.choice(variants)


def tech_screen_title_box(width: int, height: int, rng: random.Random) -> BBox:
    box_w = width * rng.uniform(0.58, 0.70)
    box_h = rng.uniform(max(28.0, height * 0.055), max(36.0, height * 0.088))
    x = (width - box_w) / 2.0 + rng.uniform(-width * 0.018, width * 0.018)
    y = rng.uniform(0.0, max(2.0, height * 0.012))
    return BBox(x, y, box_w, box_h)


def header_region_box(title_box: BBox, width: int, height: int) -> BBox:
    bottom = min(float(height), max(72.0, title_box.bottom + 14.0))
    return BBox(0.0, 0.0, float(width), bottom)


def build_slot_regions(slots: List[BBox], width: int, height: int) -> List[Dict[str, object]]:
    groups: Dict[str, List[BBox]] = {"left": [], "center": [], "right": []}
    for slot in slots:
        center_x = slot.x + slot.w / 2.0
        if center_x < width * 0.34:
            groups["left"].append(slot)
        elif center_x > width * 0.66:
            groups["right"].append(slot)
        else:
            groups["center"].append(slot)

    roles = {"left": "leftRegion", "center": "centerRegion", "right": "rightRegion"}
    regions: List[Dict[str, object]] = []
    for name in ["left", "center", "right"]:
        boxes = groups[name]
        if not boxes:
            continue
        regions.append(
            {
                "nodeId": f"region_{name}",
                "role": roles[name],
                "bbox": clamp_bbox(union_boxes(boxes, pad=8.0), width, height),
            }
        )
    return regions


def region_for_slot(slot: BBox, regions: List[Dict[str, object]]) -> str:
    if not regions:
        return "screen_0000"
    center_x, center_y = slot.center
    best_region = regions[0]
    best_score = float("inf")
    for region in regions:
        bbox = region["bbox"]
        if isinstance(bbox, BBox):
            inside = bbox.x <= center_x <= bbox.right and bbox.y <= center_y <= bbox.bottom
            distance = 0.0 if inside else abs((bbox.x + bbox.w / 2.0) - center_x) + abs((bbox.y + bbox.h / 2.0) - center_y)
            if distance < best_score:
                best_score = distance
                best_region = region
    return str(best_region["nodeId"])


def union_boxes(boxes: List[BBox], pad: float = 0.0) -> BBox:
    x1 = min(box.x for box in boxes) - pad
    y1 = min(box.y for box in boxes) - pad
    x2 = max(box.right for box in boxes) + pad
    y2 = max(box.bottom for box in boxes) + pad
    return BBox(x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1 = max(0.0, min(float(width - 1), bbox.x))
    y1 = max(0.0, min(float(height - 1), bbox.y))
    x2 = max(x1 + 1.0, min(float(width), bbox.right))
    y2 = max(y1 + 1.0, min(float(height), bbox.bottom))
    return BBox(x1, y1, x2 - x1, y2 - y1)


def layout_slots(count: int, width: int, height: int, rng: random.Random, mode: str = "grid") -> List[BBox]:
    if mode == "mixed" and count >= 7 and rng.random() < 0.45:
        return mixed_dashboard_slots(count, width, height, rng)

    cols = 4 if mode == "dense" or count > 9 else 3
    rows = int(math.ceil(count / float(cols)))
    margin_x = 24
    margin_top = 78
    margin_bottom = 20
    gap = 14
    cell_w = (width - margin_x * 2 - gap * (cols - 1)) / cols
    cell_h = (height - margin_top - margin_bottom - gap * (rows - 1)) / rows
    slots: List[BBox] = []
    for row in range(rows):
        for col in range(cols):
            if len(slots) >= count:
                break
            jitter = 12 if mode == "dense" else 6
            jitter_x = rng.uniform(-jitter, jitter)
            jitter_y = rng.uniform(-jitter * 0.75, jitter * 0.75)
            shrink_w = rng.uniform(0, 18 if mode == "dense" else 12)
            shrink_h = rng.uniform(0, 16 if mode == "dense" else 10)
            slots.append(
                BBox(
                    margin_x + col * (cell_w + gap) + jitter_x,
                    margin_top + row * (cell_h + gap) + jitter_y,
                    cell_w - shrink_w,
                    cell_h - shrink_h,
                )
            )
    rng.shuffle(slots)
    return slots


def mixed_dashboard_slots(count: int, width: int, height: int, rng: random.Random) -> List[BBox]:
    slots: List[BBox] = []
    margin_x = 22
    margin_top = 82
    margin_bottom = 22
    gap = 14
    side_w = width * rng.uniform(0.23, 0.29)
    center_w = width - margin_x * 2 - side_w * 2 - gap * 2
    usable_h = height - margin_top - margin_bottom
    side_rows = max(2, min(4, int(math.ceil((count - 1) / 2))))
    side_h = (usable_h - gap * (side_rows - 1)) / side_rows

    slots.append(BBox(margin_x + side_w + gap, margin_top, center_w, usable_h * rng.uniform(0.58, 0.74)))
    for side in [0, 1]:
        base_x = margin_x if side == 0 else margin_x + side_w + gap + center_w + gap
        for row in range(side_rows):
            if len(slots) >= count:
                break
            slots.append(
                BBox(
                    base_x + rng.uniform(-5, 5),
                    margin_top + row * (side_h + gap) + rng.uniform(-4, 4),
                    side_w + rng.uniform(-8, 8),
                    side_h + rng.uniform(-8, 6),
                )
            )

    bottom_y = margin_top + slots[0].h + gap
    bottom_h = max(48.0, height - bottom_y - margin_bottom)
    remaining = count - len(slots)
    bottom_cols = max(1, remaining)
    cell_w = (center_w - gap * (bottom_cols - 1)) / bottom_cols if bottom_cols else center_w
    for index in range(remaining):
        slots.append(
            BBox(
                margin_x + side_w + gap + index * (cell_w + gap),
                bottom_y + rng.uniform(-4, 4),
                cell_w + rng.uniform(-5, 5),
                bottom_h + rng.uniform(-5, 4),
            )
        )
    rng.shuffle(slots)
    return slots[:count]


def central_visual_slot_index(slots: List[BBox], width: int, height: int) -> int:
    target_x = width * 0.5
    target_y = height * 0.54
    best_index = 0
    best_score = -1.0
    for index, slot in enumerate(slots):
        cx, cy = slot.center
        distance = abs(cx - target_x) / max(width, 1) + abs(cy - target_y) / max(height, 1)
        area_score = slot.area / max(float(width * height), 1.0)
        score = area_score * 1.8 - distance
        if score > best_score:
            best_index = index
            best_score = score
    return best_index


def center_main_visual_box(width: int, height: int, rng: random.Random) -> BBox:
    box_w = width * rng.uniform(0.30, 0.48)
    box_h = height * rng.uniform(0.34, 0.58)
    x = (width - box_w) / 2.0 + rng.uniform(-width * 0.035, width * 0.035)
    y = height * rng.uniform(0.20, 0.28)
    return BBox(x, y, box_w, box_h)


def panel_title_box(slot: BBox, rng: random.Random, mode: str = "center") -> BBox:
    if mode == "diverse" and rng.random() < 0.52:
        inset = rng.uniform(0.0, 6.0)
        box_w = max(96.0, slot.w - inset * 2.0)
        box_h = min(34.0, max(19.0, slot.h * rng.uniform(0.12, 0.18)))
        return BBox(slot.x + inset, slot.y + rng.uniform(0.0, 7.0), box_w, box_h)

    width_ratio = rng.uniform(0.44, 0.78)
    box_w = min(slot.w - 18.0, max(108.0, slot.w * width_ratio))
    box_h = min(34.0, max(22.0, slot.h * rng.uniform(0.14, 0.2)))
    if mode == "diverse" and rng.random() < 0.26:
        x = slot.x + rng.choice([10.0, max(10.0, slot.w - box_w - 10.0)])
    elif rng.random() < 0.58:
        x = slot.x + (slot.w - box_w) / 2.0
    else:
        x = slot.x + rng.uniform(10.0, max(11.0, slot.w - box_w - 10.0))
    return BBox(x, slot.y + rng.uniform(7.0, 11.0), box_w, box_h)


def content_box_for_slot(slot: BBox, header: BBox, rng: random.Random) -> BBox:
    left_pad = rng.uniform(10.0, 18.0)
    right_pad = rng.uniform(10.0, 18.0)
    top_gap = rng.uniform(12.0, 22.0)
    bottom_pad = rng.uniform(10.0, 18.0)
    y = max(header.bottom + top_gap, slot.y + slot.h * rng.uniform(0.22, 0.28))
    return BBox(
        slot.x + left_pad,
        y,
        max(24.0, slot.w - left_pad - right_pad),
        max(24.0, slot.bottom - y - bottom_pad),
    )


def overlay_box(base: BBox, asset: ComponentAsset, rng: random.Random) -> BBox:
    category = asset.coarse_type
    if category in {"Title", "Filter", "Decorate"}:
        w = base.w * rng.uniform(0.34, 0.62)
        h = min(base.h * 0.26, rng.uniform(22.0, 42.0))
    elif category == "MetricCard":
        w = base.w * rng.uniform(0.28, 0.48)
        h = base.h * rng.uniform(0.26, 0.42)
    else:
        w = base.w * rng.uniform(0.38, 0.68)
        h = base.h * rng.uniform(0.34, 0.64)

    anchors = [
        (base.x + 8, base.y + 8),
        (base.right - w - 8, base.y + 8),
        (base.x + 8, base.bottom - h - 8),
        (base.right - w - 8, base.bottom - h - 8),
        (base.x + (base.w - w) / 2.0, base.y + (base.h - h) / 2.0),
    ]
    x, y = rng.choice(anchors)
    x += rng.uniform(-6, 6)
    y += rng.uniform(-6, 6)
    return BBox(x, y, max(16.0, w), max(14.0, h))


def title_component_id(title_assets: List[ComponentAsset]) -> Optional[str]:
    if not title_assets:
        return None
    by_key = {asset.record.key: asset for asset in title_assets}
    for key in ["title1", "TextCommon", "TextGradient", "TextBarrage"]:
        if key in by_key:
            return key
    return title_assets[0].record.key


def paste_asset(
    canvas: Image.Image,
    asset: ComponentAsset,
    target: BBox,
    rng: random.Random,
    stretch: bool = False,
) -> BBox:
    source = Image.open(asset.image_path).convert("RGBA")
    source = augment_asset(source, rng)

    if stretch:
        new_w = max(2, int(round(target.w)))
        new_h = max(2, int(round(target.h)))
    else:
        scale = min(target.w / max(source.width, 1), target.h / max(source.height, 1))
        scale *= rng.uniform(0.86, 1.0)
        new_w = max(2, int(round(source.width * scale)))
        new_h = max(2, int(round(source.height * scale)))

    resized = source.resize((new_w, new_h), Image.LANCZOS)
    x = int(round(target.x + (target.w - new_w) / 2.0))
    y = int(round(target.y + (target.h - new_h) / 2.0))
    canvas.alpha_composite(resized, (x, y)) if canvas.mode == "RGBA" else canvas.paste(resized, (x, y), resized)
    return BBox(float(x), float(y), float(new_w), float(new_h))


def augment_asset(source: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageEnhance.Brightness(source).enhance(rng.uniform(0.82, 1.16))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.86, 1.22))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.86, 1.18))
    alpha = image.getchannel("A")
    if rng.random() < 0.18:
        alpha = ImageEnhance.Brightness(alpha).enhance(rng.uniform(0.78, 0.96))
        image.putalpha(alpha)
    return image


def draw_panel(draw: ImageDraw.ImageDraw, bbox: BBox, rng: random.Random) -> None:
    fill = rng.choice([(8, 25, 50, 170), (10, 31, 62, 155), (12, 23, 45, 165)])
    outline = rng.choice([(36, 184, 234, 210), (70, 130, 255, 210), (35, 214, 198, 190)])
    draw.rectangle([bbox.x, bbox.y, bbox.right, bbox.bottom], fill=fill, outline=outline, width=2)
    corner = min(bbox.w, bbox.h) * 0.12
    draw.line([bbox.x, bbox.y, bbox.x + corner, bbox.y], fill=outline, width=3)
    draw.line([bbox.x, bbox.y, bbox.x, bbox.y + corner], fill=outline, width=3)
    draw.line([bbox.right, bbox.y, bbox.right - corner, bbox.y], fill=outline, width=3)
    draw.line([bbox.right, bbox.y, bbox.right, bbox.y + corner], fill=outline, width=3)


def draw_title_placeholder(draw: ImageDraw.ImageDraw, bbox: BBox) -> None:
    draw.rectangle([bbox.x, bbox.y, bbox.right, bbox.bottom], outline=(58, 182, 255, 210), fill=(6, 27, 54, 160), width=2)
    draw.line([bbox.x + 6, bbox.bottom - 5, bbox.right - 6, bbox.bottom - 5], fill=(252, 211, 77, 210), width=2)


def draw_rendered_title(
    draw: ImageDraw.ImageDraw,
    bbox: BBox,
    text: str,
    rng: random.Random,
    prominent: bool = False,
    style: str = "classic",
) -> None:
    if style == "tech_banner":
        draw_tech_banner_title(draw, bbox, text, rng)
        return
    if style == "panel_strip":
        draw_panel_strip_title(draw, bbox, text, rng)
        return

    fill = rng.choice([(7, 29, 68, 185), (6, 37, 78, 170), (11, 32, 63, 180)])
    outline = rng.choice([(41, 182, 246, 220), (80, 172, 255, 220), (34, 211, 238, 210)])
    accent = rng.choice([(252, 211, 77, 230), (56, 189, 248, 230), (45, 212, 191, 220)])
    draw.rectangle([bbox.x, bbox.y, bbox.right, bbox.bottom], fill=fill, outline=outline, width=2)

    notch = min(42.0, bbox.w * 0.16)
    mid_y = bbox.y + bbox.h * 0.52
    draw.line([bbox.x + 8, mid_y, bbox.x + notch, bbox.y + 3], fill=outline, width=2)
    draw.line([bbox.right - 8, mid_y, bbox.right - notch, bbox.y + 3], fill=outline, width=2)
    draw.line([bbox.x + 8, bbox.bottom - 5, bbox.right - 8, bbox.bottom - 5], fill=accent, width=2)

    max_size = int(max(12, bbox.h * (0.62 if prominent else 0.58)))
    font = fit_font(text, max_size=max_size, max_width=max(12, int(bbox.w - 18)))
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    text_x = bbox.x + (bbox.w - text_w) / 2.0
    text_y = bbox.y + (bbox.h - text_h) / 2.0 - 1
    glow = rng.choice([(18, 78, 145, 210), (4, 115, 158, 210), (0, 78, 132, 200)])
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((text_x + dx, text_y + dy), text, font=font, fill=glow)
    draw.text((text_x, text_y), text, font=font, fill=(238, 247, 255, 255))


def draw_tech_banner_title(draw: ImageDraw.ImageDraw, bbox: BBox, text: str, rng: random.Random) -> None:
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
    outline = rng.choice([(64, 195, 255, 235), (58, 150, 255, 230), (99, 216, 255, 230)])
    fill = rng.choice([(10, 54, 132, 215), (12, 63, 147, 205), (8, 45, 116, 215)])
    dark_fill = rng.choice([(2, 15, 35, 210), (4, 20, 47, 205)])
    center = [
        (x + w * 0.30, y + h * 0.18),
        (x + w * 0.70, y + h * 0.18),
        (x + w * 0.66, y + h * 0.86),
        (x + w * 0.50, y + h * 0.98),
        (x + w * 0.34, y + h * 0.86),
    ]
    left = [
        (x + w * 0.02, y + h * 0.05),
        (x + w * 0.26, y + h * 0.24),
        (x + w * 0.30, y + h * 0.78),
        (x + w * 0.11, y + h * 0.72),
        (x + w * 0.08, y + h * 0.52),
        (x, y + h * 0.46),
    ]
    right = [(x + w - (px - x), py) for px, py in left]
    draw.polygon(left, fill=dark_fill, outline=outline)
    draw.polygon(right, fill=dark_fill, outline=outline)
    draw.polygon(center, fill=fill, outline=outline)
    draw.line([x + w * 0.04, y + h * 0.50, x + w * 0.24, y + h * 0.66], fill=(0, 188, 255, 210), width=3)
    draw.line([x + w * 0.96, y + h * 0.50, x + w * 0.76, y + h * 0.66], fill=(0, 188, 255, 210), width=3)
    for base_x in [x + w * 0.25, x + w * 0.73]:
        for index in range(4):
            x1 = base_x + index * min(10.0, w * 0.012)
            draw.polygon(
                [(x1, y + h * 0.48), (x1 + w * 0.012, y + h * 0.50), (x1 + w * 0.018, y + h * 0.60), (x1 + w * 0.006, y + h * 0.58)],
                fill=(0, 238, 255, 210),
            )

    font = fit_font(text, max_size=int(max(14, h * 0.48)), max_width=int(max(24, w * 0.44)))
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    text_x = x + (w - text_w) / 2.0
    text_y = y + h * 0.20 + (h * 0.52 - text_h) / 2.0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((text_x + dx, text_y + dy), text, font=font, fill=(0, 64, 132, 230))
    draw.text((text_x, text_y), text, font=font, fill=(236, 247, 255, 255))


def draw_panel_strip_title(draw: ImageDraw.ImageDraw, bbox: BBox, text: str, rng: random.Random) -> None:
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
    top = y + h * 0.12
    bottom = y + h * 0.88
    fill = rng.choice([(5, 39, 86, 145), (8, 50, 105, 135), (4, 34, 78, 150)])
    line = rng.choice([(0, 159, 255, 230), (48, 194, 255, 230), (0, 220, 255, 220)])
    draw.rectangle([x, top, x + w, bottom], fill=fill)
    draw.line([x, top, x + w, top], fill=line, width=2)
    draw.line([x, bottom, x + w, bottom], fill=(0, 127, 220, 170), width=1)
    draw.line([x + w * 0.18, y + h * 0.03, x + w * 0.95, y + h * 0.03], fill=(70, 195, 255, 115), width=1)

    icon_cx = x + min(max(18.0, h * 0.72), 28.0)
    icon_cy = y + h * 0.50
    radius = max(5.5, min(9.0, h * 0.25))
    hex_points = [
        (icon_cx + math.cos(math.pi / 3 * i + math.pi / 6) * radius, icon_cy + math.sin(math.pi / 3 * i + math.pi / 6) * radius)
        for i in range(6)
    ]
    draw.polygon(hex_points, fill=(7, 67, 124, 230), outline=line)
    draw.ellipse([icon_cx - radius * 0.35, icon_cy - radius * 0.35, icon_cx + radius * 0.35, icon_cy + radius * 0.35], fill=(39, 220, 255, 220))

    text_x = x + min(42.0, max(30.0, h * 1.28))
    max_width = int(max(20.0, w - (text_x - x) - 10.0))
    font = fit_font(text, max_size=int(max(12.0, h * 0.48)), max_width=max_width)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_h = text_bbox[3] - text_bbox[1]
    text_y = y + (h - text_h) / 2.0 - 1
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((text_x + dx, text_y + dy), text, font=font, fill=(0, 59, 105, 220))
    draw.text((text_x, text_y), text, font=font, fill=(185, 236, 255, 255))


def draw_content_hint(draw: ImageDraw.ImageDraw, bbox: BBox, asset: ComponentAsset, rng: random.Random, compact: bool = False) -> None:
    if bbox.w < 28 or bbox.h < 22:
        return

    key = asset.record.key.lower()
    category = asset.record.category
    node_type = asset.coarse_type
    cyan = (96, 225, 255, 205)
    green = (85, 230, 160, 205)
    yellow = (255, 218, 92, 215)
    muted = (160, 205, 235, 155)

    if node_type == "Title":
        text = rng.choice(SCREEN_TITLE_TEXTS + PANEL_TITLE_TEXTS)
        draw_rendered_title(draw, bbox, text, rng, prominent=bbox.w > 220 and not compact)
        return

    if node_type == "Table" or category == "Tables":
        draw_table_hint(draw, bbox, rng, compact)
        return

    if node_type == "MetricCard":
        draw_metric_hint(draw, bbox, asset, rng, compact)
        return

    if node_type == "Map":
        draw_map_hint(draw, bbox, rng, compact)
        return

    if node_type == "Image":
        draw_image_hint(draw, bbox, asset, rng, compact)
        return

    if node_type == "Filter":
        draw_filter_hint(draw, bbox, asset, rng)
        return

    if node_type == "Chart":
        if category == "Pies" or "pie" in key:
            draw_pie_hint(draw, bbox, rng)
        elif category == "Lines" or "line" in key:
            draw_line_hint(draw, bbox, rng, fill_area=False)
        elif category == "Areas" or "area" in key:
            draw_line_hint(draw, bbox, rng, fill_area=True)
        elif category == "Funnels" or "funnel" in key:
            draw_funnel_hint(draw, bbox, rng)
        elif category == "Scatters" or "scatter" in key:
            draw_scatter_hint(draw, bbox, rng)
        elif category == "WordClouds" or "wordcloud" in key:
            draw_wordcloud_hint(draw, bbox, rng)
        else:
            draw_bar_hint(draw, bbox, rng)
        draw_legend_hint(draw, bbox, rng, [cyan, green, yellow] if not compact else [cyan, green])
        return

    if node_type == "Decorate":
        for _ in range(3 if compact else 5):
            y = rng.uniform(bbox.y + 6, bbox.bottom - 6)
            draw.line([bbox.x + 8, y, bbox.right - 8, y + rng.uniform(-6, 6)], fill=muted, width=2)


def draw_table_hint(draw: ImageDraw.ImageDraw, bbox: BBox, rng: random.Random, compact: bool) -> None:
    rows = 4 if compact else rng.randint(5, 7)
    cols = 3 if compact else rng.randint(4, 5)
    header_h = min(28.0, max(16.0, bbox.h / (rows + 1)))
    draw.rectangle([bbox.x + 6, bbox.y + 6, bbox.right - 6, bbox.y + 6 + header_h], fill=(26, 93, 145, 135))
    for row in range(rows + 1):
        y = bbox.y + 6 + row * (bbox.h - 12) / rows
        draw.line([bbox.x + 6, y, bbox.right - 6, y], fill=(118, 210, 255, 150), width=1)
    for col in range(cols + 1):
        x = bbox.x + 6 + col * (bbox.w - 12) / cols
        draw.line([x, bbox.y + 6, x, bbox.bottom - 6], fill=(118, 210, 255, 130), width=1)
    font = load_font(10 if compact else 12)
    for row in range(1, rows):
        draw.text((bbox.x + 12, bbox.y + 8 + row * (bbox.h - 12) / rows), rng.choice(["在线", "告警", "正常", "128"]), font=font, fill=(226, 246, 255, 190))


def draw_metric_hint(draw: ImageDraw.ImageDraw, bbox: BBox, asset: ComponentAsset, rng: random.Random, compact: bool) -> None:
    font_title = fit_font(asset.record.title[:8] or "指标", 14 if compact else 18, max(24, int(bbox.w - 16)))
    font_num = load_font(max(14, min(32, int(bbox.h * (0.34 if compact else 0.42)))))
    draw.text((bbox.x + 10, bbox.y + 8), asset.record.title[:8] or "指标", font=font_title, fill=(198, 231, 255, 220))
    value = rng.choice(["12,960", "98.7%", "3,248", "76.5"])
    draw.text((bbox.x + 10, bbox.y + bbox.h * 0.38), value, font=font_num, fill=(255, 220, 92, 235))
    draw.line([bbox.x + 10, bbox.bottom - 14, bbox.right - 10, bbox.bottom - 14], fill=(67, 229, 181, 210), width=3)


def draw_map_hint(draw: ImageDraw.ImageDraw, bbox: BBox, rng: random.Random, compact: bool) -> None:
    points = [(bbox.x + rng.uniform(0.15, 0.85) * bbox.w, bbox.y + rng.uniform(0.18, 0.82) * bbox.h) for _ in range(9 if not compact else 6)]
    draw.line(points + [points[0]], fill=(96, 225, 255, 170), width=2)
    for point in points[:5]:
        r = rng.uniform(3, 8)
        draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r], outline=(255, 218, 92, 210), width=2)
    font = load_font(10)
    for label in ["北京", "上海", "广州"][: 2 if compact else 3]:
        draw.text((bbox.x + rng.uniform(8, max(10, bbox.w - 42)), bbox.y + rng.uniform(8, max(10, bbox.h - 18))), label, font=font, fill=(225, 245, 255, 190))


def draw_filter_hint(draw: ImageDraw.ImageDraw, bbox: BBox, asset: ComponentAsset, rng: random.Random) -> None:
    draw.rounded_rectangle([bbox.x + 4, bbox.y + 4, bbox.right - 4, bbox.bottom - 4], radius=5, fill=(9, 34, 64, 190), outline=(108, 210, 255, 210), width=2)
    font = fit_font(asset.record.title[:8] or "筛选", 13, max(20, int(bbox.w - 26)))
    draw.text((bbox.x + 10, bbox.y + max(7, bbox.h * 0.32)), asset.record.title[:8] or "筛选", font=font, fill=(224, 244, 255, 220))
    draw.line([bbox.right - 22, bbox.y + bbox.h * 0.45, bbox.right - 14, bbox.y + bbox.h * 0.58], fill=(255, 218, 92, 220), width=2)
    draw.line([bbox.right - 14, bbox.y + bbox.h * 0.58, bbox.right - 6, bbox.y + bbox.h * 0.45], fill=(255, 218, 92, 220), width=2)


def draw_image_hint(draw: ImageDraw.ImageDraw, bbox: BBox, asset: ComponentAsset, rng: random.Random, compact: bool = False) -> None:
    if bbox.w < 36 or bbox.h < 36:
        return
    accent = rng.choice([(92, 225, 230, 75), (125, 211, 252, 70), (252, 211, 77, 58)])
    outline = (accent[0], accent[1], accent[2], min(165, accent[3] + 70))
    inset = max(3.0, min(bbox.w, bbox.h) * 0.025)
    draw.rounded_rectangle(
        [bbox.x + inset, bbox.y + inset, bbox.right - inset, bbox.bottom - inset],
        radius=max(3.0, min(16.0, min(bbox.w, bbox.h) * 0.045)),
        outline=outline,
        width=2,
    )
    if compact:
        return
    cx, cy = bbox.center
    radius = min(bbox.w, bbox.h) * rng.uniform(0.16, 0.28)
    draw.ellipse([cx - radius, cy - radius * 0.72, cx + radius, cy + radius * 0.72], outline=accent, width=2)
    key = asset.record.key.lower()
    if "shield" in key:
        points = [
            (cx, cy - radius * 0.95),
            (cx + radius * 0.58, cy - radius * 0.36),
            (cx + radius * 0.44, cy + radius * 0.60),
            (cx, cy + radius),
            (cx - radius * 0.44, cy + radius * 0.60),
            (cx - radius * 0.58, cy - radius * 0.36),
        ]
        draw.polygon(points, outline=outline)


def draw_bar_hint(draw: ImageDraw.ImageDraw, bbox: BBox, rng: random.Random) -> None:
    bars = rng.randint(5, 9)
    base_y = bbox.bottom - 16
    draw.line([bbox.x + 12, base_y, bbox.right - 10, base_y], fill=(116, 196, 255, 145), width=1)
    step = max(8.0, (bbox.w - 30) / bars)
    for index in range(bars):
        bar_h = rng.uniform(0.22, 0.82) * max(8.0, bbox.h - 34)
        x = bbox.x + 16 + index * step
        color = rng.choice([(96, 225, 255, 210), (85, 230, 160, 210), (255, 218, 92, 210)])
        draw.rectangle([x, base_y - bar_h, x + step * 0.48, base_y], fill=color)


def draw_luminous_bar_chart_hint(draw: ImageDraw.ImageDraw, bbox: BBox, rng: random.Random) -> None:
    if bbox.w < 80 or bbox.h < 70:
        draw_bar_hint(draw, bbox, rng)
        return

    plot = BBox(bbox.x + bbox.w * 0.08, bbox.y + bbox.h * 0.10, bbox.w * 0.86, bbox.h * 0.78)
    grid = rng.choice([(70, 143, 190, 74), (58, 119, 176, 66), (45, 98, 150, 62)])
    axis = rng.choice([(122, 203, 255, 125), (86, 180, 255, 118)])
    label_color = (205, 230, 245, 160)

    draw.rectangle([bbox.x + 2, bbox.y + 2, bbox.right - 2, bbox.bottom - 2], fill=(4, 18, 40, rng.randint(12, 34)))
    for row in range(5):
        y = plot.y + row * plot.h / 4.0
        draw.line([plot.x, y, plot.right, y], fill=grid, width=1)
    for col in range(6):
        x = plot.x + col * plot.w / 5.0
        draw.line([x, plot.y, x, plot.bottom], fill=(grid[0], grid[1], grid[2], max(26, grid[3] - 24)), width=1)

    draw.line([plot.x, plot.y, plot.x, plot.bottom], fill=axis, width=1)
    draw.line([plot.x, plot.bottom, plot.right, plot.bottom], fill=axis, width=1)

    font = load_font(9)
    for index, value in enumerate(["800", "600", "400", "200"]):
        y = plot.y + index * plot.h / 4.4
        draw.text((bbox.x + 6, y - 6), value, font=font, fill=label_color)

    bars = rng.randint(5, 8)
    step = plot.w / bars
    palette = rng.choice(
        [
            [(26, 215, 255, 235), (45, 234, 217, 230)],
            [(34, 211, 238, 235), (22, 163, 255, 230)],
            [(35, 236, 190, 235), (17, 198, 255, 225)],
        ]
    )
    for index in range(bars):
        bar_h = rng.uniform(0.34, 0.96) * plot.h
        bar_w = max(7.0, min(22.0, step * rng.uniform(0.30, 0.44)))
        x = plot.x + step * index + (step - bar_w) / 2.0
        y = plot.bottom - bar_h
        glow = rng.choice([(18, 214, 255, 58), (42, 235, 204, 54), (55, 190, 255, 50)])
        draw.rounded_rectangle([x - 4, y - 3, x + bar_w + 4, plot.bottom + 2], radius=4, fill=glow)
        draw.rectangle([x, y, x + bar_w, plot.bottom], fill=palette[index % len(palette)])
        draw.polygon(
            [
                (x, y),
                (x + bar_w * 0.5, y - min(7.0, bar_w * 0.45)),
                (x + bar_w, y),
                (x + bar_w * 0.5, y + min(7.0, bar_w * 0.45)),
            ],
            fill=(128, 245, 255, 230),
        )
        if bbox.w > 170:
            label = rng.choice(["应用1", "应用2", "告警", "服务", "节点"])
            draw.text((x - 4, plot.bottom + 3), label, font=font, fill=label_color)


def draw_line_hint(draw: ImageDraw.ImageDraw, bbox: BBox, rng: random.Random, fill_area: bool) -> None:
    points = []
    count = rng.randint(6, 9)
    for index in range(count):
        x = bbox.x + 14 + index * (bbox.w - 28) / max(1, count - 1)
        y = bbox.y + rng.uniform(0.22, 0.76) * bbox.h
        points.append((x, y))
    if fill_area and len(points) > 2:
        draw.polygon(points + [(bbox.right - 14, bbox.bottom - 14), (bbox.x + 14, bbox.bottom - 14)], fill=(69, 170, 220, 80))
    draw.line(points, fill=(96, 225, 255, 225), width=3)
    for point in points:
        draw.ellipse([point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3], fill=(255, 218, 92, 230))


def draw_pie_hint(draw: ImageDraw.ImageDraw, bbox: BBox, rng: random.Random) -> None:
    side = min(bbox.w, bbox.h) * rng.uniform(0.46, 0.62)
    x1 = bbox.x + rng.uniform(0.18, 0.34) * bbox.w
    y1 = bbox.y + (bbox.h - side) / 2.0
    box = [x1, y1, x1 + side, y1 + side]
    start = rng.randint(0, 60)
    colors = [(96, 225, 255, 220), (85, 230, 160, 220), (255, 218, 92, 220), (167, 139, 250, 210)]
    for color in colors:
        end = start + rng.randint(55, 125)
        draw.pieslice(box, start, end, fill=color)
        start = end
    draw.ellipse([x1 + side * 0.32, y1 + side * 0.32, x1 + side * 0.68, y1 + side * 0.68], fill=(8, 26, 50, 210))


def draw_funnel_hint(draw: ImageDraw.ImageDraw, bbox: BBox, rng: random.Random) -> None:
    levels = 4
    for index in range(levels):
        top = bbox.y + 14 + index * (bbox.h - 28) / levels
        bottom = bbox.y + 14 + (index + 0.78) * (bbox.h - 28) / levels
        ratio_top = 0.9 - index * 0.13
        ratio_bottom = 0.9 - (index + 1) * 0.13
        x_top = bbox.x + bbox.w * (1 - ratio_top) / 2
        x_bottom = bbox.x + bbox.w * (1 - ratio_bottom) / 2
        draw.polygon([(x_top, top), (bbox.right - (x_top - bbox.x), top), (bbox.right - (x_bottom - bbox.x), bottom), (x_bottom, bottom)], fill=rng.choice([(96, 225, 255, 210), (85, 230, 160, 210), (255, 218, 92, 210)]))


def draw_scatter_hint(draw: ImageDraw.ImageDraw, bbox: BBox, rng: random.Random) -> None:
    for _ in range(24):
        x = bbox.x + rng.uniform(12, max(13, bbox.w - 12))
        y = bbox.y + rng.uniform(12, max(13, bbox.h - 12))
        r = rng.uniform(2, 4)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=rng.choice([(96, 225, 255, 210), (85, 230, 160, 210), (255, 218, 92, 210)]))


def draw_wordcloud_hint(draw: ImageDraw.ImageDraw, bbox: BBox, rng: random.Random) -> None:
    words = ["服务", "告警", "AI", "流量", "安全", "DATA", "趋势", "监控"]
    for word in words:
        font = load_font(rng.randint(10, 18))
        draw.text((bbox.x + rng.uniform(8, max(9, bbox.w - 54)), bbox.y + rng.uniform(8, max(9, bbox.h - 24))), word, font=font, fill=rng.choice([(96, 225, 255, 210), (85, 230, 160, 210), (255, 218, 92, 210)]))


def draw_legend_hint(draw: ImageDraw.ImageDraw, bbox: BBox, rng: random.Random, colors: List[Tuple[int, int, int, int]]) -> None:
    font = load_font(10)
    labels = ["同比", "环比", "当前"]
    x = bbox.right - min(92, bbox.w * 0.42)
    y = bbox.y + 8
    for index, color in enumerate(colors):
        yy = y + index * 15
        draw.rectangle([x, yy + 3, x + 10, yy + 9], fill=color)
        draw.text((x + 14, yy), labels[index], font=font, fill=(216, 238, 255, 185))


def fit_font(text: str, max_size: int, max_width: int) -> ImageFont.ImageFont:
    size = max_size
    while size >= 10:
        font = load_font(size)
        probe = Image.new("RGB", (8, 8))
        probe_draw = ImageDraw.Draw(probe)
        bbox = probe_draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
        size -= 1
    return load_font(10)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        try:
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_decorate(draw: ImageDraw.ImageDraw, bbox: BBox, rng: random.Random) -> None:
    color = rng.choice([(56, 189, 248, 210), (34, 211, 238, 210), (250, 204, 21, 210)])
    draw.rounded_rectangle([bbox.x, bbox.y, bbox.right, bbox.bottom], radius=2, outline=color, width=2)
    draw.line([bbox.x + 4, bbox.y + bbox.h / 2, bbox.right - 4, bbox.y + bbox.h / 2], fill=color, width=2)


def small_decorate_box(slot: BBox, rng: random.Random) -> BBox:
    w = rng.uniform(max(28.0, slot.w * 0.16), max(32.0, slot.w * 0.32))
    h = rng.uniform(12.0, 22.0)
    return BBox(slot.right - w - 12, slot.y + 10, w, h)


def to_sketch(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.invert(edges)
    edges = ImageEnhance.Contrast(edges).enhance(1.9)
    return ImageOps.colorize(edges, black=(30, 41, 59), white=(245, 247, 250)).convert("RGB")


def save_sample(out: Path, split: str, stem: str, image: Image.Image, labels: List[str], meta: Dict[str, object]) -> None:
    image.save(out / "images" / split / f"{stem}.png")
    (out / "labels" / split / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    (out / "meta" / split / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def draw_preview(image: Image.Image, meta: Dict[str, object], output_path: Path) -> None:
    preview = image.copy().convert("RGB")
    draw = ImageDraw.Draw(preview, "RGBA")
    colors = {
        "Panel": (56, 189, 248, 230),
        "Region": (125, 211, 252, 170),
        "Content": (148, 163, 184, 210),
        "Title": (250, 204, 21, 230),
        "Chart": (74, 222, 128, 230),
        "Table": (251, 146, 60, 230),
        "Map": (168, 85, 247, 230),
        "MetricCard": (244, 114, 182, 230),
        "Border": (96, 165, 250, 230),
        "Decorate": (203, 213, 225, 230),
        "Filter": (45, 212, 191, 230),
        "Image": (14, 165, 233, 230),
    }
    for node in meta.get("nodes", []):
        bbox = node["bbox"]
        color = colors.get(node["type"], (255, 255, 255, 230))
        x1, y1 = float(bbox["x"]), float(bbox["y"])
        x2, y2 = x1 + float(bbox["w"]), y1 + float(bbox["h"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{node['type']} {node.get('text') or node.get('componentId') or ''}".strip()
        draw.rectangle([x1, max(0, y1 - 14), x1 + min(180, len(label) * 7), y1], fill=(0, 0, 0, 130))
        draw.text((x1 + 2, max(0, y1 - 13)), label, fill=(255, 255, 255, 255))
    preview.save(output_path)


def update_counts(
    meta: Dict[str, object],
    class_counts: Dict[str, int],
    component_counts: Dict[str, int],
    node_type_counts: Dict[str, int],
) -> None:
    for node in meta.get("nodes", []):
        node_type = str(node.get("type", ""))
        if node_type in node_type_counts:
            node_type_counts[node_type] += 1
        if node_type in class_counts:
            class_counts[node_type] += 1
        component_id = node.get("componentId")
        if component_id in component_counts:
            component_counts[str(component_id)] += 1


def add_node(
    nodes: List[Dict[str, object]],
    labels: List[str],
    node_id: str,
    parent_id: str,
    asset: ComponentAsset,
    bbox: BBox,
    width: int,
    height: int,
    label_mode: str,
    class_to_id: Dict[str, int],
) -> None:
    add_synthetic_node(nodes, labels, node_id, parent_id, asset.coarse_type, bbox, width, height, asset.record.key, label_mode, class_to_id)


def add_synthetic_node(
    nodes: List[Dict[str, object]],
    labels: List[str],
    node_id: str,
    parent_id: str,
    node_type: str,
    bbox: BBox,
    width: int,
    height: int,
    component_id: Optional[str],
    label_mode: str,
    class_to_id: Dict[str, int],
    text: Optional[str] = None,
    role: Optional[str] = None,
    emit_label: bool = True,
) -> None:
    clipped = clip_bbox(bbox, width, height)
    if clipped.w < 4 or clipped.h < 4:
        return
    label = yolo_label(clipped, node_type, component_id, width, height, label_mode, class_to_id) if emit_label else None
    if label:
        labels.append(label)
    node = {
        "nodeId": node_id,
        "parentId": parent_id,
        "type": node_type,
        "componentId": component_id,
        "level": level_for_type(node_type),
        "bbox": clipped.to_dict(),
    }
    if text:
        node["text"] = text
    if role:
        node["role"] = role
    nodes.append(node)


def clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1 = max(0.0, min(float(width - 1), bbox.x))
    y1 = max(0.0, min(float(height - 1), bbox.y))
    x2 = max(x1 + 1.0, min(float(width), bbox.right))
    y2 = max(y1 + 1.0, min(float(height), bbox.bottom))
    return BBox(x1, y1, x2 - x1, y2 - y1)


def yolo_label(
    bbox: BBox,
    node_type: str,
    component_id: Optional[str],
    width: int,
    height: int,
    label_mode: str,
    class_to_id: Dict[str, int],
) -> Optional[str]:
    class_name = node_type if label_mode == "coarse" else component_id
    if not class_name or class_name not in class_to_id:
        return None
    class_id = class_to_id[class_name]
    cx = (bbox.x + bbox.w / 2.0) / width
    cy = (bbox.y + bbox.h / 2.0) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bbox.w / width:.6f} {bbox.h / height:.6f}"


def write_yolo_config(out: Path, classes: List[str]) -> None:
    (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "\n".join([f"  {index}: {name}" for index, name in enumerate(classes)])
        + "\n",
        encoding="utf-8",
    )


def coarse_type(record: ComponentRecord) -> str:
    key = record.key.lower()
    category = record.category
    if record.key in MAP_COMPONENT_IDS:
        return "Map"
    if record.key in IMAGE_COMPONENT_IDS or category in IMAGE_CATEGORIES:
        return "Image"
    if category == "Borders":
        return "Border"
    if category in TITLE_CATEGORIES:
        return "Title"
    if category in TABLE_CATEGORIES:
        return "Table"
    if category in MAP_CATEGORIES:
        return "Map"
    if category == "Biz":
        return "Image"
    if category in FILTER_CATEGORIES:
        return "Filter"
    if category in DECORATE_CATEGORIES or "pipeline" in key or "decorate" in key or "fullscreen" in key:
        return "Decorate"
    if category in CHART_CATEGORIES:
        return "Chart"
    if any(token in key for token in ["pie", "bar", "line", "funnel", "radar", "sankey", "graph", "heatmap", "scatter", "wordcloud", "tree"]):
        return "Chart"
    if any(token in key for token in ["number", "energy", "status", "count", "clock", "dial", "water", "process", "flipper"]):
        return "MetricCard"
    return "Decorate"


def level_for_type(node_type: str) -> int:
    return {
        "Region": 1,
        "Panel": 2,
        "Border": 2,
        "Content": 3,
        "Title": 3,
        "Decorate": 3,
        "Filter": 4,
        "Chart": 4,
        "Table": 4,
        "Map": 4,
        "MetricCard": 4,
        "Image": 4,
    }.get(node_type, 4)


if __name__ == "__main__":
    main()
