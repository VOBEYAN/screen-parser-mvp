#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.geometry import iou
from app.schemas import BBox


@dataclass
class BoxItem:
    item_id: str
    bbox: BBox
    type: str
    matched: bool = False
    matched_id: Optional[str] = None
    match_iou: float = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize prediction against synthetic ground truth.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--gt-meta", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--out-dir", default=str(ROOT / "artifacts" / "comparisons"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(args.image)
    stem = image_path.stem

    gt = load_gt(args.gt_meta)
    pred = load_prediction(args.prediction)
    metrics = match_boxes(gt, pred, args.iou_threshold)

    image = Image.open(args.image).convert("RGB")
    comparison_path = out_dir / f"{stem}_comparison.png"
    draw_comparison(image, gt, pred, comparison_path, metrics)

    metrics_path = out_dir / f"{stem}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = out_dir / f"{stem}_comparison.md"
    report_path.write_text(markdown_report(stem, comparison_path, metrics), encoding="utf-8")

    print(json.dumps({"comparisonImage": str(comparison_path), "metricsJson": str(metrics_path), "reportMd": str(report_path), "metrics": metrics}, ensure_ascii=False, indent=2))


def load_gt(path: str) -> List[BoxItem]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = []
    for node in data["nodes"]:
        bbox = node["bbox"]
        items.append(
            BoxItem(
                item_id=node["nodeId"],
                bbox=BBox(float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"])),
                type=node["type"],
            )
        )
    return items


def load_prediction(path: str) -> List[BoxItem]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = []
    for node in data["nodes"]:
        if node["type"] == "Screen":
            continue
        bbox = node["bbox"]
        items.append(
            BoxItem(
                item_id=node["node_id"],
                bbox=BBox(float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"])),
                type=node["type"],
            )
        )
    return items


def match_boxes(gt: List[BoxItem], pred: List[BoxItem], threshold: float) -> Dict[str, object]:
    candidates: List[Tuple[float, int, int]] = []
    for gi, gt_item in enumerate(gt):
        for pi, pred_item in enumerate(pred):
            if normalize_type(gt_item.type) != normalize_type(pred_item.type):
                continue
            score = iou(gt_item.bbox, pred_item.bbox)
            if score >= threshold:
                candidates.append((score, gi, pi))

    candidates.sort(reverse=True)
    matched_gt = set()
    matched_pred = set()
    matches = []
    for score, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        gt[gi].matched = True
        pred[pi].matched = True
        gt[gi].matched_id = pred[pi].item_id
        pred[pi].matched_id = gt[gi].item_id
        gt[gi].match_iou = score
        pred[pi].match_iou = score
        matches.append({"gt": gt[gi].item_id, "pred": pred[pi].item_id, "type": gt[gi].type, "iou": round(score, 4)})

    tp = len(matches)
    fp = len(pred) - tp
    fn = len(gt) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "iouThreshold": threshold,
        "gtCount": len(gt),
        "predCount": len(pred),
        "truePositive": tp,
        "falsePositive": fp,
        "falseNegative": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "matches": matches,
    }


def normalize_type(value: str) -> str:
    if value in {"Panel", "Title", "Chart", "Table", "Map", "MetricCard"}:
        return value
    if value == "Border":
        return "Panel"
    return value


def draw_comparison(image: Image.Image, gt: List[BoxItem], pred: List[BoxItem], output_path: Path, metrics: Dict[str, object]) -> None:
    width, height = image.size
    canvas = Image.new("RGB", (width * 2, height + 70), (18, 24, 38))
    canvas.paste(image, (0, 70))
    canvas.paste(image, (width, 70))
    draw = ImageDraw.Draw(canvas, "RGBA")

    draw.text((24, 20), "Ground Truth", fill=(220, 252, 231, 255))
    draw.text((width + 24, 20), "Prediction", fill=(224, 242, 254, 255))
    summary = f"P={metrics['precision']} R={metrics['recall']} TP={metrics['truePositive']} FP={metrics['falsePositive']} FN={metrics['falseNegative']}"
    draw.text((width // 2 - 160, 20), summary, fill=(255, 255, 255, 255))

    for item in gt:
        color = (34, 197, 94, 230) if item.matched else (250, 204, 21, 230)
        draw_box(draw, item, 0, 70, color)

    for item in pred:
        color = (34, 197, 94, 230) if item.matched else (239, 68, 68, 230)
        draw_box(draw, item, width, 70, color)

    canvas.save(output_path)


def draw_box(draw: ImageDraw.ImageDraw, item: BoxItem, offset_x: int, offset_y: int, color) -> None:
    box = item.bbox
    x1 = offset_x + box.x
    y1 = offset_y + box.y
    x2 = offset_x + box.right
    y2 = offset_y + box.bottom
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    label = f"{item.item_id} {item.type}"
    if item.matched:
        label += f" IoU={item.match_iou:.2f}"
    draw.rectangle([x1, max(offset_y, y1 - 18), x1 + min(260, len(label) * 7), y1], fill=(0, 0, 0, 150))
    draw.text((x1 + 3, max(offset_y, y1 - 16)), label, fill=(255, 255, 255, 255))


def markdown_report(stem: str, comparison_path: Path, metrics: Dict[str, object]) -> str:
    lines = [
        f"# {stem} 推理结果对比",
        "",
        f"- GT: {metrics['gtCount']}",
        f"- Prediction: {metrics['predCount']}",
        f"- Precision: {metrics['precision']}",
        f"- Recall: {metrics['recall']}",
        f"- TP / FP / FN: {metrics['truePositive']} / {metrics['falsePositive']} / {metrics['falseNegative']}",
        f"- Image: `{comparison_path}`",
        "",
        "## 匹配明细",
        "",
        "| GT | Prediction | Type | IoU |",
        "|---|---|---|---|",
    ]
    for match in metrics["matches"]:
        lines.append(f"| {match['gt']} | {match['pred']} | {match['type']} | {match['iou']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

