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
class HNode:
    node_id: str
    bbox: BBox
    type: str
    parent_id: Optional[str]
    matched_id: Optional[str] = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GT hierarchy with predicted hierarchy.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--gt-meta", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--out-dir", default=str(ROOT / "artifacts" / "comparisons" / "hierarchy"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    gt_nodes = load_gt(args.gt_meta)
    pred_nodes = load_prediction(args.prediction)
    metrics = match_and_score(gt_nodes, pred_nodes, args.iou_threshold)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    out_image = out_dir / f"{stem}_hierarchy.png"
    draw_hierarchy_comparison(image, gt_nodes, pred_nodes, out_image, metrics)

    out_json = out_dir / f"{stem}_hierarchy_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"hierarchyImage": str(out_image), "metricsJson": str(out_json), "metrics": metrics}, ensure_ascii=False, indent=2))


def load_gt(path: str) -> List[HNode]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = []
    for item in data["nodes"]:
        bbox = item["bbox"]
        parent_id = item.get("parentId") or "screen_0000"
        nodes.append(HNode(item["nodeId"], BBox(bbox["x"], bbox["y"], bbox["w"], bbox["h"]), item["type"], parent_id))
    return nodes


def load_prediction(path: str) -> List[HNode]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = []
    for item in data["nodes"]:
        if item["type"] == "Screen":
            continue
        bbox = item["bbox"]
        nodes.append(HNode(item["node_id"], BBox(bbox["x"], bbox["y"], bbox["w"], bbox["h"]), item["type"], item.get("parent_id")))
    return nodes


def match_and_score(gt_nodes: List[HNode], pred_nodes: List[HNode], threshold: float) -> Dict[str, object]:
    candidates: List[Tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_nodes):
        for pi, pred in enumerate(pred_nodes):
            if gt.type != pred.type:
                continue
            score = iou(gt.bbox, pred.bbox)
            if score >= threshold:
                candidates.append((score, gi, pi))

    candidates.sort(reverse=True)
    used_gt = set()
    used_pred = set()
    matches = []
    for score, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        gt_nodes[gi].matched_id = pred_nodes[pi].node_id
        pred_nodes[pi].matched_id = gt_nodes[gi].node_id
        matches.append((gt_nodes[gi].node_id, pred_nodes[pi].node_id, score))

    pred_by_id = {node.node_id: node for node in pred_nodes}
    gt_by_id = {node.node_id: node for node in gt_nodes}
    gt_to_pred = {gt_id: pred_id for gt_id, pred_id, _ in matches}

    parent_total = 0
    parent_correct = 0
    parent_rows = []
    for gt in gt_nodes:
        parent_total += 1
        pred_id = gt_to_pred.get(gt.node_id)
        if not pred_id:
            parent_rows.append({"gt": gt.node_id, "status": "child_unmatched"})
            continue
        pred = pred_by_id[pred_id]
        if gt.parent_id == "screen_0000":
            expected_parent = "screen_0000"
        else:
            expected_parent = gt_to_pred.get(gt.parent_id)
        ok = expected_parent is not None and pred.parent_id == expected_parent
        if ok:
            parent_correct += 1
        parent_rows.append(
            {
                "gt": gt.node_id,
                "pred": pred_id,
                "gtParent": gt.parent_id,
                "expectedPredParent": expected_parent,
                "predParent": pred.parent_id,
                "correct": ok,
            }
        )

    matched = len(matches)
    return {
        "gtCount": len(gt_nodes),
        "predCount": len(pred_nodes),
        "matchedCount": matched,
        "parentTotal": parent_total,
        "parentCorrect": parent_correct,
        "parentAccuracy": round(parent_correct / parent_total, 4) if parent_total else 0.0,
        "matchRecall": round(matched / len(gt_nodes), 4) if gt_nodes else 0.0,
        "parentRows": parent_rows,
    }


def draw_hierarchy_comparison(image: Image.Image, gt_nodes: List[HNode], pred_nodes: List[HNode], out_path: Path, metrics: Dict[str, object]) -> None:
    width, height = image.size
    canvas = Image.new("RGB", (width * 2, height + 80), (17, 24, 39))
    canvas.paste(image, (0, 80))
    canvas.paste(image, (width, 80))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((24, 24), "GT hierarchy", fill=(220, 252, 231, 255))
    draw.text((width + 24, 24), "Graph Transformer prediction", fill=(224, 242, 254, 255))
    draw.text((width // 2 - 190, 24), f"parentAccuracy={metrics['parentAccuracy']} matchRecall={metrics['matchRecall']}", fill=(255, 255, 255, 255))

    draw_nodes(draw, gt_nodes, 0, 80)
    draw_edges(draw, gt_nodes, 0, 80)
    draw_nodes(draw, pred_nodes, width, 80)
    draw_edges(draw, pred_nodes, width, 80)
    canvas.save(out_path)


def draw_nodes(draw: ImageDraw.ImageDraw, nodes: List[HNode], offset_x: int, offset_y: int) -> None:
    for node in nodes:
        color = (34, 197, 94, 230) if node.matched_id else (250, 204, 21, 230)
        x1 = offset_x + node.bbox.x
        y1 = offset_y + node.bbox.y
        x2 = offset_x + node.bbox.right
        y2 = offset_y + node.bbox.bottom
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, max(offset_y, y1 - 14)), f"{node.node_id} {node.type}", fill=(255, 255, 255, 255))


def draw_edges(draw: ImageDraw.ImageDraw, nodes: List[HNode], offset_x: int, offset_y: int) -> None:
    by_id = {node.node_id: node for node in nodes}
    for node in nodes:
        if not node.parent_id or node.parent_id == "screen_0000" or node.parent_id not in by_id:
            continue
        parent = by_id[node.parent_id]
        px, py = parent.bbox.center
        cx, cy = node.bbox.center
        draw.line([offset_x + px, offset_y + py, offset_x + cx, offset_y + cy], fill=(59, 130, 246, 160), width=2)


if __name__ == "__main__":
    main()

