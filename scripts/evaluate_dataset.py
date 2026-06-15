#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.pipeline import ScreenParser
from compare_prediction import load_gt, load_prediction, match_boxes


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate parser on a synthetic dataset split.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--catalog", default=str(ROOT.parent / "ai-schema-view" / "schema-md" / "catalog.md"))
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts" / "eval_runs"))
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "evaluation"))
    parser.add_argument("--yolo-model", default=str(ROOT / "models" / "yolo_screen_components_demo.pt"))
    parser.add_argument("--graph-model", default=str(ROOT / "models" / "graph_transformer_demo.pt"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    image_dir = dataset / "images" / args.split
    meta_dir = dataset / "meta" / args.split
    images = sorted(image_dir.glob("*.png"))
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found: {image_dir}")

    parser_engine = ScreenParser(
        catalog_path=args.catalog,
        artifact_root=args.artifacts,
        yolo_model=args.yolo_model,
        graph_model=args.graph_model,
    )

    records = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_pred = 0
    total_overlaps = 0

    for image_path in images:
        meta_path = meta_dir / f"{image_path.stem}.json"
        artifacts = parser_engine.parse(str(image_path), input_type="design", top_k=1)
        gt = load_gt(str(meta_path))
        pred = load_prediction(artifacts["resultJson"])
        metrics = match_boxes(gt, pred, args.iou_threshold)
        result = json.loads(Path(artifacts["resultJson"]).read_text(encoding="utf-8"))

        total_tp += int(metrics["truePositive"])
        total_fp += int(metrics["falsePositive"])
        total_fn += int(metrics["falseNegative"])
        total_gt += int(metrics["gtCount"])
        total_pred += int(metrics["predCount"])
        total_overlaps += len(result["overlaps"])
        records.append(
            {
                "image": str(image_path),
                "runId": artifacts["runId"],
                "reportHtml": artifacts["reportHtml"],
                "gtCount": metrics["gtCount"],
                "predCount": metrics["predCount"],
                "tp": metrics["truePositive"],
                "fp": metrics["falsePositive"],
                "fn": metrics["falseNegative"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "overlapCount": len(result["overlaps"]),
            }
        )

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    summary = {
        "dataset": str(dataset),
        "split": args.split,
        "imageCount": len(images),
        "iouThreshold": args.iou_threshold,
        "gtCount": total_gt,
        "predCount": total_pred,
        "truePositive": total_tp,
        "falsePositive": total_fp,
        "falseNegative": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "avgOverlapCount": round(total_overlaps / len(images), 4),
        "records": records,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{dataset.name}_{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = out_dir / f"{dataset.name}_{args.split}_summary.md"
    md_path.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"summaryJson": str(summary_path), "summaryMd": str(md_path), "summary": summary}, ensure_ascii=False, indent=2))


def markdown(summary: dict) -> str:
    lines = [
        "# 测试集评估结果",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Split: `{summary['split']}`",
        f"- Images: {summary['imageCount']}",
        f"- IoU threshold: {summary['iouThreshold']}",
        f"- GT / Pred: {summary['gtCount']} / {summary['predCount']}",
        f"- TP / FP / FN: {summary['truePositive']} / {summary['falsePositive']} / {summary['falseNegative']}",
        f"- Precision: {summary['precision']}",
        f"- Recall: {summary['recall']}",
        f"- Avg overlap count: {summary['avgOverlapCount']}",
        "",
        "| image | TP | FP | FN | Precision | Recall | Overlaps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in summary["records"]:
        lines.append(
            f"| {Path(record['image']).name} | {record['tp']} | {record['fp']} | {record['fn']} | "
            f"{record['precision']} | {record['recall']} | {record['overlapCount']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

