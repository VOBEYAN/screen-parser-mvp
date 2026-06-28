#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO coarse structure detector on composited screen data.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml.")
    parser.add_argument("--base-model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="yolo_screen_structure_v1")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--close-mosaic", type=int, default=None)
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.base_model)
    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "project": args.project,
        "name": args.name,
        "batch": args.batch,
        "amp": args.amp,
        "cache": args.cache,
        "exist_ok": args.exist_ok,
    }
    if args.device:
        train_args["device"] = args.device
    if args.workers is not None:
        train_args["workers"] = args.workers
    if args.patience is not None:
        train_args["patience"] = args.patience
    if args.fraction is not None:
        train_args["fraction"] = args.fraction
    if args.close_mosaic is not None:
        train_args["close_mosaic"] = args.close_mosaic
    model.train(**train_args)


if __name__ == "__main__":
    main()
