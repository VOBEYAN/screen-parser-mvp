#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO detector on generated synthetic data.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml.")
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--project", default="models/yolo")
    parser.add_argument("--name", default="screen_components")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
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
    }
    if args.device:
        train_args["device"] = args.device
    model.train(**train_args)


if __name__ == "__main__":
    main()
