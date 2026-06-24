#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pipeline import ScreenParser


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a large-screen design image.")
    parser.add_argument("image", help="Path to design or sketch image.")
    parser.add_argument("--input-type", default="design", choices=["design", "sketch"])
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--catalog", default=str(ROOT.parent / "ai-schema-view" / "schema-md" / "catalog.md"))
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    parser.add_argument("--yolo-model", default=None)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--graph-model", default=None)
    parser.add_argument("--reference-library", default=None)
    parser.add_argument("--multimodal-classifier", action="store_true")
    parser.add_argument("--multimodal-model", default=None)
    parser.add_argument("--multimodal-base-url", default=None)
    parser.add_argument("--multimodal-api-key", default=None)
    parser.add_argument("--force-llm", action="store_true")
    args = parser.parse_args()

    screen_parser = ScreenParser(
        args.catalog,
        artifact_root=args.artifacts,
        yolo_model=args.yolo_model,
        yolo_conf=args.yolo_conf,
        graph_model=args.graph_model,
        reference_library=args.reference_library,
        multimodal_classifier=args.multimodal_classifier,
        multimodal_model=args.multimodal_model,
        multimodal_base_url=args.multimodal_base_url,
        multimodal_api_key=args.multimodal_api_key,
    )
    artifacts = screen_parser.parse(args.image, input_type=args.input_type, top_k=args.top_k, force_llm=args.force_llm)
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
