#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.finetune_data import export_qwen_vl_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Qwen-VL JSONL dataset for 95-class component recognition.")
    parser.add_argument("--output", default=str(ROOT / "data" / "finetune" / "qwen_vl_component_recognition.jsonl"))
    parser.add_argument("--reference-variants", type=int, default=20)
    parser.add_argument("--no-corrections", action="store_true")
    parser.add_argument("--limit-components", type=int, default=None)
    args = parser.parse_args()

    manifest = export_qwen_vl_dataset(
        Path(args.output),
        reference_variants=args.reference_variants,
        include_corrections=not args.no_corrections,
        limit_components=args.limit_components,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
