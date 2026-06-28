#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Panel", "Title", "Chart", "Table", "Map", "MetricCard", "Border", "Decorate", "Filter", "Image"]
DEFAULT_EXISTING_DATASETS = [
    ROOT / "data" / "screen-structure-local-v1",
    ROOT / "data" / "screen-structure-title-hard-v2",
    ROOT / "data" / "screen-structure-chart-hard-v3",
    ROOT / "data" / "local-run-structure",
]


@dataclass(frozen=True)
class Recipe:
    name: str
    width: int
    height: int
    train_count: int
    val_count: int
    components_per_screen: int
    layout_mode: str
    overlay_rate: float
    hard_chart_rate: float
    include_sketch: bool
    seed: int


BALANCED_RECIPES = [
    Recipe("wide_mixed_1080", 1920, 1080, 280, 60, 10, "mixed", 0.48, 0.55, True, 2026062701),
    Recipe("dense_side_768", 1365, 768, 220, 50, 12, "dense", 0.55, 0.68, True, 2026062702),
    Recipe("ultra_wide_1440", 2560, 1440, 160, 40, 10, "mixed", 0.42, 0.5, True, 2026062703),
]

SMOKE_RECIPES = [
    Recipe("smoke_1080", 1920, 1080, 8, 2, 8, "mixed", 0.45, 0.5, True, 2026062791),
]

FULL_RECIPES = [
    Recipe("wide_mixed_1080", 1920, 1080, 560, 120, 10, "mixed", 0.50, 0.60, True, 2026062701),
    Recipe("dense_side_768", 1365, 768, 440, 100, 12, "dense", 0.58, 0.72, True, 2026062702),
    Recipe("ultra_wide_1440", 2560, 1440, 320, 80, 10, "mixed", 0.45, 0.58, True, 2026062703),
    Recipe("grid_720", 1280, 720, 260, 60, 9, "grid", 0.38, 0.48, True, 2026062704),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a richer YOLO coarse-structure dataset by mixing old and newly generated dashboard data.")
    parser.add_argument("--out", default=str(ROOT / "data" / "screen-structure-rich-v5"))
    parser.add_argument("--preset", choices=["smoke", "balanced", "full"], default="balanced")
    parser.add_argument("--existing", nargs="*", default=[str(path) for path in DEFAULT_EXISTING_DATASETS])
    parser.add_argument("--reviewed-hard", nargs="*", default=[], help="Optional human-reviewed hard-case YOLO datasets to include.")
    parser.add_argument("--tmp", default=str(ROOT / "data" / "_tmp_screen_structure_rich_v5"))
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    tmp = Path(args.tmp)
    if args.clean and out.exists():
        shutil.rmtree(out)
    prepare_dataset_dirs(out)
    if not args.skip_generate:
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)

    sources = [Path(path) for path in args.existing]
    generated_sources: list[Path] = []
    if not args.skip_generate:
        for recipe in recipes_for_preset(args.preset):
            generated = generate_recipe(recipe, tmp)
            generated_sources.append(generated)
    sources.extend(generated_sources)
    sources.extend(Path(path) for path in args.reviewed_hard)

    class_counts = {name: 0 for name in CLASSES}
    source_summaries = []
    for source in sources:
        if not source.exists():
            print(f"skip missing dataset: {source}", file=sys.stderr)
            continue
        validate_classes(source)
        summary = copy_dataset(source, out, class_counts)
        source_summaries.append(summary)

    write_yolo_config(out)
    summary = {
        "out": str(out.resolve()),
        "preset": args.preset,
        "classes": CLASSES,
        "sourceCount": len(source_summaries),
        "sources": source_summaries,
        "classCounts": class_counts,
        "train": count_split(out, "train"),
        "val": count_split(out, "val"),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def recipes_for_preset(preset: str) -> list[Recipe]:
    if preset == "smoke":
        return SMOKE_RECIPES
    if preset == "full":
        return FULL_RECIPES
    return BALANCED_RECIPES


def generate_recipe(recipe: Recipe, tmp: Path) -> Path:
    out = tmp / recipe.name
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "generate_composited_training_data.py"),
        "--out",
        str(out),
        "--width",
        str(recipe.width),
        "--height",
        str(recipe.height),
        "--train-count",
        str(recipe.train_count),
        "--val-count",
        str(recipe.val_count),
        "--components-per-screen",
        str(recipe.components_per_screen),
        "--label-mode",
        "coarse",
        "--layout-mode",
        recipe.layout_mode,
        "--title-placement-mode",
        "diverse",
        "--overlay-rate",
        str(recipe.overlay_rate),
        "--hard-chart-rate",
        str(recipe.hard_chart_rate),
        "--seed",
        str(recipe.seed),
        "--clean",
    ]
    if recipe.include_sketch:
        cmd.append("--include-sketch")
    print("generate", recipe.name, " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return out


def prepare_dataset_dirs(out: Path) -> None:
    for split in ["train", "val"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out / "meta" / split).mkdir(parents=True, exist_ok=True)


def validate_classes(dataset: Path) -> None:
    classes_path = dataset / "classes.txt"
    if not classes_path.exists():
        raise SystemExit(f"Missing classes.txt in {dataset}")
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if classes == CLASSES:
        return
    if classes == CLASSES[: len(classes)]:
        print(f"allow prefix class list in {dataset}: {classes} -> {CLASSES}", file=sys.stderr)
        return
    if classes != CLASSES:
        raise SystemExit(f"Class mismatch in {dataset}: {classes} != {CLASSES}")


def copy_dataset(source: Path, out: Path, class_counts: dict[str, int]) -> dict[str, object]:
    summary = {"source": str(source), "train": {"images": 0, "labels": 0}, "val": {"images": 0, "labels": 0}}
    prefix = sanitize(source.name)
    for split in ["train", "val"]:
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        meta_dir = source / "meta" / split
        if not image_dir.exists() or not label_dir.exists():
            continue
        for image_path in sorted(iter_images(image_dir)):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            stem = f"{prefix}_{image_path.stem}"
            shutil.copy2(image_path, out / "images" / split / f"{stem}{image_path.suffix.lower()}")
            text = label_path.read_text(encoding="utf-8")
            (out / "labels" / split / f"{stem}.txt").write_text(text, encoding="utf-8")
            count_labels(text, class_counts)
            summary[split]["images"] += 1
            summary[split]["labels"] += 1
            meta_path = meta_dir / f"{image_path.stem}.json"
            if meta_path.exists():
                shutil.copy2(meta_path, out / "meta" / split / f"{stem}.json")
    return summary


def iter_images(path: Path) -> Iterable[Path]:
    for suffix in ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]:
        yield from path.glob(suffix)


def count_labels(text: str, class_counts: dict[str, int]) -> None:
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            continue
        if 0 <= class_id < len(CLASSES):
            class_counts[CLASSES[class_id]] += 1


def count_split(out: Path, split: str) -> dict[str, int]:
    images = sum(1 for _ in iter_images(out / "images" / split))
    labels = sum(1 for _ in (out / "labels" / split).glob("*.txt"))
    return {"images": images, "labels": labels}


def write_yolo_config(out: Path) -> None:
    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASSES))
        + "\n",
        encoding="utf-8",
    )


def sanitize(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_") or "dataset"


if __name__ == "__main__":
    main()
