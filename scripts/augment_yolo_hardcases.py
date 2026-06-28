#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create color/geometric variants of reviewed YOLO hard-case samples.")
    parser.add_argument("--src", required=True, help="Source YOLO dataset with images/<split> and labels/<split>.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--count-per-image", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2026062705)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    for kind in ["images", "labels"]:
        (out / kind / args.split).mkdir(parents=True, exist_ok=True)

    copy_config(src, out)
    rng = random.Random(args.seed)
    total = 0
    for image_path in iter_images(src / "images" / args.split):
        label_path = src / "labels" / args.split / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        labels = read_labels(label_path)
        image = Image.open(image_path).convert("RGB")
        for index in range(args.count_per_image):
            variant, variant_labels = augment_one(image, labels, rng)
            if not variant_labels:
                continue
            stem = f"{image_path.stem}_aug_{index:03d}"
            variant.save(out / "images" / args.split / f"{stem}.jpg", quality=rng.randint(78, 94), optimize=True)
            (out / "labels" / args.split / f"{stem}.txt").write_text(
                "\n".join(format_label(label) for label in variant_labels) + "\n",
                encoding="utf-8",
            )
            total += 1
    print({"out": str(out), "split": args.split, "images": total})


def copy_config(src: Path, out: Path) -> None:
    for name in ["classes.txt", "data.yaml"]:
        source = src / name
        if source.exists():
            if name == "data.yaml":
                classes = [line.strip() for line in (src / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
                (out / name).write_text(
                    f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
                    + "\n".join(f"  {idx}: {value}" for idx, value in enumerate(classes))
                    + "\n",
                    encoding="utf-8",
                )
            else:
                shutil.copy2(source, out / name)


def iter_images(path: Path) -> Iterable[Path]:
    for item in sorted(path.iterdir()) if path.exists() else []:
        if item.suffix.lower() in IMAGE_SUFFIXES:
            yield item


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    labels: list[tuple[int, float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            cx, cy, w, h = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        labels.append((class_id, cx, cy, w, h))
    return labels


def augment_one(
    image: Image.Image,
    labels: list[tuple[int, float, float, float, float]],
    rng: random.Random,
) -> tuple[Image.Image, list[tuple[int, float, float, float, float]]]:
    width, height = image.size
    scale = rng.uniform(0.94, 1.07)
    dx = rng.uniform(-0.035, 0.035) * width
    dy = rng.uniform(-0.03, 0.035) * height
    transformed = image.transform(
        (width, height),
        Image.Transform.AFFINE,
        (1.0 / scale, 0, -dx / scale, 0, 1.0 / scale, -dy / scale),
        resample=Image.Resampling.BICUBIC,
        fillcolor=rng.choice([(3, 10, 24), (5, 14, 32), (7, 18, 40)]),
    )
    transformed = jitter_color(transformed, rng)
    if rng.random() < 0.22:
        transformed = transformed.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 0.8)))

    adjusted = [clip_label(label, width, height, scale, dx, dy) for label in labels]
    return transformed, [label for label in adjusted if label is not None]


def jitter_color(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.78, 1.22))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.82, 1.25))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.72, 1.28))
    image = ImageEnhance.Sharpness(image).enhance(rng.uniform(0.85, 1.45))
    return image


def clip_label(
    label: tuple[int, float, float, float, float],
    width: int,
    height: int,
    scale: float,
    dx: float,
    dy: float,
) -> tuple[int, float, float, float, float] | None:
    class_id, cx, cy, box_w, box_h = label
    x1 = (cx - box_w / 2.0) * width * scale + dx
    y1 = (cy - box_h / 2.0) * height * scale + dy
    x2 = (cx + box_w / 2.0) * width * scale + dx
    y2 = (cy + box_h / 2.0) * height * scale + dy
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return (class_id, (x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height, (x2 - x1) / width, (y2 - y1) / height)


def format_label(label: tuple[int, float, float, float, float]) -> str:
    class_id, cx, cy, w, h = label
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


if __name__ == "__main__":
    main()
