#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from shutil import copy2

from safetensors import safe_open
from safetensors.torch import save_file


SKIP_FILES = {"model.safetensors", "model.safetensors.index.json"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert mlx-community Qwen3-VL bf16 keys for Transformers.")
    parser.add_argument("--source", required=True, help="Path to the mlx-community Qwen3-VL snapshot.")
    parser.add_argument("--output", required=True, help="Output directory for the HF-keyed model.")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    for path in source.iterdir():
        if path.name in SKIP_FILES:
            continue
        if path.is_file() or path.is_symlink():
            copy2(path, output / path.name, follow_symlinks=True)

    in_file = source / "model.safetensors"
    out_file = output / "model.safetensors"
    converted = {}
    with safe_open(in_file, framework="pt", device="cpu") as tensors:
        keys = list(tensors.keys())
        for index, key in enumerate(keys, start=1):
            new_key = convert_key(key)
            tensor = tensors.get_tensor(key)
            if new_key == "model.visual.patch_embed.proj.weight" and tuple(tensor.shape) == (1024, 2, 16, 16, 3):
                tensor = tensor.permute(0, 4, 1, 2, 3).contiguous()
            converted[new_key] = tensor
            if index % 100 == 0:
                print(f"converted {index}/{len(keys)}")

    save_file(converted, out_file, metadata={"format": "pt"})
    print(f"saved {out_file}")
    print(f"tensors {len(converted)}")


def convert_key(key: str) -> str:
    if key.startswith("language_model.model."):
        return "model.language_model." + key[len("language_model.model.") :]
    if key.startswith("vision_tower."):
        return "model.visual." + key[len("vision_tower.") :]
    return key


if __name__ == "__main__":
    main()
