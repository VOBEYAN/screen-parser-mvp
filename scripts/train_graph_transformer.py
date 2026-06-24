#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.graph_transformer import GraphTransformerHierarchyModel


TYPE_NAMES = ["Screen", "Region", "Panel", "Title", "Border", "Content", "Chart", "Table", "Map", "MetricCard", "Decorate", "Filter"]
TYPE_TO_ID = {name: idx for idx, name in enumerate(TYPE_NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Graph Transformer on synthetic hierarchy metadata.")
    parser.add_argument("--data", default=str(ROOT / "data" / "screen-structure-v1"), help="Composited structure dataset root.")
    parser.add_argument("--out", default=str(ROOT / "models" / "graph_transformer_structure_v1.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_items = load_graphs(Path(args.data) / "meta" / "train")
    val_items = load_graphs(Path(args.data) / "meta" / "val")
    if not train_items:
        raise SystemExit(f"No training metadata found under {Path(args.data) / 'meta' / 'train'}")

    input_dim = len(train_items[0][0][0])
    model = GraphTransformerHierarchyModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        num_types=len(TYPE_NAMES),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_items)
        model.train()
        train_loss = 0.0
        steps = 0
        for batch in batches(train_items, args.batch_size):
            features, type_targets, level_targets, parent_targets, mask = collate(batch)
            output = model(features, padding_mask=~mask)
            loss = compute_loss(output, type_targets, level_targets, parent_targets, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            steps += 1

        val_loss = evaluate(model, val_items, args.batch_size) if val_items else 0.0
        print(f"epoch={epoch} train_loss={train_loss / max(steps, 1):.4f} val_loss={val_loss:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": input_dim,
            "type_names": TYPE_NAMES,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "heads": args.heads,
        },
        out,
    )
    print(f"Graph Transformer checkpoint saved: {out}")


def load_graphs(meta_dir: Path) -> List[Tuple[List[List[float]], List[int], List[int], List[int]]]:
    items = []
    for path in sorted(meta_dir.glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        items.append(graph_from_meta(meta))
    return items


def graph_from_meta(meta: Dict[str, object]) -> Tuple[List[List[float]], List[int], List[int], List[int]]:
    width = float(meta["width"])
    height = float(meta["height"])
    raw_nodes = list(meta["nodes"])
    root = {
        "nodeId": "screen_0000",
        "parentId": None,
        "type": "Screen",
        "level": 0,
        "bbox": {"x": 0, "y": 0, "w": width, "h": height},
    }
    nodes = [root] + raw_nodes
    id_to_index = {node["nodeId"]: idx for idx, node in enumerate(nodes)}

    features: List[List[float]] = []
    type_targets: List[int] = []
    level_targets: List[int] = []
    parent_targets: List[int] = []

    for node in nodes:
        bbox = node["bbox"]
        node_type = node["type"]
        type_id = TYPE_TO_ID.get(node_type, TYPE_TO_ID["Decorate"])
        features.append(node_features(bbox, width, height, type_id))
        type_targets.append(type_id)
        level_targets.append(int(node["level"]))
        parent_id = node.get("parentId") or "screen_0000"
        parent_targets.append(id_to_index.get(parent_id, 0))
    return features, type_targets, level_targets, parent_targets


def node_features(bbox: Dict[str, float], width: float, height: float, type_id: int) -> List[float]:
    x = float(bbox["x"])
    y = float(bbox["y"])
    w = float(bbox["w"])
    h = float(bbox["h"])
    area = w * h / max(width * height, 1.0)
    aspect = min(8.0, w / max(h, 1.0)) / 8.0
    values = [
        x / width,
        y / height,
        w / width,
        h / height,
        (x + w / 2.0) / width,
        (y + h / 2.0) / height,
        area,
        aspect,
    ]
    one_hot = [0.0 for _ in TYPE_NAMES]
    one_hot[type_id] = 1.0
    return values + one_hot


def batches(items: List[Tuple[List[List[float]], List[int], List[int], List[int]]], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def collate(batch):
    max_nodes = max(len(item[0]) for item in batch)
    input_dim = len(batch[0][0][0])
    features = torch.zeros(len(batch), max_nodes, input_dim)
    type_targets = torch.zeros(len(batch), max_nodes, dtype=torch.long)
    level_targets = torch.zeros(len(batch), max_nodes, dtype=torch.long)
    parent_targets = torch.zeros(len(batch), max_nodes, dtype=torch.long)
    mask = torch.zeros(len(batch), max_nodes, dtype=torch.bool)

    for batch_index, item in enumerate(batch):
        item_features, item_types, item_levels, item_parents = item
        n = len(item_features)
        features[batch_index, :n] = torch.tensor(item_features, dtype=torch.float32)
        type_targets[batch_index, :n] = torch.tensor(item_types, dtype=torch.long)
        level_targets[batch_index, :n] = torch.tensor(item_levels, dtype=torch.long)
        parent_targets[batch_index, :n] = torch.tensor(item_parents, dtype=torch.long)
        mask[batch_index, :n] = True
    return features, type_targets, level_targets, parent_targets, mask


def compute_loss(output, type_targets, level_targets, parent_targets, mask) -> torch.Tensor:
    valid = mask
    type_loss = F.cross_entropy(output["type_logits"][valid], type_targets[valid])
    level_loss = F.cross_entropy(output["level_logits"][valid], level_targets[valid])

    parent_logits = output["parent_logits"].clone()
    parent_logits = parent_logits.masked_fill((~mask).unsqueeze(1), -1e4)
    row_index = torch.arange(mask.shape[1]).unsqueeze(0).expand_as(mask)
    parent_valid = mask & (row_index > 0)
    parent_loss = F.cross_entropy(parent_logits[parent_valid], parent_targets[parent_valid])
    return type_loss + 0.6 * level_loss + 0.8 * parent_loss


@torch.no_grad()
def evaluate(model, items, batch_size: int) -> float:
    if not items:
        return 0.0
    model.eval()
    total = 0.0
    steps = 0
    for batch in batches(items, batch_size):
        features, type_targets, level_targets, parent_targets, mask = collate(batch)
        output = model(features, padding_mask=~mask)
        total += float(compute_loss(output, type_targets, level_targets, parent_targets, mask).item())
        steps += 1
    return total / max(steps, 1)


if __name__ == "__main__":
    main()
