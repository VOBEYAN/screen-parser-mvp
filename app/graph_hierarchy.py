from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F

from .graph_transformer import GraphTransformerHierarchyModel
from .hierarchy import TYPE_LEVEL, normalize_node_type, parent_score
from .schemas import BBox, Detection, Node, Relation


TYPE_NAMES = ["Screen", "Panel", "Title", "Chart", "Table", "Map", "MetricCard", "Border", "Decorate", "Filter"]
TYPE_TO_ID = {name: idx for idx, name in enumerate(TYPE_NAMES)}


class GraphHierarchyParser:
    def __init__(self, checkpoint_path: str):
        state = torch.load(checkpoint_path, map_location="cpu")
        self.type_names = state.get("type_names", TYPE_NAMES)
        self.type_to_id = {name: idx for idx, name in enumerate(self.type_names)}
        self.input_dim = int(state.get("input_dim", 8 + len(self.type_names)))
        self.model = GraphTransformerHierarchyModel(
            input_dim=self.input_dim,
            hidden_dim=int(state.get("hidden_dim", 96)),
            num_layers=int(state.get("layers", 2)),
            num_heads=int(state.get("heads", 4)),
            num_types=len(self.type_names),
        )
        self.model.load_state_dict(state["model"])
        self.model.eval()

    @torch.no_grad()
    def parse(self, detections: Iterable[Detection], image_width: int, image_height: int) -> tuple[List[Node], List[Relation]]:
        nodes = self._build_initial_nodes(detections, image_width, image_height)
        features = torch.tensor([node_features(node, image_width, image_height, self.type_to_id) for node in nodes], dtype=torch.float32).unsqueeze(0)
        output = self.model(features)

        type_probs = F.softmax(output["type_logits"][0], dim=-1)
        level_probs = F.softmax(output["level_logits"][0], dim=-1)
        parent_probs = F.softmax(output["parent_logits"][0], dim=-1)

        for index, node in enumerate(nodes):
            if index == 0:
                continue
            type_conf, type_id = torch.max(type_probs[index], dim=-1)
            if float(type_conf) >= 0.45 and int(type_id) < len(self.type_names):
                node.type = self.type_names[int(type_id)]
            level_conf, level_id = torch.max(level_probs[index], dim=-1)
            if float(level_conf) >= 0.45:
                node.level = int(level_id)

        self._assign_graph_parents(nodes, parent_probs)
        relations = self._build_relations(nodes)
        return nodes, relations

    def _build_initial_nodes(self, detections: Iterable[Detection], image_width: int, image_height: int) -> List[Node]:
        nodes = [
            Node(
                node_id="screen_0000",
                bbox=BBox(0.0, 0.0, float(image_width), float(image_height)),
                type="Screen",
                level=0,
                confidence=1.0,
            )
        ]
        for index, detection in enumerate(detections):
            node_type = normalize_node_type(detection.class_name)
            nodes.append(
                Node(
                    node_id=f"node_{index:04d}",
                    bbox=detection.bbox,
                    type=node_type,
                    level=TYPE_LEVEL.get(node_type, 4),
                    confidence=detection.confidence,
                    detection_id=detection.detection_id,
                    component_id=detection.component_id,
                    features=detection.features,
                )
            )
        return nodes

    def _assign_graph_parents(self, nodes: List[Node], parent_probs: torch.Tensor) -> None:
        for child_index, child in enumerate(nodes):
            if child_index == 0:
                child.parent_id = None
                continue
            best_index = 0
            best_score = float(parent_probs[child_index, 0])
            for parent_index, parent in enumerate(nodes):
                if parent_index == child_index:
                    continue
                if parent_index != 0 and parent.bbox.area <= child.bbox.area:
                    continue
                rule_score = parent_score(parent, child)
                graph_score = float(parent_probs[child_index, parent_index])
                score = graph_score + 0.8 * rule_score
                if score > best_score:
                    best_score = score
                    best_index = parent_index
            child.parent_id = nodes[best_index].node_id

    def _build_relations(self, nodes: List[Node]) -> List[Relation]:
        relations: List[Relation] = []
        for node in nodes:
            if node.parent_id:
                relations.append(Relation(source=node.parent_id, target=node.node_id, type="contains", score=1.0))
        for a, b in combinations([node for node in nodes if node.parent_id], 2):
            if a.parent_id == b.parent_id:
                relations.append(Relation(source=a.node_id, target=b.node_id, type="sibling", score=1.0))
        return relations


def node_features(node: Node, width: int, height: int, type_to_id: Dict[str, int]) -> List[float]:
    bbox = node.bbox
    area = bbox.area / max(float(width * height), 1.0)
    aspect = min(8.0, bbox.w / max(bbox.h, 1.0)) / 8.0
    type_id = type_to_id.get(node.type, type_to_id.get("Decorate", 0))
    values = [
        bbox.x / width,
        bbox.y / height,
        bbox.w / width,
        bbox.h / height,
        (bbox.x + bbox.w / 2.0) / width,
        (bbox.y + bbox.h / 2.0) / height,
        area,
        aspect,
    ]
    one_hot = [0.0 for _ in type_to_id]
    if type_id < len(one_hot):
        one_hot[type_id] = 1.0
    return values + one_hot
