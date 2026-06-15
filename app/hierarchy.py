from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Optional

from .geometry import center_inside, containment, iou, normalized_center_distance
from .schemas import BBox, Detection, Node, Relation


TYPE_LEVEL = {
    "Screen": 0,
    "Region": 1,
    "Panel": 2,
    "Border": 3,
    "Title": 3,
    "Decorate": 3,
    "Filter": 4,
    "Chart": 4,
    "Table": 4,
    "Map": 4,
    "MetricCard": 4,
}

COMPATIBILITY = {
    "Screen": {"Region", "Panel", "Title", "Decorate", "Filter", "Chart", "Table", "Map", "MetricCard"},
    "Region": {"Panel", "Title", "Decorate", "Filter", "Chart", "Table", "Map", "MetricCard"},
    "Panel": {"Border", "Title", "Decorate", "Filter", "Chart", "Table", "Map", "MetricCard"},
    "Border": {"Title", "Decorate", "Filter", "Chart", "Table", "Map", "MetricCard"},
}


class HierarchyParser:
    def parse(self, detections: Iterable[Detection], image_width: int, image_height: int) -> tuple[List[Node], List[Relation]]:
        root = Node(
            node_id="screen_0000",
            bbox=BBox(0.0, 0.0, float(image_width), float(image_height)),
            type="Screen",
            level=0,
            confidence=1.0,
            parent_id=None,
        )
        nodes = [root]

        for index, detection in enumerate(detections):
            node_type = normalize_node_type(detection.class_name)
            nodes.append(
                Node(
                    node_id=f"node_{index:04d}",
                    bbox=detection.bbox,
                    type=node_type,
                    level=TYPE_LEVEL.get(node_type, 4),
                    confidence=detection.confidence,
                    parent_id=None,
                    detection_id=detection.detection_id,
                    component_id=detection.component_id,
                    features=detection.features,
                )
            )

        self._promote_container_nodes(nodes)
        self._assign_parents(nodes)
        relations = self._build_relations(nodes)
        return nodes, relations

    def _promote_container_nodes(self, nodes: List[Node]) -> None:
        for node in nodes:
            if node.type in {"Screen", "Panel", "Border", "Title"}:
                continue
            contained = 0
            for other in nodes:
                if node.node_id == other.node_id or other.type == "Screen":
                    continue
                if node.bbox.area <= other.bbox.area:
                    continue
                if containment(node.bbox, other.bbox) >= 0.75:
                    contained += 1
            if contained >= 3 and node.bbox.area > 25000:
                node.type = "Panel"
                node.level = 2

    def _assign_parents(self, nodes: List[Node]) -> None:
        for child in nodes:
            if child.type == "Screen":
                continue
            best_parent: Optional[Node] = None
            best_score = 0.0
            for parent in nodes:
                if parent.node_id == child.node_id:
                    continue
                if parent.bbox.area <= child.bbox.area:
                    continue
                score = parent_score(parent, child)
                if score > best_score:
                    best_parent = parent
                    best_score = score

            child.parent_id = best_parent.node_id if best_parent and best_score >= 0.42 else "screen_0000"
            if child.parent_id == "screen_0000" and child.level > 2:
                child.level = max(2, child.level - 1)

    def _build_relations(self, nodes: List[Node]) -> List[Relation]:
        relations: List[Relation] = []
        node_by_id: Dict[str, Node] = {node.node_id: node for node in nodes}
        for node in nodes:
            if node.parent_id:
                parent = node_by_id.get(node.parent_id)
                score = parent_score(parent, node) if parent else 1.0
                relations.append(Relation(source=node.parent_id, target=node.node_id, type="contains", score=round(score, 4)))

        for a, b in combinations([node for node in nodes if node.parent_id], 2):
            if a.parent_id == b.parent_id and a.node_id != b.node_id:
                relations.append(Relation(source=a.node_id, target=b.node_id, type="sibling", score=1.0))
        return relations


def parent_score(parent: Optional[Node], child: Node) -> float:
    if parent is None:
        return 0.0
    contains = containment(parent.bbox, child.bbox)
    if contains <= 0.05:
        return 0.0

    area_ratio = min(1.0, child.bbox.area / max(parent.bbox.area, 1.0))
    area_score = 1.0 - area_ratio
    center_score = 1.0 if center_inside(parent.bbox, child.bbox) else 0.0
    compat_score = 1.0 if compatible(parent.type, child.type) else 0.35
    distance_score = max(0.0, 1.0 - normalized_center_distance(parent.bbox, child.bbox))
    overlap_penalty = 0.25 if iou(parent.bbox, child.bbox) > 0.85 else 0.0

    return max(
        0.0,
        0.42 * contains
        + 0.2 * area_score
        + 0.18 * compat_score
        + 0.12 * center_score
        + 0.08 * distance_score
        - overlap_penalty,
    )


def compatible(parent_type: str, child_type: str) -> bool:
    return child_type in COMPATIBILITY.get(parent_type, set())


def normalize_node_type(class_name: str) -> str:
    value = class_name.strip()
    if value in TYPE_LEVEL:
        return value
    lower = value.lower()
    if "table" in lower:
        return "Table"
    if "map" in lower:
        return "Map"
    if "title" in lower or "text" in lower:
        return "Title"
    if "border" in lower:
        return "Border"
    if "panel" in lower or "region" in lower:
        return "Panel"
    if "metric" in lower or "number" in lower:
        return "MetricCard"
    if "chart" in lower or "bar" in lower or "line" in lower or "pie" in lower:
        return "Chart"
    return "Decorate"
