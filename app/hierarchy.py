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
    "Content": 3,
    "Decorate": 3,
    "Filter": 4,
    "Chart": 4,
    "Table": 4,
    "Map": 4,
    "MetricCard": 4,
}

COMPATIBILITY = {
    "Screen": {"Region", "Panel", "Border", "Title", "Decorate", "Filter", "Chart", "Table", "Map", "MetricCard"},
    "Region": {"Panel", "Border", "Title", "Decorate", "Filter", "Chart", "Table", "Map", "MetricCard"},
    "Panel": {"Border", "Title", "Content", "Decorate", "Filter", "Chart", "Table", "Map", "MetricCard"},
    "Border": {"Title", "Content", "Decorate", "Filter", "Chart", "Table", "Map", "MetricCard"},
    "Content": {"Decorate", "Filter", "Chart", "Table", "Map", "MetricCard"},
}

CONTENT_CHILD_TYPES = {"Chart", "Table", "Map", "MetricCard", "Filter"}
REGION_CHILD_TYPES = {"Panel", "Border", "Title", "Decorate", "Filter", "Chart", "Table", "Map", "MetricCard"}


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
        insert_region_nodes(nodes, image_width, image_height)
        self._insert_content_nodes(nodes)
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

    def _insert_content_nodes(self, nodes: List[Node]) -> None:
        next_index = sum(1 for node in nodes if node.node_id.startswith("content_"))
        parents = [node for node in list(nodes) if node.type in {"Panel", "Border"}]
        for parent in parents:
            children = [
                node
                for node in nodes
                if node.parent_id == parent.node_id and node.type in CONTENT_CHILD_TYPES
            ]
            if not children:
                continue
            content_bbox = union_bbox([node.bbox for node in children], pad=4.0)
            content = Node(
                node_id=f"content_{next_index:04d}",
                bbox=content_bbox,
                type="Content",
                level=3,
                confidence=round(sum(node.confidence for node in children) / max(len(children), 1), 4),
                parent_id=parent.node_id,
                features={"generated": True, "childCount": len(children)},
            )
            next_index += 1
            nodes.append(content)
            for child in children:
                child.parent_id = content.node_id
                child.level = max(child.level, 4)

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
    if "content" in lower:
        return "Content"
    if "panel" in lower or "region" in lower:
        return "Panel"
    if "metric" in lower or "number" in lower:
        return "MetricCard"
    if "chart" in lower or "bar" in lower or "line" in lower or "pie" in lower:
        return "Chart"
    return "Decorate"


def insert_region_nodes(nodes: List[Node], image_width: int, image_height: int) -> None:
    if any(node.type == "Region" for node in nodes):
        return

    screen_children = [
        node
        for node in nodes
        if node.parent_id == "screen_0000" and node.type in REGION_CHILD_TYPES
    ]
    if not screen_children:
        return

    header_children = [
        node
        for node in screen_children
        if node.type == "Title" and node.bbox.y <= image_height * 0.16
    ]
    body_children = [node for node in screen_children if node not in header_children]

    next_index = 0
    if header_children:
        header = Node(
            node_id=f"region_{next_index:04d}",
            bbox=clamp_bbox(union_bbox([node.bbox for node in header_children], pad=8.0), image_width, image_height),
            type="Region",
            level=1,
            confidence=1.0,
            parent_id="screen_0000",
            features={"generated": True, "role": "headerRegion", "childCount": len(header_children)},
        )
        next_index += 1
        nodes.append(header)
        for child in header_children:
            child.parent_id = header.node_id
            child.level = max(child.level, TYPE_LEVEL.get(child.type, child.level))

    groups: Dict[str, List[Node]] = {"left": [], "center": [], "right": []}
    for child in body_children:
        center_x, _ = child.bbox.center
        if center_x < image_width * 0.34:
            groups["left"].append(child)
        elif center_x > image_width * 0.66:
            groups["right"].append(child)
        else:
            groups["center"].append(child)

    for role in ["left", "center", "right"]:
        children = groups[role]
        if not children:
            continue
        region = Node(
            node_id=f"region_{next_index:04d}",
            bbox=clamp_bbox(union_bbox([node.bbox for node in children], pad=8.0), image_width, image_height),
            type="Region",
            level=1,
            confidence=round(sum(node.confidence for node in children) / max(len(children), 1), 4),
            parent_id="screen_0000",
            features={"generated": True, "role": f"{role}Region", "childCount": len(children)},
        )
        next_index += 1
        nodes.append(region)
        for child in children:
            child.parent_id = region.node_id
            if child.type in {"Panel", "Border"}:
                child.level = 2


def clamp_bbox(bbox: BBox, image_width: int, image_height: int) -> BBox:
    x1 = max(0.0, min(float(image_width - 1), bbox.x))
    y1 = max(0.0, min(float(image_height - 1), bbox.y))
    x2 = max(x1 + 1.0, min(float(image_width), bbox.right))
    y2 = max(y1 + 1.0, min(float(image_height), bbox.bottom))
    return BBox(x1, y1, x2 - x1, y2 - y1)


def union_bbox(boxes: List[BBox], pad: float = 0.0) -> BBox:
    if not boxes:
        return BBox(0.0, 0.0, 1.0, 1.0)
    x1 = min(box.x for box in boxes) - pad
    y1 = min(box.y for box in boxes) - pad
    x2 = max(box.right for box in boxes) + pad
    y2 = max(box.bottom for box in boxes) + pad
    return BBox(x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))
