from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Set, Tuple

from .geometry import containment, intersection, iou, overlap_ratio
from .schemas import Node, OverlapIssue


class OverlapDetector:
    def detect(self, nodes: Iterable[Node]) -> List[OverlapIssue]:
        node_list = [node for node in nodes if node.type != "Screen"]
        parent_pairs = self._ancestor_pairs(node_list)
        issues: List[OverlapIssue] = []

        for a, b in combinations(node_list, 2):
            if (a.node_id, b.node_id) in parent_pairs or (b.node_id, a.node_id) in parent_pairs:
                continue
            if a.parent_id != b.parent_id:
                continue
            if self._allowed_overlap(a, b):
                continue

            pair_iou = iou(a.bbox, b.bbox)
            pair_ratio = overlap_ratio(a.bbox, b.bbox)
            if pair_iou < 0.05 and pair_ratio < 0.15:
                continue
            severity = "warning"
            if pair_iou >= 0.15 or pair_ratio >= 0.35:
                severity = "error"
            issues.append(
                OverlapIssue(
                    source=a.node_id,
                    target=b.node_id,
                    iou=round(pair_iou, 4),
                    overlap_ratio=round(pair_ratio, 4),
                    severity=severity,
                    intersection=intersection(a.bbox, b.bbox),
                )
            )

        return issues

    def _ancestor_pairs(self, nodes: List[Node]) -> Set[Tuple[str, str]]:
        by_id: Dict[str, Node] = {node.node_id: node for node in nodes}
        pairs: Set[Tuple[str, str]] = set()
        for node in nodes:
            parent_id = node.parent_id
            while parent_id and parent_id in by_id:
                pairs.add((parent_id, node.node_id))
                parent_id = by_id[parent_id].parent_id
        return pairs

    def _allowed_overlap(self, a: Node, b: Node) -> bool:
        types = {a.type, b.type}
        if "Border" in types and ("Panel" in types or "Decorate" in types):
            return True
        if "Title" in types and ("Panel" in types or "Border" in types):
            return True
        if self._internal_visual_part(a, b):
            return True
        if min(a.bbox.area, b.bbox.area) < 800:
            return True
        return False

    def _internal_visual_part(self, a: Node, b: Node) -> bool:
        primary, secondary = (a, b) if a.bbox.area >= b.bbox.area else (b, a)
        area_ratio = secondary.bbox.area / max(primary.bbox.area, 1.0)
        inside = containment(primary.bbox, secondary.bbox)

        if primary.type in {"Chart", "Table", "Map"} and secondary.type in {"Title", "Decorate", "MetricCard"}:
            return inside >= 0.62 and area_ratio <= 0.18

        if primary.type == "Panel" and secondary.type in {"Decorate", "MetricCard"}:
            return inside >= 0.72 and area_ratio <= 0.08

        return False
