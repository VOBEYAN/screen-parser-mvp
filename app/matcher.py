from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .component_library import ComponentLibrary
from .schemas import ComponentRecord, Node
from .visual_matcher import VisualReferenceLibrary, extract_crop_features, load_bgr_image


TYPE_TO_CATEGORIES = {
    "Chart": ["Bars", "Lines", "Pies", "Scatters", "Areas", "Funnels", "WordClouds", "Mores"],
    "Table": ["Tables"],
    "Map": ["Maps", "Biz"],
    "Panel": ["Borders", "Decorates"],
    "Border": ["Borders"],
    "Title": ["Title", "Texts"],
    "MetricCard": ["Mores", "Biz", "Texts"],
    "Image": ["Biz", "Three", "Decorates"],
    "Filter": ["Inputs"],
    "Decorate": ["Decorates", "Mores"],
}
STRUCTURE_ONLY_TYPES = {"Screen", "Region", "Content"}

class ComponentMatcher:
    def __init__(self, library: ComponentLibrary, visual_library: Optional[VisualReferenceLibrary] = None):
        self.library = library
        self.visual_library = visual_library or VisualReferenceLibrary([])

    def match_nodes(self, nodes: Iterable[Node], top_k: int = 5, image_path: Optional[str] = None) -> None:
        image = load_bgr_image(image_path) if image_path and self.visual_library.enabled else None
        for node in nodes:
            if node.type in STRUCTURE_ONLY_TYPES:
                continue
            crop_features = extract_crop_features(image, node.bbox) if image is not None else None
            candidates = self.match_node(node, top_k=top_k, crop_features=crop_features)
            node.candidates = candidates
            if candidates:
                node.component_id = candidates[0]["componentId"]

    def match_node(self, node: Node, top_k: int = 5, crop_features: Optional[Dict[str, object]] = None) -> List[Dict[str, object]]:
        if node.component_id and node.component_id in self.library.by_key:
            record = self.library.by_key[node.component_id]
            return [
                {
                    "componentId": record.key,
                    "title": record.title,
                    "category": record.category,
                    "schema": record.schema,
                    "score": 0.99,
                    "matchMode": "detector_component_id",
                }
            ][:top_k]

        categories = TYPE_TO_CATEGORIES.get(node.type, [])
        records = self.library.filter_by_categories(categories) if categories else self.library.records
        scored = []
        for record in records:
            base_score = self._score(record, node, categories)
            visual_score = self._visual_score(record, crop_features)
            score = self._merge_scores(base_score, visual_score)
            scored.append(
                {
                    "componentId": record.key,
                    "title": record.title,
                    "category": record.category,
                    "schema": record.schema,
                    "score": round(score, 4),
                    "baseScore": round(base_score, 4),
                    "visualScore": round(visual_score, 4) if visual_score is not None else None,
                    "matchMode": "visual_reference" if visual_score is not None else "catalog_rules",
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def _score(self, record: ComponentRecord, node: Node, categories: List[str]) -> float:
        score = 0.18
        if record.category in categories:
            score += 0.42

        aspect = float(node.features.get("aspectRatio", node.bbox.w / max(node.bbox.h, 1.0)))
        description = record.description
        if aspect >= 2.2 and ("横向" in description or "延展" in description):
            score += 0.08
        if aspect < 1.2 and ("中心" in description or "均衡" in description):
            score += 0.06

        if node.type == "Chart":
            score += chart_keyword_bonus(record)
        if node.type == "Table" and ("表格" in record.title or "列表" in record.title):
            score += 0.18
        if node.type == "Panel" and "边框" in record.title:
            score += 0.18
        if node.type == "Title" and ("标题" in record.title or "文字" in record.title):
            score += 0.18
        if node.type == "MetricCard" and any(token in record.title for token in ["数字", "状态", "能量", "告警"]):
            score += 0.14

        return min(score, 0.99)

    def _visual_score(self, record: ComponentRecord, crop_features: Optional[Dict[str, object]]) -> Optional[float]:
        if not crop_features or not self.visual_library.enabled:
            return None
        return self.visual_library.score(record.key, crop_features)

    def _merge_scores(self, base_score: float, visual_score: Optional[float]) -> float:
        if visual_score is None:
            return min(base_score, 0.99)
        return min(0.99, 0.34 * base_score + 0.66 * visual_score)


def chart_keyword_bonus(record: ComponentRecord) -> float:
    title = record.title
    category = record.category
    if category in {"Bars", "Lines", "Pies", "Maps", "Scatters"}:
        return 0.14
    if any(token in title for token in ["柱", "折线", "饼", "地图", "漏斗", "雷达"]):
        return 0.12
    return 0.0
