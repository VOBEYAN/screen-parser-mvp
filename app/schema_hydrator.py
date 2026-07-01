from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Dict, Iterable, List, Optional

from .component_library import ComponentLibrary
from .schemas import Node, ComponentRecord


NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?%?")
LABEL_RE = re.compile(r"[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9_-]{0,18}")
DATE_LABEL_RE = re.compile(r"\d{1,2}-\d{1,2}")


def build_ai_schema_components(
    nodes: Iterable[Node],
    library: ComponentLibrary,
    classifier_summary: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    node_list = list(nodes)
    components: List[Dict[str, Any]] = []
    for node in node_list:
        if node.type in {"Screen", "Region", "Content"}:
            continue
        record = library.by_key.get(node.component_id or "")
        if not record:
            candidate = node.candidates[0] if node.candidates else {}
            record = library.by_key.get(str(candidate.get("componentId") or ""))
        if not record:
            continue

        classifier = node.features.get("contentClassifier") or {}
        record = repair_component_record_for_schema(node, record, library, node_list)
        component_bbox = repair_component_bbox_for_schema(node, record, classifier, node_list)
        dataset = infer_dataset(node, classifier, record.category)
        repeated_metric_components = build_repeated_metric_components(node, record, library, classifier)
        if repeated_metric_components:
            components.extend(repeated_metric_components)
            continue
        z_index = z_index_for_node(node.type)
        option_blueprint = library.option_blueprint(record.key)
        schema_shape = library.option_shape(record.key)
        facts = build_recognition_facts(
            component_id=record.key,
            category=record.category,
            dataset=dataset,
            classifier=classifier,
            bbox=component_bbox,
            schema_shape=schema_shape,
        )
        option_patch = build_option_patch(
            record.category,
            record.key,
            dataset,
            classifier,
            option_blueprint=option_blueprint,
            schema_shape=schema_shape,
            bbox=component_bbox,
        )
        hydrated_option = hydrate_option_from_facts(option_blueprint, option_patch, facts, schema_shape)
        component = {
            "id": f"schema_{node.node_id}",
            "nodeId": node.node_id,
            "componentId": record.key,
            "title": record.title,
            "category": record.category,
            "categoryName": record.category_name,
            "schemaShape": schema_shape,
            "bbox": component_bbox,
            "attr": {
                "x": round(float(component_bbox["x"]), 2),
                "y": round(float(component_bbox["y"]), 2),
                "w": round(float(component_bbox["w"]), 2),
                "h": round(float(component_bbox["h"]), 2),
                "offsetX": 0,
                "offsetY": 0,
                "zIndex": z_index,
            },
            "chartConfig": {
                "key": record.key,
                "chartKey": record.chart_key,
                "conKey": record.con_key,
                "title": record.title,
                "category": record.category,
                "categoryName": record.category_name,
                "chartFrame": record.chart_frame,
                "package": infer_package(record.key, record.category),
            },
            "option": hydrated_option,
            "optionPatch": option_patch,
            "recognitionFacts": facts,
            "optionHydrationVersion": "schema-aware-v1",
            "dataSource": {
                "source": "ocr+vlm",
                "ocrText": str(classifier.get("paddleOcrText") or ""),
                "modelText": str(classifier.get("text") or ""),
                "contentType": str(classifier.get("contentType") or ""),
            },
        }
        components.append(component)
        virtual = build_virtual_inner_component(node, node_list, library)
        if virtual:
            components.append(virtual)
    components.extend(build_virtual_panel_title_components(node_list, library, components))
    components.extend(build_global_virtual_components(node_list, library, components, classifier_summary or {}))
    components = suppress_overlapping_components(components)
    return sorted(components, key=lambda item: int((item.get("attr") or {}).get("zIndex") or 1))


CONTENT_CATEGORIES = {
    "Bars",
    "Lines",
    "Pies",
    "Scatters",
    "Areas",
    "Funnels",
    "WordClouds",
    "Tables",
    "Maps",
    "Mores",
    "Biz",
    "Three",
}

INNER_CONTENT_TYPES = {"Chart", "Table", "Map", "MetricCard", "Filter", "Image"}


def build_global_virtual_components(
    nodes: List[Node],
    library: ComponentLibrary,
    existing_components: List[Dict[str, Any]],
    classifier_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    components: List[Dict[str, Any]] = []
    semantic_components = build_schema_semantic_aggregate_components(nodes, library)
    if semantic_components:
        return semantic_components

    if any(item.get("componentId") == "AIShield" for item in existing_components):
        return components

    record = library.by_key.get("AIShield")
    if not record:
        return components

    text = " ".join(
        item
        for item in [
            collect_classifier_text(nodes),
            str(classifier_summary.get("paddleOcrFullText") or ""),
        ]
        if item
    )
    if not looks_like_ai_shield_scene(text):
        return components

    screen = next((node for node in nodes if node.type == "Screen"), None)
    screen_w = screen.bbox.w if screen else 1920.0
    screen_h = screen.bbox.h if screen else 1080.0
    bbox = ai_shield_bbox(screen_w, screen_h)
    classifier = {
        "contentType": "ai_shield",
        "componentType": "Image",
        "text": text,
        "paddleOcrText": text,
        "visualEvidence": "center shield with AI text, circular base, multiple risk-warning metric nodes",
        "palette": collect_classifier_colors(nodes),
    }
    option_blueprint = library.option_blueprint(record.key)
    schema_shape = library.option_shape(record.key)
    source_dataset = ai_shield_dataset_from_text(text)
    option_patch = build_option_patch(
        record.category,
        record.key,
        source_dataset,
        classifier,
        option_blueprint=option_blueprint,
        schema_shape=schema_shape,
        bbox=bbox,
    )
    facts = build_recognition_facts(record.key, record.category, source_dataset, classifier, bbox, schema_shape)
    hydrated_option = hydrate_option_from_facts(option_blueprint, option_patch, facts, schema_shape)
    return [
        {
            "id": "schema_virtual_ai_shield",
            "nodeId": "virtual_ai_shield",
            "virtual": True,
            "componentId": record.key,
            "title": record.title,
            "category": record.category,
            "categoryName": record.category_name,
            "schemaShape": schema_shape,
            "bbox": bbox,
            "attr": {
                "x": round(float(bbox["x"]), 2),
                "y": round(float(bbox["y"]), 2),
                "w": round(float(bbox["w"]), 2),
                "h": round(float(bbox["h"]), 2),
                "offsetX": 0,
                "offsetY": 0,
                "zIndex": 8,
            },
            "chartConfig": {
                "key": record.key,
                "chartKey": record.chart_key,
                "conKey": record.con_key,
                "title": record.title,
                "category": record.category,
                "categoryName": record.category_name,
                "chartFrame": record.chart_frame,
                "package": infer_package(record.key, record.category),
            },
            "option": hydrated_option,
            "optionPatch": option_patch,
            "recognitionFacts": facts,
            "optionHydrationVersion": "schema-aware-v1",
            "dataSource": {
                "source": "ocr+layout+global-repair",
                "ocrText": text[:500],
                "modelText": text[:500],
                "contentType": "ai_shield",
            },
        }
    ]


def build_schema_semantic_aggregate_components(nodes: List[Node], library: ComponentLibrary) -> List[Dict[str, Any]]:
    components: List[Dict[str, Any]] = []
    used_node_ids: set[str] = set()
    for record in library.records:
        schema_shape = library.option_shape(record.key)
        if schema_shape.get("datasetKind") != "object.bizNodes":
            continue
        terms = schema_terms_from_option(library.option_blueprint(record.key))
        if len(terms) < 3:
            continue
        contributors, matched_terms = semantic_contributors_for_terms(nodes, terms, used_node_ids)
        if not is_strong_semantic_aggregate(contributors, matched_terms, terms):
            continue
        option_blueprint = library.option_blueprint(record.key)
        bbox = semantic_aggregate_bbox(contributors, nodes, option_blueprint)
        text = " ".join(semantic_text_for_node(node) for node in contributors if semantic_text_for_node(node)).strip()
        classifier = {
            "contentType": "semantic_biz_component",
            "componentType": "Image",
            "text": text,
            "paddleOcrText": text,
            "visualEvidence": f"schema terms matched: {', '.join(sorted(matched_terms))}",
            "palette": collect_classifier_colors(contributors),
        }
        source_dataset = {"dimensions": ["name", "value"], "source": []}
        option_patch = build_option_patch(
            record.category,
            record.key,
            source_dataset,
            classifier,
            option_blueprint=option_blueprint,
            schema_shape=schema_shape,
            bbox=bbox,
        )
        facts = build_recognition_facts(record.key, record.category, source_dataset, classifier, bbox, schema_shape)
        hydrated_option = hydrate_option_from_facts(option_blueprint, option_patch, facts, schema_shape)
        components.append(
            {
                "id": f"schema_virtual_{record.key}_semantic",
                "nodeId": f"virtual_{record.key}_semantic",
                "virtual": True,
                "componentId": record.key,
                "title": record.title,
                "category": record.category,
                "categoryName": record.category_name,
                "schemaShape": schema_shape,
                "bbox": bbox,
                "contributorNodeIds": [node.node_id for node in contributors],
                "matchedSchemaTerms": sorted(matched_terms),
                "attr": {
                    "x": round(float(bbox["x"]), 2),
                    "y": round(float(bbox["y"]), 2),
                    "w": round(float(bbox["w"]), 2),
                    "h": round(float(bbox["h"]), 2),
                    "offsetX": 0,
                    "offsetY": 0,
                    "zIndex": 8,
                },
                "chartConfig": {
                    "key": record.key,
                    "chartKey": record.chart_key,
                    "conKey": record.con_key,
                    "title": record.title,
                    "category": record.category,
                    "categoryName": record.category_name,
                    "chartFrame": record.chart_frame,
                    "package": infer_package(record.key, record.category),
                },
                "option": hydrated_option,
                "optionPatch": option_patch,
                "recognitionFacts": facts,
                "optionHydrationVersion": "schema-aware-v1",
                "dataSource": {
                    "source": "ocr+schema-semantic-aggregate",
                    "ocrText": text[:500],
                    "modelText": text[:500],
                    "contentType": "semantic_biz_component",
                },
            }
        )
        used_node_ids.update(node.node_id for node in contributors)
    return components


def schema_terms_from_option(option_blueprint: Dict[str, Any]) -> List[str]:
    dataset = option_blueprint.get("dataset") if isinstance(option_blueprint, dict) else {}
    terms: set[str] = set()
    collect_schema_terms(dataset, terms)
    return sorted(terms, key=lambda item: (-len(item), item))


def collect_schema_terms(value: Any, terms: set[str]) -> None:
    if isinstance(value, str):
        text = value.strip()
        if is_semantic_schema_term(text):
            terms.add(text)
        return
    if isinstance(value, list):
        for item in value:
            collect_schema_terms(item, terms)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"label", "name", "title", "subtitle", "robotName", "statusText"} or key.endswith("Label"):
                collect_schema_terms(item, terms)
            elif isinstance(item, (list, dict)):
                collect_schema_terms(item, terms)


def is_semantic_schema_term(text: str) -> bool:
    if len(text) < 2:
        return False
    if not re.search(r"[\u4e00-\u9fff]", text):
        return False
    if looks_like_noise(text):
        return False
    return text not in {"正常", "异常", "告警", "成功", "失败", "默认", "数据项"}


def semantic_contributors_for_terms(
    nodes: List[Node],
    terms: List[str],
    used_node_ids: set[str],
) -> tuple[List[Node], set[str]]:
    contributors: List[Node] = []
    matched_terms: set[str] = set()
    for node in nodes:
        if node.node_id in used_node_ids or node.type not in {"Image", "MetricCard", "Title", "Decorate"}:
            continue
        text = semantic_text_for_node(node)
        if not text:
            continue
        hits = {term for term in terms if term in text}
        if not hits:
            continue
        contributors.append(node)
        matched_terms.update(hits)
    contributors.sort(key=lambda node: (node.bbox.y, node.bbox.x))
    return contributors, matched_terms


def semantic_text_for_node(node: Node) -> str:
    classifier = node.features.get("contentClassifier") or {}
    return " ".join(
        str(classifier.get(key) or "").strip()
        for key in ["text", "paddleOcrText"]
        if str(classifier.get(key) or "").strip()
    )


def is_strong_semantic_aggregate(contributors: List[Node], matched_terms: set[str], terms: List[str]) -> bool:
    if len(contributors) < 2:
        return False
    required_unique_terms = max(3, min(5, len(terms) // 2))
    if len(matched_terms) < required_unique_terms:
        return False
    union = union_node_bbox(contributors)
    if bbox_area(union) <= 0:
        return False
    contributor_area = sum(node.bbox.area for node in contributors)
    return contributor_area / max(1.0, bbox_area(union)) >= 0.08


def union_node_bbox(nodes: List[Node]) -> Dict[str, float]:
    left = min(node.bbox.x for node in nodes)
    top = min(node.bbox.y for node in nodes)
    right = max(node.bbox.right for node in nodes)
    bottom = max(node.bbox.bottom for node in nodes)
    return {
        "x": round(left, 2),
        "y": round(top, 2),
        "w": round(max(1.0, right - left), 2),
        "h": round(max(1.0, bottom - top), 2),
    }


def semantic_aggregate_bbox(
    contributors: List[Node],
    all_nodes: List[Node],
    option_blueprint: Dict[str, Any],
) -> Dict[str, float]:
    if semantic_aggregate_uses_scene_background(option_blueprint):
        screen = next((node for node in all_nodes if node.type == "Screen"), None)
        if screen:
            return screen.bbox.to_dict()
    return union_node_bbox(contributors)


def semantic_aggregate_uses_scene_background(option_blueprint: Dict[str, Any]) -> bool:
    visual = option_blueprint.get("visual") if isinstance(option_blueprint, dict) else None
    if not isinstance(visual, dict):
        return False
    background_image = str(visual.get("backgroundImage") or "").strip()
    if not background_image:
        return False
    return bool(visual.get("showBackground", True))


def collect_classifier_text(nodes: List[Node]) -> str:
    parts: List[str] = []
    for node in nodes:
        classifier = node.features.get("contentClassifier") or {}
        for key in ["text", "paddleOcrText", "textEvidence", "visualEvidence"]:
            value = str(classifier.get(key) or "").strip()
            if value:
                parts.append(value)
    return " ".join(parts)


def collect_classifier_colors(nodes: List[Node]) -> List[str]:
    colors: List[str] = []
    for node in nodes:
        classifier = node.features.get("contentClassifier") or {}
        colors.extend(extract_colors(classifier))
    seen = set()
    out: List[str] = []
    for color in colors:
        normalized = str(color).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(str(color))
        if len(out) >= 12:
            break
    return out


def looks_like_ai_shield_scene(text: str) -> bool:
    normalized = " ".join(str(text or "").split())
    if "AI" not in normalized:
        return False
    risk_count = normalized.count("风险预警")
    value_count = len(re.findall(r"\b32\s*条?", normalized))
    return risk_count >= 3 and value_count >= 3


def ai_shield_bbox(screen_w: float, screen_h: float) -> Dict[str, float]:
    width = screen_w * 0.56
    height = screen_h * 0.62
    x = (screen_w - width) / 2.0
    y = screen_h * 0.08
    return {
        "x": round(x, 2),
        "y": round(y, 2),
        "w": round(width, 2),
        "h": round(height, 2),
    }


def repair_component_record_for_schema(
    node: Node,
    record: ComponentRecord,
    library: ComponentLibrary,
    nodes: List[Node],
) -> ComponentRecord:
    classifier = node.features.get("contentClassifier") or {}
    if node.type not in {"Border", "Panel"} and should_repair_as_table(classifier):
        table_record = library.by_key.get("TableScrollBoard") or library.by_key.get("TablesBasic")
        if table_record:
            return table_record

    if should_render_panel_title_as_text(node, record, classifier, nodes):
        text_record = library.by_key.get("TextCommon")
        if text_record:
            return text_record
    return record


def should_repair_as_table(classifier: Dict[str, Any]) -> bool:
    content_type = str(classifier.get("contentType") or "")
    visual_form = str(classifier.get("visualForm") or classifier.get("llmVisualForm") or "")
    items = ocr_items(classifier)
    if content_type == "table" or "table" in visual_form.lower():
        return ocr_items_form_table(items)
    return False


def repair_component_bbox_for_schema(
    node: Node,
    record: ComponentRecord,
    classifier: Dict[str, Any],
    nodes: List[Node],
) -> Dict[str, float]:
    if record.key == "TextCommon" and should_render_panel_title_as_text(node, record, classifier, nodes):
        label_bbox = title_label_bbox_from_ocr_items(ocr_items(classifier), node.bbox.to_dict())
        if label_bbox:
            return label_bbox
    return node.bbox.to_dict()


def should_render_panel_title_as_text(
    node: Node,
    record: ComponentRecord,
    classifier: Dict[str, Any],
    nodes: List[Node],
) -> bool:
    if node.type != "Title":
        return False
    if record.category not in {"Title", "Decorates", "Texts"}:
        return False
    if not containing_panel_for_top_title(node, nodes):
        return False
    items = title_like_ocr_items(ocr_items(classifier))
    if not items:
        return False
    text_width = max((float((item.get("bbox") or {}).get("w") or 0) for item in items), default=0.0)
    return node.bbox.w >= 140 and text_width / max(1.0, node.bbox.w) <= 0.55


def containing_panel_for_top_title(node: Node, nodes: List[Node]) -> Optional[Node]:
    title_bbox = node.bbox.to_dict()
    for panel in nodes:
        if panel.node_id == node.node_id or panel.type not in {"Border", "Panel"}:
            continue
        panel_bbox = panel.bbox.to_dict()
        if bbox_containment(panel_bbox, title_bbox) < 0.48 and bbox_iou(panel_bbox, title_bbox) < 0.02:
            continue
        panel_top = panel.bbox.y
        if node.bbox.y <= panel_top + max(58.0, panel.bbox.h * 0.24):
            return panel
    return None


def title_like_ocr_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = []
    for item in items:
        text = str(item.get("text") or "").strip()
        if not text or looks_like_axis_or_legend_text(text):
            continue
        if NUMBER_RE.fullmatch(text) or DATE_LABEL_RE.fullmatch(text):
            continue
        if len(token_items(text)) > 3:
            continue
        filtered.append(item)
    return filtered


def title_label_bbox_from_ocr_items(
    items: List[Dict[str, Any]],
    fallback: Dict[str, float],
) -> Optional[Dict[str, float]]:
    filtered = title_like_ocr_items(items)
    if not filtered:
        return None
    left = min(float((item.get("bbox") or {}).get("x") or 0) for item in filtered)
    top = min(float((item.get("bbox") or {}).get("y") or 0) for item in filtered)
    right = max(float((item.get("bbox") or {}).get("x") or 0) + float((item.get("bbox") or {}).get("w") or 0) for item in filtered)
    bottom = max(float((item.get("bbox") or {}).get("y") or 0) + float((item.get("bbox") or {}).get("h") or 0) for item in filtered)
    pad_x = max(8.0, (right - left) * 0.18)
    pad_y = max(4.0, (bottom - top) * 0.28)
    x = max(float(fallback.get("x") or 0), left - pad_x)
    y = max(float(fallback.get("y") or 0), top - pad_y)
    max_right = float(fallback.get("x") or 0) + float(fallback.get("w") or 0)
    max_bottom = float(fallback.get("y") or 0) + float(fallback.get("h") or 0)
    return {
        "x": round(x, 2),
        "y": round(y, 2),
        "w": round(max(20.0, min(max_right, right + pad_x) - x), 2),
        "h": round(max(18.0, min(max_bottom, bottom + pad_y) - y), 2),
    }


def build_virtual_panel_title_components(
    nodes: List[Node],
    library: ComponentLibrary,
    existing_components: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    record = library.by_key.get("TextCommon")
    if not record:
        return []
    components: List[Dict[str, Any]] = []
    for node in nodes:
        if node.type not in {"Border", "Panel"}:
            continue
        if panel_has_existing_title(node, existing_components):
            continue
        title_items = panel_top_title_items(node)
        if not title_items:
            continue
        bbox = title_label_bbox_from_ocr_items(title_items, node.bbox.to_dict())
        if not bbox:
            continue
        text = " ".join(str(item.get("text") or "").strip() for item in title_items if str(item.get("text") or "").strip())
        classifier = {"text": text, "paddleOcrText": text, "paddleOcrItems": title_items, "contentType": "title"}
        option_blueprint = library.option_blueprint(record.key)
        schema_shape = library.option_shape(record.key)
        dataset = {"dimensions": ["name", "value"], "source": [{"name": text, "value": 0}]}
        option_patch = build_option_patch(
            record.category,
            record.key,
            dataset,
            classifier,
            option_blueprint=option_blueprint,
            schema_shape=schema_shape,
            bbox=bbox,
        )
        facts = build_recognition_facts(record.key, record.category, dataset, classifier, bbox, schema_shape)
        hydrated_option = hydrate_option_from_facts(option_blueprint, option_patch, facts, schema_shape)
        components.append(
            {
                "id": f"schema_{node.node_id}_title",
                "nodeId": f"{node.node_id}_title",
                "sourceNodeId": node.node_id,
                "virtual": True,
                "componentId": record.key,
                "title": record.title,
                "category": record.category,
                "categoryName": record.category_name,
                "schemaShape": schema_shape,
                "bbox": bbox,
                "attr": {
                    "x": round(float(bbox["x"]), 2),
                    "y": round(float(bbox["y"]), 2),
                    "w": round(float(bbox["w"]), 2),
                    "h": round(float(bbox["h"]), 2),
                    "offsetX": 0,
                    "offsetY": 0,
                    "zIndex": 20,
                },
                "chartConfig": {
                    "key": record.key,
                    "chartKey": record.chart_key,
                    "conKey": record.con_key,
                    "title": record.title,
                    "category": record.category,
                    "categoryName": record.category_name,
                    "chartFrame": record.chart_frame,
                    "package": infer_package(record.key, record.category),
                },
                "option": hydrated_option,
                "optionPatch": option_patch,
                "recognitionFacts": facts,
                "optionHydrationVersion": "schema-aware-v1",
                "dataSource": {
                    "source": "ocr+layout+panel-title",
                    "ocrText": text,
                    "modelText": text,
                    "contentType": "title",
                },
            }
        )
    return components


def panel_has_existing_title(node: Node, components: List[Dict[str, Any]]) -> bool:
    panel_bbox = node.bbox.to_dict()
    top_band = {
        "x": node.bbox.x,
        "y": node.bbox.y,
        "w": node.bbox.w,
        "h": max(44.0, min(node.bbox.h * 0.24, 72.0)),
    }
    for component in components:
        category = str(component.get("category") or "")
        if category not in {"Title", "Texts", "Decorates"}:
            continue
        bbox = component.get("bbox") or {}
        if bbox_containment(panel_bbox, bbox) < 0.25 and bbox_iou(panel_bbox, bbox) < 0.01:
            continue
        if bbox_iou(top_band, bbox) > 0.01 or bbox_containment(top_band, bbox) >= 0.25:
            return True
    return False


def panel_top_title_items(node: Node) -> List[Dict[str, Any]]:
    classifier = node.features.get("contentClassifier") or {}
    rows = ocr_item_rows(ocr_items(classifier))
    top_limit = node.bbox.y + max(38.0, min(node.bbox.h * 0.16, 54.0))
    for row in rows[:4]:
        row_top = min(float((item.get("bbox") or {}).get("y") or 0) for item in row)
        if row_top > top_limit:
            continue
        filtered = title_like_ocr_items(row)
        if not filtered:
            continue
        text = " ".join(str(item.get("text") or "") for item in filtered)
        if looks_like_table_header_text(text):
            continue
        if len(filtered) <= 2:
            return filtered
    return []


def looks_like_table_header_text(text: str) -> bool:
    tokens = [str(item.get("text") or "") for item in token_items(text)]
    if len(tokens) < 2:
        return False
    header_hits = sum(1 for token in tokens if re.search(r"排名|排行|序号|名称|服务|次数|数值|状态|时间|单位|云池", token))
    return header_hits >= 2


def build_repeated_metric_components(
    node: Node,
    record: ComponentRecord,
    library: ComponentLibrary,
    classifier: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if node.type != "MetricCard" or record.category != "Mores":
        return []
    option_blueprint = library.option_blueprint(record.key)
    schema_shape = library.option_shape(record.key)
    if str(schema_shape.get("datasetKind") or dataset_kind(option_blueprint)) != "number":
        return []

    clusters = repeated_metric_clusters(node, ocr_items(classifier))
    if len(clusters) < 2:
        return []

    components: List[Dict[str, Any]] = []
    for index, cluster in enumerate(clusters):
        cluster_classifier = dict(classifier)
        cluster_text = " ".join(str(item.get("text") or "") for item in cluster["items"]).strip()
        cluster_classifier["text"] = cluster_text
        cluster_classifier["paddleOcrText"] = cluster_text
        cluster_classifier["paddleOcrItems"] = cluster["items"]
        dataset = chart_dataset(
            [
                {
                    "name": cluster.get("label") or f"指标{index + 1}",
                    "value": cluster.get("value", 0),
                    "raw": cluster.get("raw", ""),
                }
            ]
        )
        bbox = cluster["bbox"]
        option_patch = build_option_patch(
            record.category,
            record.key,
            dataset,
            cluster_classifier,
            option_blueprint=option_blueprint,
            schema_shape=schema_shape,
            bbox=bbox,
        )
        facts = build_recognition_facts(record.key, record.category, dataset, cluster_classifier, bbox, schema_shape)
        hydrated_option = hydrate_option_from_facts(option_blueprint, option_patch, facts, schema_shape)
        components.append(
            {
                "id": f"schema_{node.node_id}_metric_{index + 1}",
                "nodeId": f"{node.node_id}_metric_{index + 1}",
                "sourceNodeId": node.node_id,
                "virtual": True,
                "componentId": record.key,
                "title": record.title,
                "category": record.category,
                "categoryName": record.category_name,
                "schemaShape": schema_shape,
                "bbox": bbox,
                "attr": {
                    "x": round(float(bbox["x"]), 2),
                    "y": round(float(bbox["y"]), 2),
                    "w": round(float(bbox["w"]), 2),
                    "h": round(float(bbox["h"]), 2),
                    "offsetX": 0,
                    "offsetY": 0,
                    "zIndex": z_index_for_node(node.type),
                },
                "chartConfig": {
                    "key": record.key,
                    "chartKey": record.chart_key,
                    "conKey": record.con_key,
                    "title": record.title,
                    "category": record.category,
                    "categoryName": record.category_name,
                    "chartFrame": record.chart_frame,
                    "package": infer_package(record.key, record.category),
                },
                "option": hydrated_option,
                "optionPatch": option_patch,
                "recognitionFacts": facts,
                "optionHydrationVersion": "schema-aware-v1",
                "dataSource": {
                    "source": "ocr+layout+metric-split",
                    "ocrText": cluster_text,
                    "modelText": cluster_text,
                    "contentType": "metric_card",
                },
            }
        )
    return components


def repeated_metric_clusters(node: Node, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    percent_items = [
        item
        for item in items
        if NUMBER_RE.fullmatch(str(item.get("text") or "").strip()) and str(item.get("text") or "").strip().endswith("%")
    ]
    if len(percent_items) < 2 or node.bbox.w / max(1.0, node.bbox.h) < 1.7:
        return []
    percent_items.sort(key=lambda item: bbox_center(item.get("bbox") or {})[0])
    centers = [bbox_center(item.get("bbox") or {})[0] for item in percent_items]
    if min((centers[i + 1] - centers[i] for i in range(len(centers) - 1)), default=0.0) < node.bbox.w * 0.12:
        return []

    boundaries = [node.bbox.x]
    boundaries.extend((centers[i] + centers[i + 1]) / 2.0 for i in range(len(centers) - 1))
    boundaries.append(node.bbox.right)

    clusters: List[Dict[str, Any]] = []
    for index, percent_item in enumerate(percent_items):
        left = boundaries[index]
        right = boundaries[index + 1]
        cluster_items = [
            item
            for item in items
            if left <= bbox_center(item.get("bbox") or {})[0] <= right
        ]
        text_tokens = [str(item.get("text") or "").strip() for item in cluster_items if str(item.get("text") or "").strip()]
        labels = [token for token in text_tokens if not NUMBER_RE.fullmatch(token)]
        raw = str(percent_item.get("text") or "")
        inset = max(2.0, (right - left) * 0.04)
        bbox = {
            "x": round(left + inset, 2),
            "y": round(node.bbox.y, 2),
            "w": round(max(24.0, right - left - inset * 2.0), 2),
            "h": round(node.bbox.h, 2),
        }
        clusters.append(
            {
                "bbox": bbox,
                "items": sorted(cluster_items, key=lambda item: (float((item.get("bbox") or {}).get("y") or 0), float((item.get("bbox") or {}).get("x") or 0))),
                "label": labels[-1] if labels else "",
                "value": parse_bar_number(raw),
                "raw": raw,
            }
        )
    return clusters


def bbox_center(bbox: Dict[str, Any]) -> tuple[float, float]:
    return (
        float(bbox.get("x") or 0) + float(bbox.get("w") or 0) / 2.0,
        float(bbox.get("y") or 0) + float(bbox.get("h") or 0) / 2.0,
    )


def build_virtual_inner_component(node: Node, nodes: List[Node], library: ComponentLibrary) -> Optional[Dict[str, Any]]:
    if node.type not in {"Panel", "Border"}:
        return None

    classifier = node.features.get("contentClassifier") or {}
    record = infer_virtual_content_record(classifier, library)
    if not record:
        return None
    if has_inner_content_node(node, nodes, record.category):
        return None

    bbox = virtual_content_bbox(node, record.category, classifier)
    dataset = infer_dataset(node, classifier, record.category)
    option_blueprint = library.option_blueprint(record.key)
    schema_shape = library.option_shape(record.key)
    option_patch = build_option_patch(
        record.category,
        record.key,
        dataset,
        classifier,
        option_blueprint=option_blueprint,
        schema_shape=schema_shape,
        bbox=bbox,
    )
    facts = build_recognition_facts(record.key, record.category, dataset, classifier, bbox, schema_shape)
    hydrated_option = hydrate_option_from_facts(option_blueprint, option_patch, facts, schema_shape)
    return {
        "id": f"schema_{node.node_id}_inner",
        "nodeId": f"{node.node_id}_inner",
        "sourceNodeId": node.node_id,
        "virtual": True,
        "componentId": record.key,
        "title": record.title,
        "category": record.category,
        "categoryName": record.category_name,
        "schemaShape": schema_shape,
        "bbox": bbox,
        "attr": {
            "x": round(float(bbox["x"]), 2),
            "y": round(float(bbox["y"]), 2),
            "w": round(float(bbox["w"]), 2),
            "h": round(float(bbox["h"]), 2),
            "offsetX": 0,
            "offsetY": 0,
            "zIndex": 10,
        },
        "chartConfig": {
            "key": record.key,
            "chartKey": record.chart_key,
            "conKey": record.con_key,
            "title": record.title,
            "category": record.category,
            "categoryName": record.category_name,
            "chartFrame": record.chart_frame,
            "package": infer_package(record.key, record.category),
        },
        "option": hydrated_option,
        "optionPatch": option_patch,
        "recognitionFacts": facts,
        "optionHydrationVersion": "schema-aware-v1",
        "dataSource": {
            "source": "ocr+vlm+container-repair",
            "ocrText": str(classifier.get("paddleOcrText") or ""),
            "modelText": str(classifier.get("text") or ""),
            "contentType": virtual_content_type(record.category, record.key),
            "llmComponentId": str(classifier.get("llmComponentId") or ""),
            "llmVisualForm": str(classifier.get("llmVisualForm") or ""),
        },
    }


def has_inner_content_node(node: Node, nodes: List[Node], expected_category: str = "") -> bool:
    outer = node.bbox.to_dict()
    for child in nodes:
        if child.node_id == node.node_id or child.type not in INNER_CONTENT_TYPES:
            continue
        if expected_category == "Tables" and child.type != "Table":
            continue
        inner = child.bbox.to_dict()
        if bbox_containment(outer, inner) >= 0.82 and bbox_area(outer) / max(1.0, bbox_area(inner)) >= 1.12:
            return True
        if bbox_iou(outer, inner) >= 0.72:
            return True
    return False


def infer_virtual_content_record(classifier: Dict[str, Any], library: ComponentLibrary):
    text = " ".join(
        str(classifier.get(key) or "")
        for key in ["text", "paddleOcrText", "textEvidence", "visualEvidence"]
    )
    visual_form = str(classifier.get("llmVisualForm") or "").lower()
    llm_component_id = str(classifier.get("llmComponentId") or "")
    content_type = str(classifier.get("contentType") or "")
    items = ocr_items(classifier)

    preferred = ""
    if (content_type == "table" or "table" in visual_form) and (ocr_items_form_table(items) or looks_like_simple_scroll_table(text)):
        preferred = "TableScrollBoard" if "TableScrollBoard" in library.by_key else "TablesBasic"
    elif "学历" in text or any(token in text for token in ["高中以下", "本科", "硕士"]):
        preferred = "liquidBar"
    elif re.search(r"\d{2}-\d{2}", text):
        preferred = "VChartLine" if "VChartLine" in library.by_key else "LineCommon"
    elif any(token in text for token in ["服务分布", "平台分布"]) and "%" in text:
        preferred = "PieCircle" if "PieCircle" in library.by_key else "PieCommon"
    elif "cylinder" in visual_form and "CylinderBar" in library.by_key:
        preferred = "CylinderBar"
    elif "liquid" in visual_form and "liquidBar" in library.by_key:
        preferred = "liquidBar"
    elif llm_component_id in library.by_key:
        record = library.by_key[llm_component_id]
        if record.category in CONTENT_CATEGORIES:
            preferred = llm_component_id

    return library.by_key.get(preferred) if preferred else None


def virtual_content_bbox(node: Node, category: str, classifier: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    x = node.bbox.x + max(12.0, node.bbox.w * 0.04)
    y = node.bbox.y + max(44.0, node.bbox.h * 0.15)
    w = node.bbox.w - max(24.0, node.bbox.w * 0.08)
    h = node.bbox.h - max(66.0, node.bbox.h * 0.24)
    if category == "Tables":
        table_bbox = table_bbox_from_ocr_items(node, ocr_items(classifier or {}))
        if table_bbox:
            return table_bbox
        y = node.bbox.y + max(36.0, node.bbox.h * 0.08)
        h = node.bbox.h - max(44.0, node.bbox.h * 0.14)
    return {"x": round(x, 2), "y": round(y, 2), "w": round(max(20.0, w), 2), "h": round(max(20.0, h), 2)}


def virtual_content_type(category: str, component_id: str) -> str:
    if category == "Bars":
        return "bar_chart"
    if category == "Lines":
        return "line_chart"
    if category == "Pies":
        return "pie_chart"
    if category == "Tables":
        return "table"
    if category == "Maps":
        return "map"
    return component_id


def suppress_overlapping_components(components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    suppressed: set[str] = set()
    for item in components:
        if is_chart_text_artifact(item, components):
            suppressed.add(str(item.get("id") or ""))

    for outer in components:
        outer_id = str(outer.get("id") or "")
        if outer_id in suppressed:
            continue
        outer_bbox = outer.get("bbox") or {}
        outer_category = str(outer.get("category") or "")
        outer_content_type = str(((outer.get("dataSource") or {}).get("contentType")) or "")
        outer_contributors = {
            str(item)
            for item in outer.get("contributorNodeIds", [])
        } if isinstance(outer.get("contributorNodeIds"), list) else set()

        if outer_category == "Borders" and is_container_only_border(outer, components):
            suppressed.add(outer_id)
            continue

        for inner in components:
            inner_id = str(inner.get("id") or "")
            if not inner_id or inner_id == outer_id or inner_id in suppressed:
                continue
            inner_bbox = inner.get("bbox") or {}
            inner_category = str(inner.get("category") or "")
            inner_node_id = str(inner.get("sourceNodeId") or inner.get("nodeId") or "")
            if outer_contributors and inner_node_id in outer_contributors:
                suppressed.add(inner_id)
                continue
            contain = bbox_containment(outer_bbox, inner_bbox)
            iou = bbox_iou(outer_bbox, inner_bbox)
            area_ratio = bbox_area(outer_bbox) / max(1.0, bbox_area(inner_bbox))

            if iou >= 0.86 and same_component_family(outer_category, inner_category):
                loser = weaker_duplicate(outer, inner)
                suppressed.add(str(loser.get("id") or ""))
                continue

            if contain < 0.82 or area_ratio < 1.22:
                continue

            if outer_category == "Tables" and inner_category == "Tables":
                loser = weaker_table_duplicate(outer, inner)
                suppressed.add(str(loser.get("id") or ""))
                if str(loser.get("id") or "") == outer_id:
                    break
                continue

            if outer_category == "Tables" and inner_category != "Tables":
                suppressed.add(inner_id)
                continue

            outer_is_visual_container = outer_category in CONTENT_CATEGORIES and outer_content_type in {
                "chart",
                "bar_chart",
                "line_chart",
                "pie_chart",
                "scatter_chart",
                "table",
            }
            inner_is_real_content = inner_category in CONTENT_CATEGORIES
            if outer_is_visual_container and inner_is_real_content:
                suppressed.add(outer_id)
                break

    return [item for item in components if str(item.get("id") or "") not in suppressed]


def is_container_only_border(border: Dict[str, Any], components: List[Dict[str, Any]]) -> bool:
    border_id = str(border.get("id") or "")
    border_bbox = border.get("bbox") or {}
    text = " ".join(
        str((border.get("dataSource") or {}).get(key) or "")
        for key in ["ocrText", "modelText"]
    ).strip()
    if len(text) <= 8:
        return False

    has_title = False
    has_content = False
    for item in components:
        if str(item.get("id") or "") == border_id:
            continue
        bbox = item.get("bbox") or {}
        category = str(item.get("category") or "")
        contain = bbox_containment(border_bbox, bbox)
        overlap = bbox_iou(border_bbox, bbox)
        if category == "Title" and (contain >= 0.7 or overlap > 0.04):
            has_title = True
        if category in CONTENT_CATEGORIES and (contain >= 0.72 or overlap > 0.12):
            has_content = True
        if has_title and has_content:
            return True
    return False


def is_chart_text_artifact(component: Dict[str, Any], components: List[Dict[str, Any]]) -> bool:
    category = str(component.get("category") or "")
    if not is_string_component(component) and category not in {"Title", "Texts", "Decorates", "Inputs"}:
        return False
    text = string_component_text(component)
    if not looks_like_axis_or_legend_text(text):
        return False

    bbox = component.get("bbox") or {}
    for chart in components:
        if chart is component:
            continue
        if str(chart.get("category") or "") not in {"Bars", "Lines", "Pies", "Scatters", "Areas", "Funnels", "Mores"}:
            continue
        chart_bbox = chart.get("bbox") or {}
        if bbox_containment(chart_bbox, bbox) >= 0.52:
            return True
        if is_tight_chart_bottom_text(chart_bbox, bbox):
            return True
    return False


def is_string_component(component: Dict[str, Any]) -> bool:
    shape = component.get("schemaShape") if isinstance(component.get("schemaShape"), dict) else {}
    if shape.get("datasetKind") == "string":
        return True
    dataset = (component.get("optionPatch") or {}).get("dataset") if isinstance(component.get("optionPatch"), dict) else None
    return isinstance(dataset, str)


def string_component_text(component: Dict[str, Any]) -> str:
    patch = component.get("optionPatch") if isinstance(component.get("optionPatch"), dict) else {}
    dataset = patch.get("dataset")
    if isinstance(dataset, str):
        return dataset
    source = component.get("dataSource") if isinstance(component.get("dataSource"), dict) else {}
    return str(source.get("ocrText") or source.get("modelText") or "")


def looks_like_axis_or_legend_text(text: str) -> bool:
    tokens = token_items(text)
    if len(tokens) >= 4:
        return True
    numbers = [item for item in tokens if item["type"] in {"number", "date"}]
    labels = [item for item in tokens if item["type"] == "label"]
    if len(numbers) >= 2 and len(labels) >= 1:
        return True
    return len(str(text or "")) >= 18 and len(labels) >= 2


def is_tight_chart_bottom_text(chart_bbox: Dict[str, Any], text_bbox: Dict[str, Any]) -> bool:
    chart_left = float(chart_bbox.get("x") or 0)
    chart_right = chart_left + float(chart_bbox.get("w") or 0)
    chart_bottom = float(chart_bbox.get("y") or 0) + float(chart_bbox.get("h") or 0)
    text_left = float(text_bbox.get("x") or 0)
    text_right = text_left + float(text_bbox.get("w") or 0)
    text_top = float(text_bbox.get("y") or 0)
    text_height = float(text_bbox.get("h") or 0)
    overlap_w = max(0.0, min(chart_right, text_right) - max(chart_left, text_left))
    text_w = max(1.0, float(text_bbox.get("w") or 0))
    return overlap_w / text_w >= 0.58 and -8.0 <= text_top - chart_bottom <= max(24.0, text_height * 0.9)


def bbox_area(bbox: Dict[str, Any]) -> float:
    return max(0.0, float(bbox.get("w") or 0)) * max(0.0, float(bbox.get("h") or 0))


def bbox_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    intersection = bbox_intersection(a, b)
    union = bbox_area(a) + bbox_area(b) - intersection
    return intersection / max(1.0, union)


def bbox_containment(outer: Dict[str, Any], inner: Dict[str, Any]) -> float:
    return bbox_intersection(outer, inner) / max(1.0, bbox_area(inner))


def bbox_intersection(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax1 = float(a.get("x") or 0)
    ay1 = float(a.get("y") or 0)
    ax2 = ax1 + float(a.get("w") or 0)
    ay2 = ay1 + float(a.get("h") or 0)
    bx1 = float(b.get("x") or 0)
    by1 = float(b.get("y") or 0)
    bx2 = bx1 + float(b.get("w") or 0)
    by2 = by1 + float(b.get("h") or 0)
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def same_component_family(left: str, right: str) -> bool:
    if left == right:
        return True
    chart_categories = CONTENT_CATEGORIES - {"Tables", "Maps", "Biz", "Three"}
    return left in chart_categories and right in chart_categories


def weaker_duplicate(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    left_bbox = left.get("bbox") or {}
    right_bbox = right.get("bbox") or {}
    left_area = bbox_area(left_bbox)
    right_area = bbox_area(right_bbox)
    if abs(left_area - right_area) / max(left_area, right_area, 1.0) > 0.08:
        return left if left_area > right_area else right
    left_z = int((left.get("attr") or {}).get("zIndex") or 0)
    right_z = int((right.get("attr") or {}).get("zIndex") or 0)
    return left if left_z < right_z else right


def weaker_table_duplicate(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    left_source = str((left.get("dataSource") or {}).get("source") or "")
    right_source = str((right.get("dataSource") or {}).get("source") or "")
    if "container-repair" in left_source and "container-repair" not in right_source:
        return right
    if "container-repair" in right_source and "container-repair" not in left_source:
        return left

    left_rows = table_patch_row_count(left)
    right_rows = table_patch_row_count(right)
    if left_rows != right_rows:
        return left if left_rows < right_rows else right
    return weaker_duplicate(left, right)


def table_patch_row_count(component: Dict[str, Any]) -> int:
    patch = component.get("optionPatch") if isinstance(component.get("optionPatch"), dict) else {}
    dataset = patch.get("dataset")
    if isinstance(dataset, list):
        return len(dataset)
    if isinstance(dataset, dict):
        source = dataset.get("source")
        if isinstance(source, list):
            return len(source)
    return 0


def z_index_for_node(node_type: str) -> int:
    if node_type in {"Panel", "Border", "Decorate"}:
        return 1
    if node_type in {"Chart", "Table", "Map", "MetricCard", "Filter", "Image"}:
        return 10
    if node_type == "Title":
        return 20
    return 10


def ocr_items(classifier: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = classifier.get("paddleOcrItems") if isinstance(classifier, dict) else None
    if not isinstance(raw, list):
        return []
    items: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox") if isinstance(item.get("bbox"), dict) else None
        text = str(item.get("text") or "").strip()
        if not text or not bbox:
            continue
        items.append({"text": text, "bbox": bbox, "score": item.get("score")})
    return items


def ocr_item_rows(items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    clean = [
        item
        for item in items
        if isinstance(item.get("bbox"), dict) and str(item.get("text") or "").strip()
    ]
    if not clean:
        return []
    heights = sorted(float((item.get("bbox") or {}).get("h") or 0) for item in clean)
    median_h = heights[len(heights) // 2] if heights else 12.0
    tolerance = max(8.0, median_h * 0.72)
    rows: List[List[Dict[str, Any]]] = []
    for item in sorted(clean, key=lambda entry: (bbox_center(entry.get("bbox") or {})[1], bbox_center(entry.get("bbox") or {})[0])):
        _, cy = bbox_center(item.get("bbox") or {})
        target: Optional[List[Dict[str, Any]]] = None
        for row in rows:
            row_cy = sum(bbox_center(entry.get("bbox") or {})[1] for entry in row) / max(1, len(row))
            if abs(cy - row_cy) <= tolerance:
                target = row
                break
        if target is None:
            rows.append([item])
        else:
            target.append(item)
    for row in rows:
        row.sort(key=lambda entry: bbox_center(entry.get("bbox") or {})[0])
    rows.sort(key=lambda row: min(float((item.get("bbox") or {}).get("y") or 0) for item in row))
    return rows


def ocr_items_form_table(items: List[Dict[str, Any]]) -> bool:
    rows = table_candidate_rows(items)
    if len(rows) < 3:
        return False
    column_centers = table_column_centers(rows)
    if len(column_centers) < 3:
        return False
    if not any(len(row) >= 3 for row in rows):
        return False
    populated_rows = 0
    for row in rows:
        occupied = {
            nearest_column_index(bbox_center(item.get("bbox") or {})[0], column_centers)
            for item in row
        }
        if len(occupied) >= 2:
            populated_rows += 1
    return populated_rows >= 3


def table_candidate_rows(items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    rows = ocr_item_rows(items)
    if not rows:
        return []
    # A panel title is usually a single short text row above the table header.
    if len(rows) >= 2 and len(rows[0]) <= 1 and len(rows[1]) >= 2:
        return rows[1:]
    return rows


def table_column_centers(rows: List[List[Dict[str, Any]]]) -> List[float]:
    centers: List[float] = []
    for row in rows:
        if len(row) < 2:
            continue
        centers.extend(bbox_center(item.get("bbox") or {})[0] for item in row)
    if not centers:
        return []
    centers.sort()
    clusters: List[List[float]] = []
    tolerance = max(34.0, (max(centers) - min(centers)) / 16.0)
    for center in centers:
        if not clusters or abs(center - (sum(clusters[-1]) / len(clusters[-1]))) > tolerance:
            clusters.append([center])
        else:
            clusters[-1].append(center)
    return [sum(cluster) / len(cluster) for cluster in clusters if len(cluster) >= 2 or len(clusters) <= 4]


def nearest_column_index(center: float, columns: List[float]) -> int:
    if not columns:
        return 0
    return min(range(len(columns)), key=lambda index: abs(center - columns[index]))


def table_bbox_from_ocr_items(node: Node, items: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    rows = table_candidate_rows(items)
    if len(rows) < 2:
        return None
    selected = [item for row in rows for item in row]
    left = min(float((item.get("bbox") or {}).get("x") or 0) for item in selected)
    top = min(float((item.get("bbox") or {}).get("y") or 0) for item in selected)
    right = max(float((item.get("bbox") or {}).get("x") or 0) + float((item.get("bbox") or {}).get("w") or 0) for item in selected)
    bottom = max(float((item.get("bbox") or {}).get("y") or 0) + float((item.get("bbox") or {}).get("h") or 0) for item in selected)
    pad_x = max(16.0, node.bbox.w * 0.035)
    pad_y = 8.0
    x = max(node.bbox.x, left - pad_x)
    y = max(node.bbox.y, top - pad_y)
    max_right = min(node.bbox.right, right + pad_x)
    max_bottom = min(node.bbox.bottom, bottom + pad_y)
    return {
        "x": round(x, 2),
        "y": round(y, 2),
        "w": round(max(24.0, max_right - x), 2),
        "h": round(max(24.0, max_bottom - y), 2),
    }


def table_dataset_from_ocr_items(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rows = table_candidate_rows(items)
    if len(rows) < 2:
        return None
    columns = table_column_centers(rows)
    if len(columns) < 2:
        return None

    header_index = 0
    for index, row in enumerate(rows[:3]):
        non_numeric = sum(1 for item in row if not NUMBER_RE.fullmatch(str(item.get("text") or "")))
        if len(row) >= 2 and non_numeric >= max(1, len(row) - 1):
            header_index = index
            break

    header_cells = assign_row_to_columns(rows[header_index], columns)
    headers = [cell or f"字段{index + 1}" for index, cell in enumerate(header_cells)]
    body_rows = rows[header_index + 1 :]
    if len(body_rows) < 1:
        return None

    dimensions = [
        {"key": f"col_{index + 1}", "title": header, "width": inferred_column_width(index, headers)}
        for index, header in enumerate(headers)
    ]
    source = []
    for row in body_rows[:12]:
        cells = assign_row_to_columns(row, columns)
        if headers and re.search(r"排名|排行|序号", headers[0]) and not cells[0]:
            cells[0] = str(len(source) + 1)
        if sum(1 for cell in cells if cell) < 2:
            continue
        source.append({f"col_{index + 1}": cells[index] if index < len(cells) else "" for index in range(len(headers))})
    if not source:
        return None
    return {"dimensions": dimensions, "source": source}


def assign_row_to_columns(row: List[Dict[str, Any]], columns: List[float]) -> List[str]:
    cells: List[List[str]] = [[] for _ in columns]
    for item in sorted(row, key=lambda entry: bbox_center(entry.get("bbox") or {})[0]):
        index = nearest_column_index(bbox_center(item.get("bbox") or {})[0], columns)
        cells[index].append(str(item.get("text") or "").strip())
    return [" ".join(cell).strip() for cell in cells]


def inferred_column_width(index: int, headers: List[str]) -> float:
    title = headers[index] if index < len(headers) else ""
    if re.search(r"名称|服务|任务|标题|内容", title):
        return 1.45
    if re.search(r"时间|日期", title):
        return 1.35
    if re.search(r"次数|数值|数量|排名|状态|云", title):
        return 0.9
    return 1.0


def infer_dataset(node: Node, classifier: Dict[str, Any], category: str) -> Dict[str, Any]:
    extracted_line_data = classifier.get("extractedLineData")
    if (category == "Lines" or str(classifier.get("contentType") or "") == "line_chart") and isinstance(extracted_line_data, dict):
        source = extracted_line_data.get("source")
        if isinstance(source, list) and len(source) >= 2:
            return {
                "dimensions": extracted_line_data.get("dimensions") or ["name", "value"],
                "source": source,
            }

    model_text = str(classifier.get("text") or "").strip()
    ocr_text = str(classifier.get("paddleOcrText") or "").strip()
    content_type = str(classifier.get("contentType") or "")
    model_pairs = explicit_model_pairs(model_text, category) if model_text else []
    if model_text and category in {"Bars", "Funnels", "Mores"} and "%" not in model_text:
        ordered_pairs = explicit_ordered_number_pairs(model_text, ocr_text)
        if len(ordered_pairs) >= 2:
            model_pairs = ordered_pairs
    if len(model_pairs) >= 2:
        if category == "Tables":
            return table_dataset_from_text(model_text) or table_dataset(model_pairs)
        if category == "Mores" and content_type == "radar_chart":
            return radar_dataset(model_pairs)
        return chart_dataset(model_pairs)

    text = " ".join(
        str(value or "")
        for value in [
            classifier.get("text"),
            classifier.get("paddleOcrText"),
            classifier.get("textEvidence"),
        ]
    ).strip()
    items = classifier.get("paddleOcrItems") if isinstance(classifier.get("paddleOcrItems"), list) else []
    if category == "Tables":
        ocr_table = table_dataset_from_ocr_items(ocr_items(classifier))
        if ocr_table:
            return ocr_table
    item_texts = [str(item.get("text") or "") for item in items if isinstance(item, dict)]
    source_text = " ".join(item_texts) or text
    pairs = typed_pairs(source_text, category, content_type)
    if not pairs:
        pairs = extract_pairs(source_text)
    if not pairs:
        pairs = fallback_pairs(text, node)

    if category == "Tables":
        return table_dataset_from_text(text) or table_dataset(pairs)
    if category == "Mores" and str(classifier.get("contentType") or "") == "radar_chart":
        return radar_dataset(pairs)
    return chart_dataset(pairs)


def typed_pairs(text: str, category: str, content_type: str) -> List[Dict[str, Any]]:
    if category == "Pies" or content_type == "pie_chart":
        return percent_pairs(text)
    if category == "Lines" or content_type == "line_chart":
        return line_pairs(text)
    if category == "Bars" or content_type == "bar_chart":
        return bar_pairs(text)
    return []


def explicit_model_pairs(text: str, category: str) -> List[Dict[str, Any]]:
    if category not in {"Bars", "Pies", "Funnels", "Mores"}:
        return []
    if "%" in str(text or ""):
        rows: List[Dict[str, Any]] = []
        for segment in re.split(r"[,，;；、\n]+", text or ""):
            rows.extend(explicit_percent_pairs_in_segment(segment, category))
        if len(rows) < 2:
            rows = explicit_percent_pairs_in_segment(text, category)
        if len(rows) >= 2:
            return dedupe_pairs(rows)[:24]
    return bar_pairs(text) if category in {"Bars", "Funnels", "Mores"} else []


def explicit_ordered_number_pairs(model_text: str, label_text: str) -> List[Dict[str, Any]]:
    model_tokens = token_items(model_text)
    values: List[Dict[str, Any]] = []
    for item in model_tokens:
        if item["type"] != "number":
            break
        raw = str(item.get("text") or "")
        if raw.endswith("%"):
            return []
        values.append({"value": parse_bar_number(raw), "raw": raw})
    if len(values) < 2:
        return []
    label_tokens = token_items(label_text)
    labels = [
        str(item.get("text") or "")
        for item in label_tokens
        if item["type"] == "label" and not looks_like_unit_or_title(str(item.get("text") or ""))
    ]
    if len(labels) < len(values):
        labels = [
            str(item.get("text") or "")
            for item in model_tokens
            if item["type"] == "label" and not looks_like_unit_or_title(str(item.get("text") or ""))
        ]
    if len(labels) < len(values):
        return []
    used_labels: Dict[str, int] = {}
    return [
        {
            "name": unique_label(labels[index], used_labels),
            "value": values[index]["value"],
            "raw": values[index]["raw"],
        }
        for index in range(len(values))
    ][:24]


def explicit_percent_pairs_in_segment(text: str, category: str) -> List[Dict[str, Any]]:
    value_first = re.compile(r"(?:^|\s)([-+]?\d+(?:\.\d+)?%)\s+([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z_-]{0,18})(?=(?:\s+\d+(?:\.\d+)?[\u4e00-\u9fffA-Za-z]+)?(?:\s+[-+]?\d+(?:\.\d+)?%|$))")
    label_first = re.compile(r"(?:^|\s)([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z_-]{0,18})\s+([-+]?\d+(?:\.\d+)?%)(?=\s|$)")
    used_labels: Dict[str, int] = {}
    rows = [
        {"name": unique_label(label, used_labels), "value": parse_model_percent(value, category), "raw": value}
        for value, label in value_first.findall(text or "")
        if not looks_like_unit_or_title(label) and not looks_like_noise(label)
    ]
    if rows:
        return rows
    return [
        {"name": unique_label(label, used_labels), "value": parse_model_percent(value, category), "raw": value}
        for label, value in label_first.findall(text or "")
        if not looks_like_unit_or_title(label) and not looks_like_noise(label)
    ]


def parse_model_percent(value: str, category: str) -> float:
    if category == "Pies":
        return parse_number(value)
    return parse_bar_number(value)


def dedupe_pairs(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for item in pairs:
        key = (str(item.get("name") or ""), float(item.get("value") or 0))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def percent_pairs(text: str) -> List[Dict[str, Any]]:
    tokens = token_items(text)
    percent_items = [item for item in tokens if item["type"] == "number" and str(item["text"]).endswith("%")]
    if not percent_items:
        return []
    labels = [item for item in tokens if item["type"] == "label" and not looks_like_unit_or_title(str(item["text"]))]
    used_labels: Dict[str, int] = {}
    pairs: List[Dict[str, Any]] = []
    for item in percent_items:
        previous = [label for label in labels if label["index"] < item["index"]]
        following = [label for label in labels if label["index"] > item["index"]]
        label = ""
        if previous and abs(item["index"] - previous[-1]["index"]) <= 2:
            label = str(previous[-1]["text"])
        elif following:
            label = str(following[0]["text"])
        elif previous:
            label = str(previous[-1]["text"])
        pairs.append({"name": unique_label(label or f"占比{len(pairs) + 1}", used_labels), "value": parse_number(str(item["text"])), "raw": item["text"]})
    return pairs[:12]


def line_pairs(text: str) -> List[Dict[str, Any]]:
    dates = DATE_LABEL_RE.findall(text or "")
    if len(dates) < 2:
        return []
    first_date_pos = (text or "").find(dates[0])
    before_dates = (text or "")[:first_date_pos]
    values = [parse_number(item) for item in NUMBER_RE.findall(before_dates) if not item.endswith("%")]
    if not values:
        return []
    used_labels: Dict[str, int] = {}
    return [
        {"name": unique_label(dates[index], used_labels), "value": values[index], "raw": str(values[index])}
        for index in range(min(len(dates), len(values), 24))
    ]


def bar_pairs(text: str) -> List[Dict[str, Any]]:
    tokens = token_items(text)
    numbers = [item for item in tokens if item["type"] == "number"]
    labels = [item for item in tokens if item["type"] == "label" and not looks_like_unit_or_title(str(item["text"]))]
    if len(numbers) < 2 or len(labels) < 2:
        return []
    used_labels: Dict[str, int] = {}

    first_number = numbers[0]["index"]
    tail_labels = [item for item in labels if item["index"] > first_number]
    if len(tail_labels) >= len(numbers):
        return [
            {
                "name": unique_label(str(tail_labels[index]["text"]), used_labels),
                "value": parse_bar_number(str(number["text"])),
                "raw": number["text"],
            }
            for index, number in enumerate(numbers[: len(tail_labels)])
        ][:24]
    return []


def parse_bar_number(value: str) -> float:
    text = str(value or "").strip()
    if text.endswith("%"):
        return round(float(text.rstrip("%") or 0), 4)
    return parse_number(text)


def token_items(text: str) -> List[Dict[str, Any]]:
    raw_tokens = re.findall(r"\d{1,2}-\d{1,2}|[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9_-]{0,18}|[-+]?\d+(?:\.\d+)?%?", text or "")
    items = []
    for index, token in enumerate(raw_tokens):
        if DATE_LABEL_RE.fullmatch(token):
            items.append({"index": index, "type": "date", "text": token})
        elif NUMBER_RE.fullmatch(token):
            items.append({"index": index, "type": "number", "text": token})
        elif not looks_like_noise(token):
            items.append({"index": index, "type": "label", "text": token})
    return items


def looks_like_unit_or_title(token: str) -> bool:
    text = str(token or "").strip()
    return text in {"万人", "人", "条", "个", "次", "AI", "平台分布", "服务分布", "学历分布", "调用分布指数", "实时调用"} or text.endswith("分布")


def build_recognition_facts(
    component_id: str,
    category: str,
    dataset: Dict[str, Any],
    classifier: Dict[str, Any],
    bbox: Dict[str, Any],
    schema_shape: Dict[str, Any],
) -> Dict[str, Any]:
    rows = normalized_rows(dataset_rows(dataset))
    text = classifier_text(classifier)
    title = first_title_text(classifier)
    values = [float(row.get("value") or 0) for row in rows]
    labels = [str(row.get("name") or f"指标{index + 1}") for index, row in enumerate(rows)]
    return {
        "componentId": component_id,
        "category": category,
        "datasetKind": str(schema_shape.get("datasetKind") or ""),
        "bbox": {
            "x": round(float(bbox.get("x") or 0), 2),
            "y": round(float(bbox.get("y") or 0), 2),
            "w": round(float(bbox.get("w") or 0), 2),
            "h": round(float(bbox.get("h") or 0), 2),
        },
        "text": text[:500],
        "title": title,
        "labels": labels[:24],
        "values": values[:24],
        "unit": infer_unit(text),
        "colors": extract_colors(classifier),
        "series": dataset_rows(dataset)[:24],
        "optionKeys": schema_shape.get("optionKeys") or [],
        "semanticPaths": schema_shape.get("semanticPaths") or {},
    }


def hydrate_option_from_facts(
    option_blueprint: Dict[str, Any],
    option_patch: Dict[str, Any],
    facts: Dict[str, Any],
    schema_shape: Dict[str, Any],
) -> Dict[str, Any]:
    option = deepcopy(option_blueprint or {})
    option = merge_option_patch(option, option_patch or {})
    apply_semantic_facts_to_option(option, facts, schema_shape or {})
    sync_series_data_from_facts(option, facts)
    return option


def merge_option_patch(base: Any, patch: Any) -> Any:
    if not isinstance(base, dict):
        return deepcopy(patch)
    if not isinstance(patch, dict):
        return deepcopy(patch) if patch not in (None, "") else deepcopy(base)
    out = deepcopy(base)
    for key, value in patch.items():
        if key == "dataset":
            out[key] = deepcopy(value)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_option_patch(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def apply_semantic_facts_to_option(option: Dict[str, Any], facts: Dict[str, Any], schema_shape: Dict[str, Any]) -> None:
    semantic_paths = schema_shape.get("semanticPaths") if isinstance(schema_shape.get("semanticPaths"), dict) else {}
    title = str(facts.get("title") or "").strip()
    unit = str(facts.get("unit") or "").strip()
    colors = [str(item) for item in facts.get("colors") or [] if is_hex_color(str(item))]
    bbox = facts.get("bbox") if isinstance(facts.get("bbox"), dict) else {}

    if title:
        for path in semantic_paths.get("title") or []:
            set_option_path_if_compatible(option, path, title, overwrite_empty=True)
    if unit:
        for path in semantic_paths.get("unit") or []:
            set_option_path_if_compatible(option, path, unit, overwrite_empty=True)
    if colors:
        for index, path in enumerate(semantic_paths.get("color") or []):
            value: Any = colors if accepts_color_list_path(path) else color_for_option_path(path, colors, index)
            set_option_path_if_compatible(option, path, value, overwrite_empty=True)
        apply_palette_entrypoints(option, colors)

    text = option.get("dataset") if isinstance(option.get("dataset"), str) else title
    if text:
        for path in semantic_paths.get("fontSize") or []:
            current = get_option_path(option, path)
            size = adaptive_text_size(str(text), numeric_value(bbox.get("w"), None), numeric_value(bbox.get("h"), 0.0) or 0.0, numeric_value(current, 20.0) or 20.0, height_ratio=0.56)
            set_option_path_if_compatible(option, path, size, overwrite_empty=False)


def sync_series_data_from_facts(option: Dict[str, Any], facts: Dict[str, Any]) -> None:
    rows = fact_rows(facts)
    if not rows:
        return
    sync_category_axes_from_facts(option, rows)
    series = option.get("series")
    if isinstance(series, dict):
        sync_single_series_data(series, rows)
    elif isinstance(series, list):
        for item in series:
            if isinstance(item, dict):
                sync_single_series_data(item, rows)


def apply_palette_entrypoints(option: Dict[str, Any], colors: List[str]) -> None:
    if not colors or not looks_like_echarts_option(option):
        return
    palette = colors[:8]
    current_color = option.get("color")
    if current_color is None or (isinstance(current_color, list) and all(isinstance(item, str) for item in current_color)):
        option["color"] = palette
    current_color_list = option.get("colorList")
    if current_color_list is None or (isinstance(current_color_list, list) and all(isinstance(item, str) for item in current_color_list)):
        option["colorList"] = palette


def looks_like_echarts_option(option: Dict[str, Any]) -> bool:
    return any(key in option for key in ("series", "xAxis", "yAxis", "legend", "tooltip", "grid"))


def sync_category_axes_from_facts(option: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    labels = [str(row.get("name") or f"指标{index + 1}") for index, row in enumerate(rows)]
    for axis_key in ("xAxis", "yAxis"):
        axis = option.get(axis_key)
        if isinstance(axis, dict):
            sync_single_category_axis(axis, labels)
        elif isinstance(axis, list):
            for item in axis:
                if isinstance(item, dict):
                    sync_single_category_axis(item, labels)


def sync_single_category_axis(axis: Dict[str, Any], labels: List[str]) -> None:
    if axis.get("type") == "category" and isinstance(axis.get("data"), list):
        axis["data"] = labels


def fact_rows(facts: Dict[str, Any]) -> List[Dict[str, Any]]:
    labels = [str(item) for item in facts.get("labels") or []]
    values = [numeric_value(item, None) for item in facts.get("values") or []]
    rows: List[Dict[str, Any]] = []
    for index, value in enumerate(values):
        if value is None:
            continue
        rows.append({"name": labels[index] if index < len(labels) else f"指标{index + 1}", "value": float(value)})
    return rows[:24]


def sync_single_series_data(series: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    data = series.get("data")
    if not isinstance(data, list) or not data:
        return
    if isinstance(data[0], dict):
        max_value = max(float(row.get("value") or 0) for row in rows)
        series["data"] = [series_data_object(data, row, index, max_value) for index, row in enumerate(rows)]
        return
    if isinstance(data[0], (int, float)) and should_replace_numeric_series(series, data, rows):
        series["data"] = [float(row.get("value") or 0) for row in rows]
        return
    if isinstance(data[0], list) and should_replace_tuple_series(series):
        series["data"] = [[index, float(row.get("value") or 0), str(row.get("name") or f"指标{index + 1}")] for index, row in enumerate(rows)]


def series_data_object(default_data: List[Any], row: Dict[str, Any], index: int, max_value: float) -> Dict[str, Any]:
    sample = default_data[min(index, len(default_data) - 1)]
    out = deepcopy(sample) if isinstance(sample, dict) else {}
    out["name"] = str(row.get("name") or out.get("name") or f"指标{index + 1}")
    value = float(row.get("value") or 0)
    if isinstance(out.get("value"), list):
        out["value"] = series_tuple_value(out.get("value"), row, value, max_value)
    else:
        out["value"] = value
    return out


def series_tuple_value(sample_value: Any, row: Dict[str, Any], value: float, max_value: float) -> List[Any]:
    sample = list(sample_value) if isinstance(sample_value, list) else []
    if not sample:
        return [str(row.get("name") or ""), value]
    out = deepcopy(sample)
    name = str(row.get("name") or "")
    if out and isinstance(out[0], str):
        out[0] = name
    elif out and isinstance(out[0], (int, float)):
        out[0] = name
    numeric_positions = [index for index, item in enumerate(out) if isinstance(item, (int, float))]
    if numeric_positions:
        out[numeric_positions[-1]] = value
        for position in numeric_positions[:-1]:
            out[position] = max_value if max_value > value else value
    return out



def should_replace_numeric_series(series: Dict[str, Any], data: List[Any], rows: List[Dict[str, Any]]) -> bool:
    label = series.get("label") if isinstance(series.get("label"), dict) else {}
    tooltip = series.get("tooltip") if isinstance(series.get("tooltip"), dict) else {}
    if label.get("show") is True:
        return True
    if series.get("silent") is True and tooltip.get("show") is False:
        return False
    return len(data) >= 2 and len(rows) >= 2


def should_replace_tuple_series(series: Dict[str, Any]) -> bool:
    encode = series.get("encode")
    return isinstance(encode, dict) or str(series.get("type") or "") == "custom"


def set_option_path_if_compatible(option: Dict[str, Any], path: str, value: Any, overwrite_empty: bool) -> bool:
    if path.startswith("dataset."):
        return False
    parts = [part for part in path.split(".") if part]
    if not parts:
        return False
    return set_option_path_targets([option], parts, value, overwrite_empty)


def set_option_path_targets(targets: List[Any], parts: List[str], value: Any, overwrite_empty: bool) -> bool:
    if not parts:
        return False
    raw_part = parts[0]
    is_array = raw_part.endswith("[]")
    part = raw_part[:-2] if is_array else raw_part
    changed = False
    if len(parts) == 1:
        for index, target in enumerate(targets):
            if not isinstance(target, dict) or part not in target:
                continue
            changed = assign_option_value(target, part, indexed_path_value(value, index), overwrite_empty) or changed
        return changed

    next_targets: List[Any] = []
    for target in targets:
        if not isinstance(target, dict) or part not in target:
            continue
        child = target.get(part)
        if is_array and isinstance(child, list):
            next_targets.extend(item for item in child if isinstance(item, dict))
        elif not is_array:
            next_targets.append(child)
    return set_option_path_targets(next_targets, parts[1:], value, overwrite_empty)


def assign_option_value(target: Dict[str, Any], key: str, value: Any, overwrite_empty: bool) -> bool:
    current = target.get(key)
    if isinstance(value, list):
        string_values = [str(item) for item in value if isinstance(item, str) and is_hex_color(item)]
        if isinstance(current, list) and all(isinstance(item, str) for item in current) and string_values:
            target[key] = string_values[: max(len(current), 1)]
            return True
        if isinstance(current, str) and string_values and (overwrite_empty or not current):
            target[key] = string_values[0]
            return True
    if isinstance(value, str):
        if isinstance(current, str) and (overwrite_empty or not current):
            target[key] = value
            return True
        if isinstance(current, list) and all(isinstance(item, str) for item in current):
            target[key] = [value, *current[1:]] if current else [value]
            return True
    if isinstance(value, (int, float)) and isinstance(current, (int, float)):
        target[key] = value
        return True
    return False


def indexed_path_value(value: Any, index: int) -> Any:
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return value[index % len(value)]
    return value


def accepts_color_list_path(path: str) -> bool:
    leaf = str(path or "").lower().rsplit(".", 1)[-1].replace("[]", "")
    return leaf in {"colors", "colorlist"}


def get_option_path(option: Dict[str, Any], path: str) -> Any:
    if "[]" in path:
        return None
    target: Any = option
    for part in path.split("."):
        if not isinstance(target, dict):
            return None
        target = target.get(part)
    return target


def infer_unit(text: str) -> str:
    matches = re.findall(r"\d+(?:\.\d+)?\s*([\u4e00-\u9fffA-Za-z%]{1,6})", text or "")
    for unit in matches:
        unit = str(unit).strip()
        if unit and not NUMBER_RE.fullmatch(unit) and unit not in {"AI"}:
            return unit
    return ""


def extract_colors(classifier: Dict[str, Any]) -> List[str]:
    colors: List[str] = []
    for key in ["dominantColor", "color", "fontColor", "textColor"]:
        value = classifier.get(key)
        if isinstance(value, str) and is_hex_color(value):
            colors.append(value)
    for key in ["colors", "palette", "dominantColors"]:
        value = classifier.get(key)
        if isinstance(value, list):
            colors.extend(str(item) for item in value if is_hex_color(str(item)))
    signature = classifier.get("visualSignature")
    if isinstance(signature, dict):
        value = signature.get("palette")
        if isinstance(value, list):
            colors.extend(str(item) for item in value if is_hex_color(str(item)))
    text = " ".join(str(classifier.get(key) or "") for key in ["text", "visualEvidence", "rawModelOutput"])
    colors.extend(re.findall(r"#[0-9a-fA-F]{6}\\b", text))
    seen = set()
    out: List[str] = []
    for color in colors:
        normalized = color.lower()
        if normalized not in seen:
            seen.add(normalized)
            out.append(color)
    return out[:12]


def is_hex_color(value: str) -> bool:
    return bool(re.fullmatch(r"#[0-9a-fA-F]{6}", str(value or "").strip()))


def color_for_option_path(path: str, colors: List[str], index: int) -> str:
    if not colors:
        return "#ffffff"
    normalized = str(path or "").lower()
    leaf = normalized.rsplit(".", 1)[-1].replace("[]", "")
    visible_colors = [color for color in colors if color_luminance(color) >= 0.22] or colors
    bright_colors = [color for color in colors if color_luminance(color) >= 0.48] or visible_colors
    dark_colors = [color for color in colors if color_luminance(color) <= 0.32] or colors

    if any(token in normalized for token in ["font", "text", "label", "legend", "axislabel", "title"]):
        return bright_colors[index % len(bright_colors)]
    if any(token in leaf for token in ["background", "bgc"]) or leaf in {"basecolor", "shadowcolor"}:
        return dark_colors[index % len(dark_colors)]
    return visible_colors[index % len(visible_colors)]


def color_luminance(color: str) -> float:
    text = str(color or "").strip().lstrip("#")
    if len(text) != 6:
        return 0.0
    try:
        r = int(text[0:2], 16) / 255.0
        g = int(text[2:4], 16) / 255.0
        b = int(text[4:6], 16) / 255.0
    except ValueError:
        return 0.0
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def build_option_patch(
    category: str,
    component_id: str,
    dataset: Dict[str, Any],
    classifier: Dict[str, Any],
    option_blueprint: Optional[Dict[str, Any]] = None,
    schema_shape: Optional[Dict[str, Any]] = None,
    bbox: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    option_blueprint = option_blueprint or {}
    if category in {"Borders", "FlowChart", "Three"}:
        return {}
    if category == "Decorates" and dataset_kind(option_blueprint) == "none":
        return {}
    if option_blueprint and "dataset" not in option_blueprint:
        return {}
    if component_id == "AIShield":
        source_dataset = dataset
        if not (isinstance(source_dataset, dict) and isinstance(source_dataset.get("nodes"), list)):
            source_dataset = ai_shield_dataset_from_text(classifier_text(classifier))
        return ai_shield_option_patch(source_dataset)
    if component_id == "AIRobot":
        return ai_robot_option_patch(dataset, classifier, option_blueprint)
    if component_id == "TableScrollBoard":
        return table_scroll_board_option_patch(dataset)

    adapted_dataset = adapt_dataset_to_component_option(
        component_id,
        category,
        dataset,
        classifier,
        option_blueprint,
        schema_shape or {},
    )
    patch: Dict[str, Any] = {"dataset": adapted_dataset}
    title_text = first_title_text(classifier)
    if title_text and (
        component_id in {"title1", "TextCommon", "TextGradient", "TextBarrage", "InputsInput"}
        or dataset_kind(option_blueprint or {}) == "string"
    ):
        patch["dataset"] = title_text
    patch = apply_string_component_style_patch(patch, option_blueprint or {}, schema_shape or {}, bbox or {})
    if category == "Mores" and "radarIndicator" in dataset:
        patch["radar"] = {"indicator": dataset["radarIndicator"]}
    if component_id == "LineGradientSingle":
        patch["preserveRuntimeStyle"] = True
        patch["series"] = [
            {
                "lineStyle": {
                    "width": 3,
                    "color": "#ff6a2a",
                    "shadowColor": "rgba(255, 73, 38, 0.5)",
                    "shadowBlur": 10,
                    "shadowOffsetY": 12,
                },
                "itemStyle": {"color": "#ff6a2a"},
                "areaStyle": {
                    "opacity": 0.8,
                    "color": {
                        "type": "linear",
                        "x": 0,
                        "y": 0,
                        "x2": 0,
                        "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": "rgba(255, 83, 43, 0.42)"},
                            {"offset": 1, "color": "rgba(255, 83, 43, 0.02)"},
                        ],
                    },
                },
            }
        ]
    return patch


def apply_string_component_style_patch(
    patch: Dict[str, Any],
    option_blueprint: Dict[str, Any],
    schema_shape: Dict[str, Any],
    bbox: Dict[str, Any],
) -> Dict[str, Any]:
    if str(schema_shape.get("datasetKind") or dataset_kind(option_blueprint)) != "string":
        return patch
    text = str(patch.get("dataset") or "")
    if not text:
        return patch
    height = numeric_value(bbox.get("h"), None)
    width = numeric_value(bbox.get("w"), None)
    if not height or height <= 0:
        return patch

    next_patch = dict(patch)
    if "fontSize" in option_blueprint:
        default_size = numeric_value(option_blueprint.get("fontSize"), 20.0) or 20.0
        next_patch["fontSize"] = adaptive_text_size(text, width, height, default_size, height_ratio=0.56)
    if "textSize" in option_blueprint:
        default_size = numeric_value(option_blueprint.get("textSize"), 20.0) or 20.0
        next_patch["textSize"] = adaptive_text_size(text, width, height, default_size, height_ratio=0.46)
    if "letterSpacing" in option_blueprint:
        next_patch["letterSpacing"] = 0
    return next_patch


def adaptive_text_size(
    text: str,
    width: Optional[float],
    height: float,
    default_size: float,
    height_ratio: float,
) -> int:
    size_by_height = max(10.0, height * height_ratio)
    size_by_width = default_size
    if width and width > 0:
        units = visual_text_units(text)
        if units > 0:
            size_by_width = max(10.0, width / units * 0.88)
    return int(round(max(10.0, min(default_size, size_by_height, size_by_width))))


def visual_text_units(text: str) -> float:
    units = 0.0
    for char in str(text or ""):
        if char.isspace():
            units += 0.35
        elif re.match(r"[\u4e00-\u9fff]", char):
            units += 1.0
        elif char.isupper() or char.isdigit():
            units += 0.62
        else:
            units += 0.54
    return max(1.0, units)


def adapt_dataset_to_component_option(
    component_id: str,
    category: str,
    dataset: Dict[str, Any],
    classifier: Dict[str, Any],
    option_blueprint: Dict[str, Any],
    schema_shape: Dict[str, Any],
) -> Any:
    if "dataset" not in option_blueprint:
        return dataset
    default_dataset = option_blueprint.get("dataset")
    kind = str(schema_shape.get("datasetKind") or "")

    if isinstance(default_dataset, str) or kind == "string":
        return first_title_text(classifier) or classifier_text(classifier)[:80]
    if isinstance(default_dataset, (int, float)) or kind == "number":
        return first_dataset_number(dataset, classifier, float(default_dataset or 0))
    if isinstance(default_dataset, list):
        rows = normalized_rows(dataset_rows(dataset))
        if not rows:
            return deepcopy(default_dataset)
        return adapt_array_dataset(default_dataset, rows)
    if not isinstance(default_dataset, dict):
        return dataset

    if component_id == "Radar" or "radarIndicator" in default_dataset:
        rows = normalized_rows(dataset_rows(dataset))
        return radar_dataset(rows) if rows else deepcopy(default_dataset)
    if "source" in default_dataset and isinstance(default_dataset.get("source"), list):
        return adapt_source_dataset(default_dataset, dataset, category)
    if "values" in default_dataset and isinstance(default_dataset.get("values"), list):
        return adapt_values_dataset(default_dataset, dataset, category)
    if component_id == "KeySecurity3D":
        return key_security_dataset_patch(default_dataset, dataset, classifier)
    if "regions" in default_dataset:
        return map_region_dataset_patch(default_dataset, dataset)
    if "markers" in default_dataset:
        return map_marker_dataset_patch(default_dataset, dataset)
    if "point" in default_dataset:
        return map_point_dataset_patch(default_dataset, dataset)
    if {"xAxis", "yAxis", "seriesData"}.issubset(default_dataset.keys()):
        return heatmap_dataset_patch(default_dataset, dataset)
    if "nodes" in default_dataset:
        return node_object_dataset_patch(default_dataset, dataset)
    return generic_object_dataset_patch(default_dataset, dataset)


def dataset_kind(option_blueprint: Dict[str, Any]) -> str:
    dataset = option_blueprint.get("dataset") if isinstance(option_blueprint, dict) else None
    if isinstance(dataset, str):
        return "string"
    if isinstance(dataset, (int, float)):
        return "number"
    if isinstance(dataset, list):
        return "array"
    if isinstance(dataset, dict):
        if isinstance(dataset.get("source"), list):
            return "object.source"
        if isinstance(dataset.get("values"), list):
            return "object.values"
        return "object"
    return "none"


def classifier_text(classifier: Dict[str, Any]) -> str:
    return " ".join(
        str(classifier.get(key) or "").strip()
        for key in ["text", "paddleOcrText", "textEvidence", "visualEvidence"]
        if str(classifier.get(key) or "").strip()
    )


def classifier_visible_text(classifier: Dict[str, Any]) -> str:
    return " ".join(
        str(classifier.get(key) or "").strip()
        for key in ["text", "paddleOcrText"]
        if str(classifier.get(key) or "").strip()
    )


def dataset_rows(dataset: Any) -> List[Any]:
    if isinstance(dataset, dict):
        for key in ["source", "values", "data", "seriesData"]:
            value = dataset.get(key)
            if isinstance(value, list):
                return value
    if isinstance(dataset, list):
        return dataset
    return []


def normalized_rows(rows: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        if isinstance(row, dict):
            name = row_label(row, index)
            value = row_numeric_value(row)
            next_row = dict(row)
            next_row["name"] = name
            next_row["value"] = value
            out.append(next_row)
        elif isinstance(row, (list, tuple)):
            name = str(row[0]) if row else f"指标{index + 1}"
            value = next((parse_number(str(item)) for item in row if NUMBER_RE.fullmatch(str(item))), 0.0)
            out.append({"name": name, "value": value})
        else:
            out.append({"name": f"指标{index + 1}", "value": numeric_value(row, 0.0)})
    return out


def row_label(row: Dict[str, Any], index: int) -> str:
    for key in ["name", "product", "label", "title", "type", "year", "productName", "message", "region"]:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return f"指标{index + 1}"


def row_numeric_value(row: Dict[str, Any]) -> float:
    for key in ["value", "data1", "data", "count", "num", "total", "totalSum", "totalAmount", "percent"]:
        value = row.get(key)
        parsed = numeric_value(value, None)
        if parsed is not None:
            return parsed
    for value in row.values():
        parsed = numeric_value(value, None)
        if parsed is not None:
            return parsed
    return 0.0


def numeric_value(value: Any, fallback: Optional[float]) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return fallback
    if NUMBER_RE.fullmatch(text):
        return parse_bar_number(text)
    match = NUMBER_RE.search(text)
    return parse_bar_number(match.group(0)) if match else fallback


def first_dataset_number(dataset: Dict[str, Any], classifier: Dict[str, Any], fallback: float = 0.0) -> float:
    rows = normalized_rows(dataset_rows(dataset))
    if rows:
        return float(rows[0].get("value") or fallback)
    text = classifier_text(classifier)
    match = NUMBER_RE.search(text)
    return parse_bar_number(match.group(0)) if match else fallback


def adapt_source_dataset(default_dataset: Dict[str, Any], dataset: Dict[str, Any], category: str) -> Dict[str, Any]:
    rows = normalized_rows(dataset_rows(dataset))
    if not rows:
        return deepcopy(default_dataset)
    dimensions = default_dataset.get("dimensions") if isinstance(default_dataset.get("dimensions"), list) else []
    keys = dimension_keys(dimensions)
    if not keys:
        keys = ["product", "value"] if category in {"Bars", "Pies", "Lines", "Areas"} else ["name", "value"]
    source = [source_row_for_keys(keys, row, index) for index, row in enumerate(rows[:24])]
    return {
        **deepcopy(default_dataset),
        "dimensions": deepcopy(dimensions) if dimensions else keys,
        "source": source,
    }


def source_row_for_keys(keys: List[str], row: Dict[str, Any], index: int) -> Dict[str, Any]:
    value = float(row.get("value") or 0)
    out: Dict[str, Any] = {}
    for key_index, key in enumerate(keys):
        normalized = key.lower()
        if key_index == 0 or re.search(r"name|label|title|product|type|year|time|message", normalized):
            out[key] = row.get(key) or row.get("name") or f"指标{index + 1}"
        elif key_index == 1 or re.search(r"value|data|count|num|total|amount|sum|percent", normalized):
            out[key] = row.get(key, value)
        elif "status" in normalized or "level" in normalized:
            out[key] = row.get(key) or row.get("status") or "正常"
        else:
            scaled = max(0.0, round(value * (0.72 if key_index == 2 else 1.0), 4))
            out[key] = row.get(key, scaled if re.search(r"\d|data|value", normalized) else "")
    return out


def adapt_values_dataset(default_dataset: Dict[str, Any], dataset: Dict[str, Any], category: str) -> Dict[str, Any]:
    rows = normalized_rows(dataset_rows(dataset))
    if not rows:
        return deepcopy(default_dataset)
    sample = next((item for item in default_dataset.get("values") or [] if isinstance(item, dict)), {})
    keys = list(sample.keys()) or ["name", "value"]
    values = [value_row_for_keys(keys, row, index, category) for index, row in enumerate(rows[:24])]
    return {**deepcopy(default_dataset), "values": values}


def value_row_for_keys(keys: List[str], row: Dict[str, Any], index: int, category: str) -> Dict[str, Any]:
    value = float(row.get("value") or 0)
    out: Dict[str, Any] = {}
    for key in keys:
        normalized = key.lower()
        if key in {"country", "year", "cylinders"}:
            out[key] = row.get(key) or ("系列1" if category in {"Lines", "Areas", "Bars"} else row.get("name") or f"系列{index + 1}")
        elif re.search(r"value|horsepower|count|num|total", normalized):
            out[key] = row.get(key, value)
        elif re.search(r"name|type|label|title|product", normalized):
            out[key] = row.get(key) or row.get("name") or f"指标{index + 1}"
        elif key in {"x", "y"}:
            out[key] = row.get(key, index + 1 if key == "x" else value)
        else:
            out[key] = row.get(key, "")
    return out


def adapt_array_dataset(default_dataset: List[Any], rows: List[Dict[str, Any]]) -> List[Any]:
    sample = next((item for item in default_dataset if isinstance(item, dict)), None)
    if not sample:
        return [[row.get("name", ""), row.get("value", "")] for row in rows[:12]]
    keys = list(sample.keys()) or ["name", "value"]
    return [array_object_row(keys, sample, row, index) for index, row in enumerate(rows[:24])]


def array_object_row(keys: List[str], sample: Dict[str, Any], row: Dict[str, Any], index: int) -> Dict[str, Any]:
    value = float(row.get("value") or 0)
    out: Dict[str, Any] = {}
    for key in keys:
        normalized = key.lower()
        if re.search(r"name|label|title|product|type|year", normalized):
            out[key] = row.get(key) or row.get("name") or f"指标{index + 1}"
        elif re.search(r"value|count|num|total|data|percent", normalized):
            out[key] = row.get(key, value)
        else:
            out[key] = deepcopy(row.get(key, sample.get(key)))
    return out


def dimension_keys(dimensions: List[Any]) -> List[str]:
    keys: List[str] = []
    for item in dimensions:
        if isinstance(item, dict):
            key = item.get("key") or item.get("name") or item.get("title")
        else:
            key = item
        if key:
            keys.append(str(key))
    return keys


def map_region_dataset_patch(default_dataset: Dict[str, Any], dataset: Dict[str, Any]) -> Dict[str, Any]:
    rows = normalized_rows(dataset_rows(dataset))
    out = deepcopy(default_dataset)
    regions = out.get("regions") if isinstance(out.get("regions"), list) else []
    for index, row in enumerate(rows[: len(regions)]):
        if isinstance(regions[index], dict):
            regions[index]["name"] = row.get("name") or regions[index].get("name")
            regions[index]["value"] = row.get("value", regions[index].get("value"))
    return out


def map_marker_dataset_patch(default_dataset: Dict[str, Any], dataset: Dict[str, Any]) -> Dict[str, Any]:
    rows = normalized_rows(dataset_rows(dataset))
    out = deepcopy(default_dataset)
    markers = out.get("markers") if isinstance(out.get("markers"), list) else []
    for index, row in enumerate(rows[: len(markers)]):
        if isinstance(markers[index], dict):
            markers[index]["name"] = row.get("name") or markers[index].get("name")
            markers[index]["value"] = row.get("value", markers[index].get("value"))
    return out


def map_point_dataset_patch(default_dataset: Dict[str, Any], dataset: Dict[str, Any]) -> Dict[str, Any]:
    rows = normalized_rows(dataset_rows(dataset))
    out = deepcopy(default_dataset)
    points = out.get("point") if isinstance(out.get("point"), list) else []
    for index, row in enumerate(rows[: len(points)]):
        if not isinstance(points[index], dict):
            continue
        points[index]["name"] = row.get("name") or points[index].get("name")
        value = points[index].get("value")
        if isinstance(value, list) and len(value) >= 3:
            value[2] = row.get("value", value[2])
    return out


def heatmap_dataset_patch(default_dataset: Dict[str, Any], dataset: Dict[str, Any]) -> Dict[str, Any]:
    rows = normalized_rows(dataset_rows(dataset))
    out = deepcopy(default_dataset)
    if not rows:
        return out
    x_axis = out.get("xAxis") if isinstance(out.get("xAxis"), list) else []
    y_axis = out.get("yAxis") if isinstance(out.get("yAxis"), list) else []
    if not x_axis or not y_axis:
        return out
    series = []
    for index, row in enumerate(rows[: min(len(x_axis) * len(y_axis), 48)]):
        series.append([index % len(x_axis), index // len(x_axis), row.get("value", 0)])
    out["seriesData"] = series
    return out


def key_security_dataset_patch(
    default_dataset: Dict[str, Any],
    dataset: Dict[str, Any],
    classifier: Dict[str, Any],
) -> Dict[str, Any]:
    rows = normalized_rows(dataset_rows(dataset))
    out = deepcopy(default_dataset)
    text_numbers = [parse_bar_number(match.group(0)) for match in NUMBER_RE.finditer(classifier_visible_text(classifier))]
    numbers = text_numbers[:]
    if not numbers:
        numbers = []
    for key, value in zip(["total", "symmetric", "asymmetric"], numbers[:3]):
        out[key] = value
    services = out.get("services") if isinstance(out.get("services"), list) else []
    for index, row in enumerate(rows[: len(services)]):
        if isinstance(services[index], dict):
            services[index]["name"] = row.get("name") or services[index].get("name")
            if numbers:
                services[index]["percent"] = row.get("value", services[index].get("percent"))
    return out


def node_object_dataset_patch(default_dataset: Dict[str, Any], dataset: Dict[str, Any]) -> Dict[str, Any]:
    rows = normalized_rows(dataset_rows(dataset))
    out = deepcopy(default_dataset)
    nodes = out.get("nodes") if isinstance(out.get("nodes"), list) else []
    for index, row in enumerate(rows[: len(nodes)]):
        if isinstance(nodes[index], dict):
            nodes[index]["label"] = row.get("name") or nodes[index].get("label")
            nodes[index]["value"] = row.get("value", nodes[index].get("value"))
    return out


def generic_object_dataset_patch(default_dataset: Dict[str, Any], dataset: Dict[str, Any]) -> Dict[str, Any]:
    rows = normalized_rows(dataset_rows(dataset))
    out = deepcopy(default_dataset)
    if not rows:
        return out
    for key, value in list(out.items()):
        if isinstance(value, list):
            continue
        if isinstance(value, (int, float)):
            out[key] = rows[0].get("value", value)
        elif isinstance(value, str) and not value:
            out[key] = str(rows[0].get("name") or "")
    return out


def ai_robot_option_patch(dataset: Dict[str, Any], classifier: Dict[str, Any], option_blueprint: Dict[str, Any]) -> Dict[str, Any]:
    default_dataset = option_blueprint.get("dataset") if isinstance(option_blueprint.get("dataset"), dict) else {}
    out = deepcopy(default_dataset)
    text = classifier_visible_text(classifier)
    values = [parse_bar_number(match.group(0)) for match in NUMBER_RE.finditer(text)]
    for node_key in ["productNodes", "reportNodes", "platformNodes"]:
        nodes = out.get(node_key) if isinstance(out.get(node_key), list) else []
        for index, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            if values:
                node["value"] = values[index % len(values)]
    return {
        "dataset": out,
        "visual": deepcopy(option_blueprint.get("visual") or {}),
    }


def ai_shield_dataset_from_text(text: str) -> Dict[str, Any]:
    normalized = " ".join(str(text or "").split())
    values = re.findall(r"(?<!\d)(\d{1,4})\s*条", normalized)
    values = [value for value in values if 1 <= int(value) <= 999]
    if not values and re.search(r"(?<!\d)32(?!\d)", normalized):
        values = ["32"]
    risk_count = max(1, normalized.count("风险预警"))
    node_count = max(6, min(8, risk_count))
    positions = [
        (7, 60),
        (18, 54),
        (74, 54),
        (88, 60),
        (17, 78),
        (34, 88),
        (62, 88),
        (78, 78),
    ]
    nodes = []
    for index in range(node_count):
        x, y = positions[index % len(positions)]
        value = values[index % len(values)] if values else "32"
        nodes.append(
            {
                "label": "风险预警",
                "value": value,
                "unit": "条",
                "status": "warning",
                "visible": True,
                "x": x,
                "y": y,
                "offsetX": 0,
                "offsetY": 0,
            }
        )
    return {
        "title": "",
        "subtitle": "",
        "centerLabel": "",
        "centerValue": "",
        "centerUnit": "",
        "nodes": nodes,
    }


def ai_shield_option_patch(dataset: Dict[str, Any]) -> Dict[str, Any]:
    nodes = dataset.get("nodes") if isinstance(dataset, dict) else []
    if not isinstance(nodes, list) or not nodes:
        nodes = ai_shield_dataset_from_text("").get("nodes", [])
    return {
        "dataset": {
            "title": str(dataset.get("title") or "") if isinstance(dataset, dict) else "",
            "subtitle": str(dataset.get("subtitle") or "") if isinstance(dataset, dict) else "",
            "centerLabel": str(dataset.get("centerLabel") or "") if isinstance(dataset, dict) else "",
            "centerValue": dataset.get("centerValue", "") if isinstance(dataset, dict) else "",
            "centerUnit": str(dataset.get("centerUnit") or "") if isinstance(dataset, dict) else "",
            "nodes": nodes,
        },
        "visual": {
            "backgroundImage": "ai-shield-background.png",
            "shieldImage": "ai-shield-body.png",
            "baseImage": "ai-shield-base.png",
            "nodeBaseImage": "ai-shield-node-base.png",
            "nodeLabelImage": "ai-shield-node-name.png",
            "haloImage": "ai-shield-halo.png",
            "showBackground": True,
            "showHalo": True,
            "showBase": True,
            "showShield": True,
            "showNodeBase": True,
            "shieldScale": 1.0,
            "baseScale": 1.0,
            "glowOpacity": 0.8,
            "valueFontSize": 26,
            "labelFontSize": 14,
        },
    }


def table_scroll_board_option_patch(dataset: Dict[str, Any]) -> Dict[str, Any]:
    dimensions = dataset.get("dimensions") if isinstance(dataset, dict) else []
    source = dataset.get("source") if isinstance(dataset, dict) else []
    if not isinstance(dimensions, list) or not dimensions:
        dimensions = [
            {"key": "name", "title": "名称", "width": 1.2},
            {"key": "status", "title": "状态", "width": 1.0},
            {"key": "value", "title": "数值", "width": 1.0},
        ]
    headers = [str(item.get("title") or item.get("key") or item) if isinstance(item, dict) else str(item) for item in dimensions]
    keys = [str(item.get("key") or item.get("title") or item) if isinstance(item, dict) else str(item) for item in dimensions]
    rows = []
    if isinstance(source, list):
        for row in source[:12]:
            if isinstance(row, dict):
                rows.append([str(row.get(key, "")) for key in keys])
            elif isinstance(row, (list, tuple)):
                rows.append([str(value) for value in row[: len(keys)]])
    if not rows:
        rows = [["数据项1", "正常", "0"], ["数据项2", "正常", "0"], ["数据项3", "正常", "0"]]
    align = ["left"] * len(headers)
    if align:
        align[-1] = "right"
    widths = []
    for item in dimensions:
        raw_width = item.get("width") if isinstance(item, dict) else None
        try:
            width = float(raw_width)
        except (TypeError, ValueError):
            width = 1.0
        widths.append(width)
    total = sum(widths) or 1.0
    return {
        "header": headers,
        "dataset": rows,
        "align": align,
        "columnWidth": [],
        "rowNum": min(max(len(rows), 3), 8),
        "waitTime": 999999,
        "headerHeight": 35,
        "headerBGC": "#00BAFF",
        "oddRowBGC": "#003B51",
        "evenRowBGC": "#0A2732",
        "sanitizeHtml": True,
        "tableColumnWeights": [round(width / total, 4) for width in widths],
    }


def extract_pairs(text: str) -> List[Dict[str, Any]]:
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9_-]{0,18}|[-+]?\d+(?:\.\d+)?%?", text or "")
    numbers = [{"index": index, "raw": token, "value": parse_number(token)} for index, token in enumerate(tokens) if NUMBER_RE.fullmatch(token)]
    labels = [
        {"index": index, "text": token}
        for index, token in enumerate(tokens)
        if not NUMBER_RE.fullmatch(token) and not looks_like_noise(token)
    ]
    paired_by_runs = pair_number_label_runs(numbers, labels)
    if paired_by_runs:
        return paired_by_runs

    pairs: List[Dict[str, Any]] = []
    pending_label: Optional[str] = None
    used_labels: Dict[str, int] = {}
    for token in tokens:
        if NUMBER_RE.fullmatch(token):
            label = pending_label or f"指标{len(pairs) + 1}"
            value = parse_number(token)
            pairs.append({"name": unique_label(label, used_labels), "value": value, "raw": token})
            pending_label = None
        elif not looks_like_noise(token):
            pending_label = token
    return pairs[:24]


def pair_number_label_runs(numbers: List[Dict[str, Any]], labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(numbers) < 2 or len(labels) < 2:
        return []
    used_labels: Dict[str, int] = {}

    leading_numbers = []
    for item in numbers:
        if labels and item["index"] < labels[0]["index"]:
            leading_numbers.append(item)
    if len(leading_numbers) >= 2:
        tail_labels = [item for item in labels if item["index"] > leading_numbers[0]["index"]]
        if len(tail_labels) >= len(leading_numbers):
            return [
                {
                    "name": unique_label(tail_labels[index]["text"], used_labels),
                    "value": number["value"],
                    "raw": number["raw"],
                }
                for index, number in enumerate(leading_numbers[: len(tail_labels)])
            ][:24]

    pairs: List[Dict[str, Any]] = []
    for number in numbers:
        previous_labels = [item for item in labels if item["index"] < number["index"]]
        if not previous_labels:
            continue
        label = previous_labels[-1]["text"]
        if re.fullmatch(r"(万人|人|条|个|次|%|AI)", label, flags=re.I):
            label = previous_labels[-2]["text"] if len(previous_labels) >= 2 else label
        pairs.append({"name": unique_label(label, used_labels), "value": number["value"], "raw": number["raw"]})
    return pairs[:24]


def fallback_pairs(text: str, node: Node) -> List[Dict[str, Any]]:
    labels = [item for item in LABEL_RE.findall(text or "") if not looks_like_noise(item)]
    if not labels:
        labels = [node.component_id or node.type or "组件"]
    values = [parse_number(item) for item in NUMBER_RE.findall(text or "")]
    if not values:
        values = [round(max(node.bbox.w, node.bbox.h), 2)]
    pairs = []
    used_labels: Dict[str, int] = {}
    for index, value in enumerate(values[: max(1, min(len(values), 12))]):
        label = labels[index % len(labels)] if labels else f"指标{index + 1}"
        pairs.append({"name": unique_label(label, used_labels), "value": value, "raw": str(value)})
    return pairs


def chart_dataset(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "dimensions": ["name", "value"],
        "source": [{"name": item["name"], "value": item["value"]} for item in pairs],
    }


def table_dataset(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "dimensions": [
            {"key": "name", "title": "名称"},
            {"key": "value", "title": "数值"},
        ],
        "source": [{"name": item["name"], "value": item["value"]} for item in pairs],
    }


def table_dataset_from_text(text: str) -> Optional[Dict[str, Any]]:
    normalized = " ".join(str(text or "").split())

    headers = ["任务名称", "责任单位", "任务类型", "创建时间", "完成时间", "任务状态"]
    task_header_count = sum(1 for header in headers if header in normalized)
    if task_header_count < 3:
        simple = simple_scroll_table_dataset(normalized)
        if simple:
            return simple
        return None

    status_tokens = ["已完成", "未完成", "进行中", "超时未完成", "超时完成"]
    unit_tokens = ["全国中心", "中邮实业", "中邮电商", "中邮传媒", "新闻中心", "石邮院", "中邮保险"]
    type_tokens = ["关基", "告警", "资产", "重保"]
    dates = re.findall(r"\d{4}-\d{2}-\d{2}\s*\d{0,2}:?\d{0,2}:?\d{0,2}", normalized)
    statuses = [token for token in status_tokens if token in normalized]

    segments = re.split("|".join(re.escape(token) for token in unit_tokens), normalized)
    rows: List[Dict[str, Any]] = []
    for index, unit in enumerate(unit_tokens):
        if unit not in normalized:
            continue
        unit_pos = normalized.find(unit)
        before = normalized[max(0, unit_pos - 48):unit_pos].strip()
        after = normalized[unit_pos:unit_pos + 180]
        task = cleanup_table_task(before)
        task_type = next((token for token in type_tokens if re.search(rf"\b{re.escape(token)}\b|{re.escape(token)}", after)), "")
        row_dates = re.findall(r"\d{4}-\d{2}-\d{2}\s*\d{0,2}:?\d{0,2}:?\d{0,2}", after)
        status = next((token for token in status_tokens if token in after), statuses[index % len(statuses)] if statuses else "")
        rows.append(
            {
                "taskName": task or f"任务{len(rows) + 1}",
                "unit": unit,
                "type": task_type or "任务",
                "createdAt": row_dates[0] if row_dates else (dates[index * 2] if index * 2 < len(dates) else ""),
                "finishedAt": row_dates[1] if len(row_dates) > 1 else (dates[index * 2 + 1] if index * 2 + 1 < len(dates) else ""),
                "status": status or "未完成",
            }
        )
        if len(rows) >= 8:
            break

    if not rows:
        return None
    return {
        "dimensions": [
            {"key": "taskName", "title": "任务名称", "width": 2.2},
            {"key": "unit", "title": "责任单位", "width": 0.85},
            {"key": "type", "title": "任务类型", "width": 0.7},
            {"key": "createdAt", "title": "创建时间", "width": 1.35},
            {"key": "finishedAt", "title": "完成时间", "width": 1.35},
            {"key": "status", "title": "任务状态", "width": 0.85},
        ],
        "source": rows,
    }


def looks_like_simple_scroll_table(text: str) -> bool:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return False
    if sum(1 for token in ["任务名称", "责任单位", "任务类型", "创建时间", "完成时间", "任务状态"] if token in normalized) >= 3:
        return False
    header_hits = sum(1 for token in ["名称", "状态", "数值"] if token in normalized)
    return header_hits >= 2


def simple_scroll_table_dataset(text: str) -> Optional[Dict[str, Any]]:
    if not looks_like_simple_scroll_table(text):
        return None
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9_-]{0,18}|[-+]?\d+(?:\.\d+)?%?", text or "")
    headers = [token for token in ["名称", "状态", "数值"] if token in tokens or token in text]
    if len(headers) < 2:
        headers = ["名称", "状态", "数值"]

    body = [token for token in tokens if token not in {"名称", "状态", "数值"} and not looks_like_noise(token)]
    rows: List[Dict[str, Any]] = []
    for index in range(0, min(len(body), 30), 3):
        chunk = body[index:index + 3]
        if len(chunk) < 2:
            continue
        rows.append(
            {
                "name": chunk[0],
                "status": chunk[1],
                "value": chunk[2] if len(chunk) > 2 else "",
            }
        )
    if not rows:
        rows = [
            {"name": "数据项1", "status": "正常", "value": "0"},
            {"name": "数据项2", "status": "正常", "value": "0"},
            {"name": "数据项3", "status": "正常", "value": "0"},
        ]
    return {
        "dimensions": [
            {"key": "name", "title": "名称", "width": 1.2},
            {"key": "status", "title": "状态", "width": 1.0},
            {"key": "value", "title": "数值", "width": 1.0},
        ],
        "source": rows[:10],
    }


def cleanup_table_task(text: str) -> str:
    cleaned = re.sub(r"任务名称|责任单位|任务类型|创建时间|完成时间|任务状态", " ", text)
    cleaned = re.sub(r"\d{4}-\d{2}-\d{2}.*$", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned[-36:] if cleaned else ""


def radar_dataset(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    indicators = [{"name": item["name"], "max": max(100, float(item["value"]) * 1.2)} for item in pairs[:8]]
    values = [item["value"] for item in pairs[:8]]
    return {
        "radarIndicator": indicators,
        "seriesData": [{"name": "识别数据", "value": values}],
        "dimensions": ["name", "value"],
        "source": [{"name": item["name"], "value": item["value"]} for item in pairs[:8]],
    }


def parse_number(value: str) -> float:
    text = str(value or "").strip()
    is_percent = text.endswith("%")
    number = float(text.rstrip("%") or 0)
    return round(number / 100, 4) if is_percent else round(number, 4)


def unique_label(label: str, used: Dict[str, int]) -> str:
    clean = label.strip()[:18] or "指标"
    used[clean] = used.get(clean, 0) + 1
    return clean if used[clean] == 1 else f"{clean}{used[clean]}"


def looks_like_noise(token: str) -> bool:
    text = str(token or "").strip()
    if len(text) <= 1 and not re.match(r"[\u4e00-\u9fff]", text):
        return True
    return text.lower() in {"data", "mon", "tue", "wed", "thu", "fri", "sat", "sun"}


def first_title_text(classifier: Dict[str, Any]) -> str:
    for key in ("text", "paddleOcrText"):
        text = str(classifier.get(key) or "").strip()
        if text:
            return text[:80]
    return ""


def infer_package(component_id: str, category: str) -> str:
    if component_id.startswith("VChart"):
        return "VChart"
    if category in {"Bars", "Pies", "Lines", "Scatters", "Maps", "Mores", "Areas", "Funnels"}:
        chart_more_ids = {
            "Dial",
            "Funnel",
            "Graph",
            "Heatmap",
            "Process",
            "Radar",
            "Sankey",
            "TreeMap",
            "WaterPolo",
        }
        decorate_more_ids = {
            "CirclePoint",
            "Clock",
            "CountDown",
            "EnergyValue",
            "FlipperNumber",
            "FullScreen",
            "Number",
            "PipelineH",
            "PipelineV",
            "TimeCommon",
        }
        if category == "Mores" and component_id in decorate_more_ids:
            return "Decorates"
        if category == "Mores" and component_id in chart_more_ids:
            return "Charts"
        return "Charts"
    if category == "Tables":
        return "Tables"
    if category in {"Borders", "Decorates", "FlowChart", "Three"}:
        return "Decorates"
    if category in {"Title", "Texts", "Inputs", "WordClouds"}:
        return "Informations"
    if category == "Biz":
        return "Customs"
    return "Charts"
