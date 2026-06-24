from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from .component_library import ComponentLibrary
from .schemas import Node


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
        dataset = infer_dataset(node, classifier, record.category)
        z_index = z_index_for_node(node.type)
        option_patch = build_option_patch(record.category, record.key, dataset, classifier)
        component = {
            "id": f"schema_{node.node_id}",
            "nodeId": node.node_id,
            "componentId": record.key,
            "title": record.title,
            "category": record.category,
            "categoryName": record.category_name,
            "bbox": node.bbox.to_dict(),
            "attr": {
                "x": round(node.bbox.x, 2),
                "y": round(node.bbox.y, 2),
                "w": round(node.bbox.w, 2),
                "h": round(node.bbox.h, 2),
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
            "optionPatch": option_patch,
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
}

INNER_CONTENT_TYPES = {"Chart", "Table", "Map", "MetricCard", "Filter"}


def build_global_virtual_components(
    nodes: List[Node],
    library: ComponentLibrary,
    existing_components: List[Dict[str, Any]],
    classifier_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    components: List[Dict[str, Any]] = []
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
        "componentType": "MetricCard",
        "text": text,
        "paddleOcrText": text,
        "visualEvidence": "center shield with AI text, circular base, multiple risk-warning metric nodes",
    }
    option_patch = build_option_patch(record.category, record.key, ai_shield_dataset_from_text(text), classifier)
    return [
        {
            "id": "schema_virtual_ai_shield",
            "nodeId": "virtual_ai_shield",
            "virtual": True,
            "componentId": record.key,
            "title": record.title,
            "category": record.category,
            "categoryName": record.category_name,
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
            "optionPatch": option_patch,
            "dataSource": {
                "source": "ocr+layout+global-repair",
                "ocrText": text[:500],
                "modelText": text[:500],
                "contentType": "ai_shield",
            },
        }
    ]


def collect_classifier_text(nodes: List[Node]) -> str:
    parts: List[str] = []
    for node in nodes:
        classifier = node.features.get("contentClassifier") or {}
        for key in ["text", "paddleOcrText", "textEvidence", "visualEvidence"]:
            value = str(classifier.get(key) or "").strip()
            if value:
                parts.append(value)
    return " ".join(parts)


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


def build_virtual_inner_component(node: Node, nodes: List[Node], library: ComponentLibrary) -> Optional[Dict[str, Any]]:
    if node.type not in {"Panel", "Border"}:
        return None
    if has_inner_content_node(node, nodes):
        return None

    classifier = node.features.get("contentClassifier") or {}
    record = infer_virtual_content_record(classifier, library)
    if not record:
        return None

    bbox = virtual_content_bbox(node, record.category)
    dataset = infer_dataset(node, classifier, record.category)
    option_patch = build_option_patch(record.category, record.key, dataset, classifier)
    return {
        "id": f"schema_{node.node_id}_inner",
        "nodeId": f"{node.node_id}_inner",
        "sourceNodeId": node.node_id,
        "virtual": True,
        "componentId": record.key,
        "title": record.title,
        "category": record.category,
        "categoryName": record.category_name,
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
        "optionPatch": option_patch,
        "dataSource": {
            "source": "ocr+vlm+container-repair",
            "ocrText": str(classifier.get("paddleOcrText") or ""),
            "modelText": str(classifier.get("text") or ""),
            "contentType": virtual_content_type(record.category, record.key),
            "llmComponentId": str(classifier.get("llmComponentId") or ""),
            "llmVisualForm": str(classifier.get("llmVisualForm") or ""),
        },
    }


def has_inner_content_node(node: Node, nodes: List[Node]) -> bool:
    outer = node.bbox.to_dict()
    for child in nodes:
        if child.node_id == node.node_id or child.type not in INNER_CONTENT_TYPES:
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

    preferred = ""
    if "任务名称" in text and "责任单位" in text:
        preferred = "TableScrollBoard" if "TableScrollBoard" in library.by_key else "TablesBasic"
    elif looks_like_simple_scroll_table(text):
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


def virtual_content_bbox(node: Node, category: str) -> Dict[str, float]:
    x = node.bbox.x + max(12.0, node.bbox.w * 0.04)
    y = node.bbox.y + max(44.0, node.bbox.h * 0.15)
    w = node.bbox.w - max(24.0, node.bbox.w * 0.08)
    h = node.bbox.h - max(66.0, node.bbox.h * 0.24)
    if category == "Tables":
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
    for outer in components:
        outer_id = str(outer.get("id") or "")
        if outer_id in suppressed:
            continue
        outer_bbox = outer.get("bbox") or {}
        outer_category = str(outer.get("category") or "")
        outer_content_type = str(((outer.get("dataSource") or {}).get("contentType")) or "")

        if outer_category == "Borders" and is_container_only_border(outer, components):
            suppressed.add(outer_id)
            continue

        for inner in components:
            inner_id = str(inner.get("id") or "")
            if not inner_id or inner_id == outer_id or inner_id in suppressed:
                continue
            inner_bbox = inner.get("bbox") or {}
            inner_category = str(inner.get("category") or "")
            contain = bbox_containment(outer_bbox, inner_bbox)
            iou = bbox_iou(outer_bbox, inner_bbox)
            area_ratio = bbox_area(outer_bbox) / max(1.0, bbox_area(inner_bbox))

            if iou >= 0.86 and same_component_family(outer_category, inner_category):
                loser = weaker_duplicate(outer, inner)
                suppressed.add(str(loser.get("id") or ""))
                continue

            if contain < 0.82 or area_ratio < 1.22:
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
    chart_categories = CONTENT_CATEGORIES - {"Tables", "Maps", "Biz"}
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


def z_index_for_node(node_type: str) -> int:
    if node_type in {"Panel", "Border", "Decorate"}:
        return 1
    if node_type in {"Chart", "Table", "Map", "MetricCard", "Filter"}:
        return 10
    if node_type == "Title":
        return 20
    return 10


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


def build_option_patch(
    category: str,
    component_id: str,
    dataset: Dict[str, Any],
    classifier: Dict[str, Any],
) -> Dict[str, Any]:
    if category in {"Borders", "Decorates", "FlowChart", "Three"}:
        return {}
    if component_id == "AIShield":
        return ai_shield_option_patch(dataset)
    if component_id == "TableScrollBoard":
        return table_scroll_board_option_patch(dataset)
    patch: Dict[str, Any] = {"dataset": dataset}
    title_text = first_title_text(classifier)
    if title_text and component_id in {"title1", "TextCommon", "TextGradient"}:
        patch["dataset"] = title_text
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
