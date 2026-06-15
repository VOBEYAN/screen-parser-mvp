from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
from PIL import Image

from .component_library import ComponentLibrary
from .matcher import TYPE_TO_CATEGORIES
from .schemas import BBox, ComponentRecord, Node


CONTENT_TO_TYPE = {
    "title": "Title",
    "table": "Table",
    "map": "Map",
    "metric_card": "MetricCard",
    "filter": "Filter",
    "bar_chart": "Chart",
    "line_chart": "Chart",
    "area_chart": "Chart",
    "pie_chart": "Chart",
    "scatter_chart": "Chart",
    "funnel_chart": "Chart",
    "wordcloud": "Chart",
    "chart": "Chart",
    "decorate": "Decorate",
    "panel": "Panel",
    "border": "Border",
}

CONTENT_TO_CATEGORIES = {
    "title": ["Title", "Texts"],
    "table": ["Tables"],
    "map": ["Maps", "Biz", "Three"],
    "metric_card": ["Mores", "Biz", "Texts"],
    "filter": ["Inputs"],
    "bar_chart": ["Bars"],
    "line_chart": ["Lines"],
    "area_chart": ["Areas", "Lines"],
    "pie_chart": ["Pies"],
    "scatter_chart": ["Scatters"],
    "funnel_chart": ["Funnels"],
    "wordcloud": ["WordClouds"],
    "chart": ["Bars", "Lines", "Pies", "Scatters", "Areas", "Funnels", "WordClouds", "FlowChart", "Mores"],
    "decorate": ["Decorates", "Mores"],
    "panel": ["Borders", "Decorates"],
    "border": ["Borders"],
}

CONTENT_KEYWORDS = {
    "bar_chart": ["柱", "条", "bar", "排行", "排名"],
    "line_chart": ["折线", "趋势", "line"],
    "area_chart": ["面积", "area", "趋势"],
    "pie_chart": ["饼", "环", "pie"],
    "scatter_chart": ["散点", "scatter"],
    "funnel_chart": ["漏斗", "funnel"],
    "wordcloud": ["词云", "word"],
    "table": ["表", "列表", "排行", "明细"],
    "map": ["地图", "map", "地球", "三维", "区域"],
    "metric_card": ["数字", "指标", "状态", "告警", "能量", "翻牌", "总数"],
    "filter": ["输入", "选择", "筛选", "下拉"],
    "title": ["标题", "文字", "文本"],
    "decorate": ["装饰", "线", "点缀"],
    "border": ["边框", "框"],
}


@dataclass
class MultimodalConfig:
    enabled: bool = True
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: float = 30.0
    max_nodes: int = 36
    candidate_k: int = 8

    @property
    def llm_enabled(self) -> bool:
        return bool(self.enabled and self.model and self.api_key)


class MultimodalComponentClassifier:
    """Final-stage content classifier.

    The class can call an OpenAI-compatible multimodal chat endpoint when
    configured. Without credentials it still extracts visual/text-like content
    cues locally and uses them to rerank catalog candidates.
    """

    def __init__(self, library: ComponentLibrary, config: Optional[MultimodalConfig] = None):
        self.library = library
        self.config = config or MultimodalConfig(enabled=False)

    @classmethod
    def from_env(
        cls,
        library: ComponentLibrary,
        enabled: bool = True,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_nodes: int = 36,
        candidate_k: int = 8,
    ) -> "MultimodalComponentClassifier":
        resolved_model = model or os.getenv("SCREEN_PARSER_VLM_MODEL")
        resolved_api_key = api_key or os.getenv("SCREEN_PARSER_VLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        resolved_base_url = (
            base_url
            or os.getenv("SCREEN_PARSER_VLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        timeout = float(os.getenv("SCREEN_PARSER_VLM_TIMEOUT", "30"))
        config = MultimodalConfig(
            enabled=enabled,
            model=resolved_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            timeout=timeout,
            max_nodes=max_nodes,
            candidate_k=candidate_k,
        )
        return cls(library, config)

    @property
    def mode(self) -> str:
        if not self.config.enabled:
            return "disabled"
        return "multimodal_llm" if self.config.llm_enabled else "visual_content_rules"

    def refine_nodes(self, nodes: Iterable[Node], image_path: str, top_k: int = 1) -> Dict[str, object]:
        if not self.config.enabled:
            return {"mode": self.mode, "processedNodeCount": 0, "llmEnabled": False}

        image = Image.open(image_path).convert("RGB")
        processed = 0
        llm_calls = 0
        errors = []
        for node in nodes:
            if node.type == "Screen":
                continue
            if processed >= self.config.max_nodes:
                break
            if node.bbox.w < 8 or node.bbox.h < 8:
                continue

            crop = crop_node(image, node.bbox)
            local_result = classify_crop_locally(crop, node)
            result = local_result

            if self.config.llm_enabled and should_call_llm(node, local_result):
                try:
                    llm_result = self.classify_with_llm(crop, node, top_k=max(top_k, self.config.candidate_k))
                    if llm_result:
                        result = merge_llm_and_local(llm_result, local_result)
                        llm_calls += 1
                except Exception as exc:  # pragma: no cover - depends on external service
                    errors.append({"nodeId": node.node_id, "error": str(exc)[:240]})

            self.apply_result(node, result, top_k=max(top_k, self.config.candidate_k))
            processed += 1

        return {
            "mode": self.mode,
            "processedNodeCount": processed,
            "llmEnabled": self.config.llm_enabled,
            "llmCallCount": llm_calls,
            "errors": errors[:8],
        }

    def classify_with_llm(self, crop: Image.Image, node: Node, top_k: int) -> Optional[Dict[str, object]]:
        candidates = self.candidate_records(node, top_k=top_k)
        prompt = build_prompt(node, candidates)
        image_url = encode_image_url(crop)
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict UI component classifier for large-screen dashboard designs. "
                        "Return only JSON. Classify by visible text, chart form, table/grid structure, map shape, and metric-card content."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        }
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - depends on external service
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"multimodal API failed: HTTP {exc.code} {body[:240]}") from exc

        content = response_payload["choices"][0]["message"]["content"]
        return parse_json_object(content)

    def candidate_records(self, node: Node, top_k: int) -> List[ComponentRecord]:
        candidate_ids = []
        for candidate in node.candidates or []:
            component_id = str(candidate.get("componentId") or "")
            if component_id and component_id in self.library.by_key and component_id not in candidate_ids:
                candidate_ids.append(component_id)
        if node.component_id and node.component_id in self.library.by_key and node.component_id not in candidate_ids:
            candidate_ids.insert(0, node.component_id)

        records = [self.library.by_key[component_id] for component_id in candidate_ids[:top_k]]
        if len(records) >= top_k:
            return records

        categories = TYPE_TO_CATEGORIES.get(node.type, [])
        pool = self.library.filter_by_categories(categories) if categories else self.library.records
        for record in pool:
            if record.key not in candidate_ids:
                records.append(record)
                candidate_ids.append(record.key)
            if len(records) >= top_k:
                break
        return records

    def apply_result(self, node: Node, result: Dict[str, object], top_k: int) -> None:
        content_type = normalize_content_type(str(result.get("contentType") or result.get("visualForm") or ""))
        predicted_type = normalize_component_type(str(result.get("componentType") or CONTENT_TO_TYPE.get(content_type, "")))
        confidence = float(result.get("confidence") or 0.0)
        text = str(result.get("text") or result.get("ocrText") or "").strip()

        if predicted_type and should_update_type(node.type, predicted_type, confidence):
            node.type = predicted_type
            node.level = level_for_type(predicted_type)

        ranked = self.rank_records(node, result, top_k=top_k)
        if ranked:
            node.component_id = ranked[0]["componentId"]
            node.candidates = ranked

        node.features["contentClassifier"] = {
            "mode": self.mode,
            "contentType": content_type,
            "componentType": predicted_type or node.type,
            "confidence": round(confidence, 4),
            "text": text,
            "reason": str(result.get("reason") or "")[:240],
            "llmComponentId": result.get("componentId"),
        }

    def rank_records(self, node: Node, result: Dict[str, object], top_k: int) -> List[Dict[str, object]]:
        content_type = normalize_content_type(str(result.get("contentType") or result.get("visualForm") or ""))
        llm_component_id = str(result.get("componentId") or "")
        text = str(result.get("text") or result.get("ocrText") or "")
        categories = CONTENT_TO_CATEGORIES.get(content_type) or TYPE_TO_CATEGORIES.get(node.type, [])
        records = self.library.filter_by_categories(categories) if categories else self.library.records

        scored = []
        base_by_id = {str(item.get("componentId")): item for item in node.candidates or []}
        for record in records:
            base = float(base_by_id.get(record.key, {}).get("score") or 0.32)
            score = 0.25 + 0.3 * base
            if record.key == llm_component_id:
                score += 0.36
            if record.category in (categories or []):
                score += 0.18
            score += keyword_score(record, content_type, text)
            score += aspect_score(record, node)
            scored.append(
                {
                    "componentId": record.key,
                    "title": record.title,
                    "category": record.category,
                    "schema": record.schema,
                    "score": round(min(0.99, score), 4),
                    "matchMode": self.mode,
                    "contentType": content_type,
                    "contentText": text[:80],
                }
            )

        if llm_component_id in self.library.by_key and all(item["componentId"] != llm_component_id for item in scored):
            record = self.library.by_key[llm_component_id]
            scored.append(
                {
                    "componentId": record.key,
                    "title": record.title,
                    "category": record.category,
                    "schema": record.schema,
                    "score": 0.95,
                    "matchMode": self.mode,
                    "contentType": content_type,
                    "contentText": text[:80],
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]


def crop_node(image: Image.Image, bbox: BBox) -> Image.Image:
    width, height = image.size
    pad = max(2, int(round(min(bbox.w, bbox.h) * 0.04)))
    left = max(0, int(round(bbox.x)) - pad)
    top = max(0, int(round(bbox.y)) - pad)
    right = min(width, int(round(bbox.right)) + pad)
    bottom = min(height, int(round(bbox.bottom)) + pad)
    return image.crop((left, top, right, bottom))


def classify_crop_locally(crop: Image.Image, node: Node) -> Dict[str, object]:
    rgb = np.array(crop.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    edges = cv2.Canny(gray, 45, 135)
    area = float(max(1, width * height))

    horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, width // 10), 1)))
    vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, height // 10))))
    horizontal_density = float(np.count_nonzero(horizontal)) / area
    vertical_density = float(np.count_nonzero(vertical)) / area
    edge_density = float(np.count_nonzero(edges)) / area
    aspect = width / float(max(1, height))
    text = ocr_text(crop)

    circles = circle_count(gray)
    connected = connected_component_count(edges)
    content_type = "decorate"
    confidence = 0.42

    if height < 54 and aspect > 2.1:
        content_type = "title"
        confidence = 0.68
    elif horizontal_density > 0.026 and vertical_density > 0.014:
        content_type = "table"
        confidence = 0.78
    elif likely_filter(text, aspect, height):
        content_type = "filter"
        confidence = 0.68
    elif has_big_numeric_text(text):
        content_type = "metric_card"
        confidence = 0.72
    elif circles >= 1 and 0.65 <= aspect <= 1.55:
        content_type = "pie_chart"
        confidence = 0.64
    elif vertical_density > horizontal_density * 1.35 and vertical_density > 0.018:
        content_type = "bar_chart"
        confidence = 0.66
    elif horizontal_density > 0.018 and edge_density > 0.045:
        content_type = "line_chart"
        confidence = 0.62
    elif connected >= 18 and edge_density < 0.08:
        content_type = "scatter_chart"
        confidence = 0.55
    elif node.type in {"Map"} or looks_like_map(gray, edges):
        content_type = "map"
        confidence = 0.55
    elif node.type == "Chart":
        content_type = "chart"
        confidence = 0.54
    elif node.type == "MetricCard":
        content_type = "metric_card"
        confidence = 0.54
    elif node.type in {"Panel", "Border"}:
        content_type = "panel" if node.type == "Panel" else "border"
        confidence = 0.52
    elif node.type == "Title":
        content_type = "title"
        confidence = 0.58

    return {
        "contentType": content_type,
        "componentType": CONTENT_TO_TYPE.get(content_type, node.type),
        "confidence": confidence,
        "text": text,
        "reason": (
            f"local features: aspect={aspect:.2f}, edge={edge_density:.3f}, "
            f"hline={horizontal_density:.3f}, vline={vertical_density:.3f}, circles={circles}, cc={connected}"
        ),
    }


def should_call_llm(node: Node, local_result: Dict[str, object]) -> bool:
    if node.type in {"Screen", "Panel", "Border"}:
        return False
    confidence = float(local_result.get("confidence") or 0.0)
    return confidence < 0.82


def build_prompt(node: Node, candidates: List[ComponentRecord]) -> str:
    candidate_lines = []
    for record in candidates:
        candidate_lines.append(
            {
                "componentId": record.key,
                "title": record.title,
                "category": record.category,
                "categoryName": record.category_name,
                "description": record.description[:160],
            }
        )
    return json.dumps(
        {
            "task": "Classify this cropped dashboard component and choose the best componentId from candidates when possible.",
            "detectorType": node.type,
            "bbox": node.bbox.to_dict(),
            "candidateComponentIds": candidate_lines,
            "allowedContentTypes": sorted(CONTENT_TO_TYPE.keys()),
            "returnJsonSchema": {
                "contentType": "one of allowedContentTypes, e.g. bar_chart/table/metric_card/title/map/filter",
                "componentType": "Panel|Title|Chart|Table|Map|MetricCard|Border|Decorate|Filter",
                "componentId": "best candidate componentId or empty string",
                "confidence": "0-1",
                "text": "visible OCR text if any",
                "reason": "short reason based on visible evidence",
            },
        },
        ensure_ascii=False,
    )


def encode_image_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def parse_json_object(content: str) -> Optional[Dict[str, object]]:
    try:
        payload = json.loads(content)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            return None
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else None


def merge_llm_and_local(llm_result: Dict[str, object], local_result: Dict[str, object]) -> Dict[str, object]:
    merged = dict(local_result)
    merged.update({key: value for key, value in llm_result.items() if value not in [None, ""]})
    merged["localContentType"] = local_result.get("contentType")
    merged["localReason"] = local_result.get("reason")
    return merged


def ocr_text(crop: Image.Image) -> str:
    try:
        import pytesseract  # type: ignore
    except Exception:
        return ""
    try:
        text = pytesseract.image_to_string(crop, lang=os.getenv("SCREEN_PARSER_OCR_LANG", "chi_sim+eng"))
    except Exception:
        return ""
    return " ".join(text.split())[:160]


def circle_count(gray: np.ndarray) -> int:
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.3,
        minDist=max(12, min(gray.shape[:2]) // 4),
        param1=80,
        param2=24,
        minRadius=max(6, min(gray.shape[:2]) // 10),
        maxRadius=max(8, min(gray.shape[:2]) // 2),
    )
    return 0 if circles is None else int(circles.shape[1])


def connected_component_count(edges: np.ndarray) -> int:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((edges > 0).astype(np.uint8), 8)
    valid = 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if 3 <= area <= 150:
            valid += 1
    return valid


def likely_filter(text: str, aspect: float, height: int) -> bool:
    if aspect > 2.6 and height < 86:
        return any(token in text for token in ["选择", "筛选", "输入", "搜索", "请选择"]) or not text
    return False


def has_big_numeric_text(text: str) -> bool:
    if re.search(r"\d[\d,.%万亿]*", text):
        return True
    return False


def looks_like_map(gray: np.ndarray, edges: np.ndarray) -> bool:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_irregular = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 40:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.35:
            large_irregular += 1
    return large_irregular >= 3 and float(np.std(gray)) > 18


def normalize_content_type(value: str) -> str:
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bar": "bar_chart",
        "bars": "bar_chart",
        "line": "line_chart",
        "lines": "line_chart",
        "area": "area_chart",
        "pie": "pie_chart",
        "scatter": "scatter_chart",
        "funnel": "funnel_chart",
        "metric": "metric_card",
        "number": "metric_card",
        "kpi": "metric_card",
        "text": "title",
        "input": "filter",
        "select": "filter",
        "word_cloud": "wordcloud",
    }
    key = aliases.get(key, key)
    return key if key in CONTENT_TO_TYPE else "chart" if "chart" in key else key


def normalize_component_type(value: str) -> str:
    value = value.strip()
    if value in {"Panel", "Title", "Chart", "Table", "Map", "MetricCard", "Border", "Decorate", "Filter"}:
        return value
    lowered = value.lower()
    for content_type, component_type in CONTENT_TO_TYPE.items():
        if content_type in lowered:
            return component_type
    return ""


def should_update_type(current: str, predicted: str, confidence: float) -> bool:
    if predicted == current:
        return False
    if current in {"Panel", "Border"} and confidence < 0.8:
        return False
    if predicted in {"Panel", "Border"} and current not in {"Panel", "Border"}:
        return False
    return confidence >= 0.52


def level_for_type(node_type: str) -> int:
    return {
        "Panel": 2,
        "Border": 2,
        "Title": 3,
        "Decorate": 3,
        "Filter": 4,
        "Chart": 4,
        "Table": 4,
        "Map": 4,
        "MetricCard": 4,
    }.get(node_type, 4)


def keyword_score(record: ComponentRecord, content_type: str, text: str) -> float:
    haystack = f"{record.key} {record.title} {record.category} {record.category_name} {record.description} {text}".lower()
    score = 0.0
    for token in CONTENT_KEYWORDS.get(content_type, []):
        if token.lower() in haystack:
            score += 0.055
    return min(0.24, score)


def aspect_score(record: ComponentRecord, node: Node) -> float:
    aspect = float(node.features.get("aspectRatio", node.bbox.w / max(node.bbox.h, 1.0)))
    description = record.description
    score = 0.0
    if aspect >= 2.2 and ("横向" in description or "延展" in description):
        score += 0.045
    if aspect < 1.2 and ("中心" in description or "均衡" in description or "圆" in description):
        score += 0.045
    return score
