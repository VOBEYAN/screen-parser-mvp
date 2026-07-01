from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .component_library import ComponentLibrary
from .component_profiles import load_component_profiles, normalize_visual_form, profile_match_score, visual_form_compatible
from .local_qwen_vl import LocalQwenVLComponentRecognizer
from .matcher import TYPE_TO_CATEGORIES
from .schemas import BBox, ComponentRecord, Node
from .visual_matcher import VisualReferenceLibrary, border_shell_mask, extract_image_features, form_family


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PADDLE_OCR_ENGINE = None
DATE_LABEL_RE = re.compile(r"\d{1,2}-\d{1,2}")


def resolve_project_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


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
    "image": "Image",
    "ai_shield": "Image",
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
    "image": ["Biz", "Three", "Decorates"],
    "ai_shield": ["Biz"],
    "decorate": ["Decorates", "Mores"],
    "panel": ["Borders", "Decorates"],
    "border": ["Borders"],
}

PACKAGE_DIVERSE_CATEGORIES = [
    "Bars",
    "Pies",
    "Lines",
    "Scatters",
    "Maps",
    "Areas",
    "Funnels",
    "WordClouds",
    "Tables",
    "Title",
    "Texts",
    "Inputs",
    "Mores",
    "Borders",
    "Decorates",
    "FlowChart",
    "Three",
    "Biz",
]

FOCUSED_COMPONENT_SETS = {
    "bar_chart": [
        "ColorPrismBar",
        "PrismaticBar",
        "CylinderBar",
        "clor",
        "liquidBar",
        "BarCommon",
        "BarLine",
        "CapsuleChart",
        "VChartBarCommon",
        "VChartBarStack",
        "BarCrossrange",
        "VChartBarCrossrange",
    ],
    "pie_chart": [
        "PieCircle",
        "PieCommon",
        "VChartPie",
        "Pie3DExploded",
        "Pie3DRingRegion",
        "Pie3DRingUser",
        "Pie3DMultiLayer",
        "Pie3DTwoBlue",
        "Pie3DTwoCyan",
    ],
    "line_chart": [
        "LineCommon",
        "LineGradientSingle",
        "LineGradients",
        "LineLinearSingle",
        "VChartLine",
        "VChartArea",
        "VChartPercentArea",
        "BarLine",
    ],
    "table": ["AlarmList", "TableList", "TableScrollBoard", "TablesBasic"],
    "Chart": [
        "ColorPrismBar",
        "PrismaticBar",
        "CylinderBar",
        "liquidBar",
        "PieCircle",
        "PieCommon",
        "Pie3DExploded",
        "VChartPie",
        "LineCommon",
        "LineGradientSingle",
        "VChartLine",
        "VChartArea",
    ],
    "Table": ["AlarmList", "TableList", "TableScrollBoard", "TablesBasic"],
    "Map": ["ChinaMap", "MapAmap", "MapBase"],
    "map": ["ChinaMap", "MapAmap", "MapBase"],
    "Title": ["title1", "TextCommon", "TextGradient", "TextBarrage", "Decorates06"],
    "Border": ["Border04", "Border02", "Border13", "Border05", "Border01", "Border07", "Border14"],
    "Panel": ["Border04", "Border02", "Border13", "Border05", "Border01", "Border07", "Border14"],
    "Image": ["AIShield", "AIRobot", "KeySecurity3D", "ThreeEarth01"],
    "image": ["AIShield", "AIRobot", "KeySecurity3D", "ThreeEarth01"],
    "ai_shield": ["AIShield", "KeySecurity3D", "AIRobot"],
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
    "image": ["图片", "图像", "主视觉", "盾牌", "机器人", "3d", "地球", "AI"],
    "ai_shield": ["盾牌", "AI", "风险预警", "安全"],
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
    candidate_k: int = 95
    force_llm: bool = False
    local_qwen_enabled: bool = False
    local_qwen_model_path: Optional[str] = None
    local_qwen_adapter_path: Optional[str] = None
    local_qwen_device: str = "auto"
    local_qwen_image_size: int = 224
    local_qwen_max_new_tokens: int = 96

    @property
    def llm_enabled(self) -> bool:
        return bool(self.enabled and self.model and self.api_key)

    @property
    def local_enabled(self) -> bool:
        return bool(
            self.enabled
            and self.local_qwen_enabled
            and self.local_qwen_model_path
            and self.local_qwen_adapter_path
        )


class MultimodalComponentClassifier:
    """Final-stage content classifier.

    The class can call an OpenAI-compatible multimodal chat endpoint when
    configured. Without credentials it still extracts visual/text-like content
    cues locally and uses them to rerank catalog candidates.
    """

    def __init__(
        self,
        library: ComponentLibrary,
        config: Optional[MultimodalConfig] = None,
        visual_library: Optional[VisualReferenceLibrary] = None,
        local_qwen: Optional[LocalQwenVLComponentRecognizer] = None,
    ):
        self.library = library
        self.config = config or MultimodalConfig(enabled=False)
        self.visual_library = visual_library or VisualReferenceLibrary([])
        reference_root = PROJECT_ROOT / "data" / "component-reference"
        self.component_profiles = load_component_profiles(str(reference_root), library)
        self.local_qwen = local_qwen or self._build_local_qwen()

    @classmethod
    def from_env(
        cls,
        library: ComponentLibrary,
        enabled: bool = True,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        visual_library: Optional[VisualReferenceLibrary] = None,
        max_nodes: int = 36,
        candidate_k: int = 95,
        force_llm: Optional[bool] = None,
        local_qwen_model_path: Optional[str] = None,
        local_qwen_adapter_path: Optional[str] = None,
        local_qwen_device: Optional[str] = None,
        local_qwen_enabled: Optional[bool] = None,
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
        if force_llm is None:
            force_llm = os.getenv("SCREEN_PARSER_VLM_FORCE", "").lower() in {"1", "true", "yes", "on"}
        max_nodes = int(os.getenv("SCREEN_PARSER_VLM_MAX_NODES", str(max_nodes)))
        candidate_k = int(os.getenv("SCREEN_PARSER_VLM_CANDIDATE_K", str(candidate_k)))
        default_local_model = PROJECT_ROOT / "models" / "qwen3-vl-2b-instruct-mlx-bf16-hfkeyed"
        default_local_adapter = PROJECT_ROOT / "output" / "qwen3-vl-mps-peft-component-lora-render-mixed"
        resolved_local_model = resolve_project_path(
            local_qwen_model_path
            or os.getenv("SCREEN_PARSER_LOCAL_QWEN_MODEL")
            or str(default_local_model)
        )
        resolved_local_adapter = resolve_project_path(
            local_qwen_adapter_path
            or os.getenv("SCREEN_PARSER_LOCAL_QWEN_ADAPTER")
            or str(default_local_adapter)
        )
        local_flag = os.getenv("SCREEN_PARSER_LOCAL_QWEN_ENABLE", "auto").lower()
        if local_qwen_enabled is None:
            if local_flag in {"1", "true", "yes", "on"}:
                local_qwen_enabled = True
            elif local_flag in {"0", "false", "no", "off"}:
                local_qwen_enabled = False
            else:
                local_qwen_enabled = Path(resolved_local_model).exists() and Path(resolved_local_adapter).exists()
        resolved_local_device = local_qwen_device or os.getenv("SCREEN_PARSER_LOCAL_QWEN_DEVICE") or "auto"
        local_image_size = int(os.getenv("SCREEN_PARSER_LOCAL_QWEN_IMAGE_SIZE", "224"))
        local_max_new_tokens = int(os.getenv("SCREEN_PARSER_LOCAL_QWEN_MAX_NEW_TOKENS", "96"))
        config = MultimodalConfig(
            enabled=enabled,
            model=resolved_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            timeout=timeout,
            max_nodes=max_nodes,
            candidate_k=candidate_k,
            force_llm=force_llm,
            local_qwen_enabled=bool(local_qwen_enabled),
            local_qwen_model_path=resolved_local_model,
            local_qwen_adapter_path=resolved_local_adapter,
            local_qwen_device=resolved_local_device,
            local_qwen_image_size=local_image_size,
            local_qwen_max_new_tokens=local_max_new_tokens,
        )
        return cls(library, config, visual_library=visual_library)

    @property
    def mode(self) -> str:
        if not self.config.enabled:
            return "disabled"
        if self.config.local_enabled:
            return "local_qwen3_vl_lora"
        return "multimodal_llm" if self.config.llm_enabled else "visual_content_rules"

    @property
    def local_qwen_status(self) -> Dict[str, object]:
        if not self.local_qwen:
            return {
                "configured": self.config.local_enabled,
                "loaded": False,
                "loadError": None,
                "modelPath": self.config.local_qwen_model_path,
                "adapterPath": self.config.local_qwen_adapter_path,
                "device": self.config.local_qwen_device,
                "imageSize": self.config.local_qwen_image_size,
            }
        return self.local_qwen.status()

    def _build_local_qwen(self) -> Optional[LocalQwenVLComponentRecognizer]:
        if not self.config.local_enabled:
            return None
        return LocalQwenVLComponentRecognizer(
            model_path=str(self.config.local_qwen_model_path),
            adapter_path=str(self.config.local_qwen_adapter_path),
            library=self.library,
            device=self.config.local_qwen_device,
            image_size=self.config.local_qwen_image_size,
            max_new_tokens=self.config.local_qwen_max_new_tokens,
        )

    def for_request(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        force_llm: Optional[bool] = None,
    ) -> "MultimodalComponentClassifier":
        has_override = any([model, base_url, api_key]) or force_llm is not None
        config = MultimodalConfig(
            enabled=self.config.enabled or has_override,
            model=model or self.config.model,
            base_url=base_url or self.config.base_url,
            api_key=api_key or self.config.api_key,
            timeout=self.config.timeout,
            max_nodes=self.config.max_nodes,
            candidate_k=self.config.candidate_k,
            force_llm=self.config.force_llm if force_llm is None else force_llm,
            local_qwen_enabled=self.config.local_qwen_enabled,
            local_qwen_model_path=self.config.local_qwen_model_path,
            local_qwen_adapter_path=self.config.local_qwen_adapter_path,
            local_qwen_device=self.config.local_qwen_device,
            local_qwen_image_size=self.config.local_qwen_image_size,
            local_qwen_max_new_tokens=self.config.local_qwen_max_new_tokens,
        )
        return MultimodalComponentClassifier(
            self.library,
            config=config,
            visual_library=self.visual_library,
            local_qwen=self.local_qwen,
        )

    def refine_nodes(self, nodes: Iterable[Node], image_path: str, top_k: int = 1) -> Dict[str, object]:
        if not self.config.enabled:
            return {"mode": self.mode, "processedNodeCount": 0, "llmEnabled": False}

        node_list = list(nodes)
        image = Image.open(image_path).convert("RGB")
        paddle_ocr = run_paddle_ocr(image_path)
        processed = 0
        llm_calls = 0
        local_qwen_calls = 0
        local_qwen_disabled_for_request = False
        errors = []
        for node in sorted(node_list, key=node_processing_priority):
            if node.type in {"Screen", "Region", "Content"}:
                continue
            if processed >= self.config.max_nodes:
                break
            if node.bbox.w < 8 or node.bbox.h < 8:
                continue

            crop = crop_node(image, node.bbox)
            feature_focus = "border_shell" if node.type in {"Border", "Panel"} else "content"
            crop_features = extract_image_features(cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2BGR), focus=feature_focus)
            local_result = classify_crop_locally(crop, node)
            node_ocr = ocr_for_node(paddle_ocr, node.bbox)
            if node_ocr["text"]:
                local_result["text"] = node_ocr["text"]
                local_result["ocrText"] = node_ocr["text"]
                local_result = apply_ocr_semantic_cues(local_result, str(node_ocr["text"] or ""))
            if node.type in {"Border", "Panel"}:
                local_result["contentType"] = "panel" if node.type == "Panel" else "border"
                local_result["componentType"] = node.type
            local_result = add_crop_visual_signature(local_result, crop)
            line_dataset = extract_line_dataset_from_crop(crop, node.bbox, node_ocr)
            if line_dataset:
                local_result["extractedLineData"] = line_dataset
            result = local_result

            used_local_qwen = False
            if (
                self.config.local_enabled
                and not local_qwen_disabled_for_request
                and should_call_local_qwen(node, local_result, force=self.config.force_llm)
            ):
                try:
                    qwen_result = self.classify_with_local_qwen(crop)
                    if qwen_result:
                        result = merge_llm_and_local(qwen_result, local_result)
                        local_qwen_calls += 1
                        used_local_qwen = True
                except Exception as exc:
                    local_qwen_disabled_for_request = True
                    errors.append({"nodeId": node.node_id, "source": "local_qwen3_vl_lora", "error": str(exc)[:240]})

            if (
                not used_local_qwen
                and self.config.llm_enabled
                and should_call_llm(node, local_result, force=self.config.force_llm)
            ):
                try:
                    structure_context = build_structure_context(node, node_list)
                    llm_result = self.classify_with_llm(
                        crop,
                        image,
                        node,
                        top_k=max(top_k, self.config.candidate_k),
                        paddle_ocr=node_ocr,
                        structure_context=structure_context,
                        content_type_hint=str(local_result.get("contentType") or ""),
                        crop_features=crop_features,
                    )
                    if llm_result:
                        result = merge_llm_and_local(llm_result, local_result)
                        llm_calls += 1
                except Exception as exc:  # pragma: no cover - depends on external service
                    errors.append({"nodeId": node.node_id, "error": str(exc)[:240]})

            self.apply_result(
                node,
                result,
                top_k=max(top_k, self.config.candidate_k),
                paddle_ocr=node_ocr,
                crop_features=crop_features,
            )
            processed += 1

        return {
            "mode": self.mode,
            "processedNodeCount": processed,
            "llmEnabled": self.config.llm_enabled,
            "llmCallCount": llm_calls,
            "llmModel": self.config.model if self.config.llm_enabled else None,
            "llmBaseUrl": self.config.base_url if self.config.llm_enabled else None,
            "localQwenEnabled": self.config.local_enabled,
            "localQwenConfigured": bool(self.config.local_qwen_model_path and self.config.local_qwen_adapter_path),
            "localQwenCallCount": local_qwen_calls,
            "localQwenStatus": self.local_qwen_status,
            "forceLlm": self.config.force_llm,
            "paddleOcrEnabled": paddle_ocr["enabled"],
            "paddleOcrTextCount": len(paddle_ocr["items"]),
            "paddleOcrFullText": paddle_ocr["fullText"][:500],
            "paddleOcrError": paddle_ocr.get("error"),
            "errors": errors[:8],
        }

    def classify_with_llm(
        self,
        crop: Image.Image,
        full_image: Image.Image,
        node: Node,
        top_k: int,
        paddle_ocr: Dict[str, object],
        structure_context: Dict[str, object],
        content_type_hint: str = "",
        crop_features: Optional[Dict[str, object]] = None,
    ) -> Optional[Dict[str, object]]:
        candidates = self.candidate_records(
            node,
            top_k=min(max(top_k, self.config.candidate_k), len(self.library.records)),
            content_type_hint=content_type_hint,
        )
        if crop_features and self.visual_library.enabled:
            candidates.sort(
                key=lambda record: self.visual_library.score(record.key, crop_features) or 0.0,
                reverse=True,
            )
        prompt = build_prompt(
            node,
            candidates,
            paddle_ocr=paddle_ocr,
            structure_context=structure_context,
            component_profiles=self.component_profiles,
            crop_features=crop_features,
        )
        image_url = encode_image_url(crop)
        candidate_sheet = build_candidate_sheet(candidates, self.visual_library)
        context_image = build_context_image(full_image, node)
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "image_url", "image_url": {"url": encode_image_url(context_image)}},
        ]
        if candidate_sheet:
            user_content.append({"type": "image_url", "image_url": {"url": encode_image_url(candidate_sheet)}})
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "max_tokens": 3000,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict UI component matcher for large-screen dashboard designs. "
                        "Return only JSON. Image1 is the detected component crop. "
                        "Image2 is the full dashboard with the target component marked by a red box. "
                        "When Image3 is provided, it is a numbered candidate sheet from the component library. "
                        "Choose the single best componentId from the supplied candidates whenever possible. "
                        "Be detail-obsessed: compare small visual differences, reject near-misses explicitly, "
                        "and trust the crop/context/library previews over broad detector labels."
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
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
        parsed = parse_json_object(content)
        if not parsed:
            return None
        resolved = resolve_llm_candidate_from_records(parsed, candidates)
        attach_candidate_scores(resolved, candidates)
        return resolved

    def classify_with_local_qwen(self, crop: Image.Image) -> Optional[Dict[str, object]]:
        if not self.local_qwen:
            return None
        return self.local_qwen.classify(crop)

    def candidate_records(self, node: Node, top_k: int, content_type_hint: str = "") -> List[ComponentRecord]:
        top_k = min(len(self.library.records), max(top_k, 8))
        candidate_ids = []
        focused = FOCUSED_COMPONENT_SETS.get(content_type_hint) or FOCUSED_COMPONENT_SETS.get(node.type, [])
        for component_id in focused:
            if component_id in self.library.by_key and component_id not in candidate_ids:
                candidate_ids.append(component_id)
        for candidate in node.candidates or []:
            component_id = str(candidate.get("componentId") or "")
            if component_id and component_id in self.library.by_key and component_id not in candidate_ids:
                candidate_ids.append(component_id)
        if node.component_id and node.component_id in self.library.by_key and node.component_id not in candidate_ids:
            candidate_ids.insert(0, node.component_id)

        records = [self.library.by_key[component_id] for component_id in candidate_ids[:top_k]]
        if len(records) >= top_k:
            return records

        categories = CONTENT_TO_CATEGORIES.get(content_type_hint) or TYPE_TO_CATEGORIES.get(node.type, [])
        pool = self.library.filter_by_categories(categories) if categories else self.library.records
        for record in pool:
            if record.key not in candidate_ids:
                records.append(record)
                candidate_ids.append(record.key)
            if len(records) >= top_k:
                break

        if len(records) >= top_k:
            return records

        # Keep candidates diverse across the whole ai-schema-view package tree.
        # Detector types are imperfect: a title, table, KPI card, image or decorate
        # can be misdetected as Chart, so the VLM must still see non-chart options.
        for category in PACKAGE_DIVERSE_CATEGORIES:
            for record in self.library.filter_by_categories([category]):
                if record.key not in candidate_ids:
                    records.append(record)
                    candidate_ids.append(record.key)
                    break
            if len(records) >= top_k:
                break

        if len(records) >= top_k:
            return records

        for record in self.library.records:
            if record.key not in candidate_ids:
                records.append(record)
                candidate_ids.append(record.key)
            if len(records) >= top_k:
                break
        return records

    def apply_result(
        self,
        node: Node,
        result: Dict[str, object],
        top_k: int,
        paddle_ocr: Optional[Dict[str, object]] = None,
        crop_features: Optional[Dict[str, object]] = None,
    ) -> None:
        result = resolve_llm_candidate(result, node.candidates or [])
        content_type = normalize_content_type(str(result.get("contentType") or result.get("visualForm") or ""))
        predicted_type = normalize_component_type(str(result.get("componentType") or CONTENT_TO_TYPE.get(content_type, "")))
        confidence = float(result.get("confidence") or 0.0)
        text = str(result.get("text") or result.get("ocrText") or "").strip()
        paddle_ocr = paddle_ocr or {"text": "", "items": []}

        raw_type = str(node.features.get("rawClassName") or "")
        if predicted_type and should_update_type(node.type, predicted_type, confidence, raw_type=raw_type, content_type=content_type, text=text):
            node.type = predicted_type
            node.level = level_for_type(predicted_type)

        effective_result = dict(result)
        if node.type in {"Panel", "Border"}:
            content_type = "panel" if node.type == "Panel" else "border"
            predicted_type = node.type
            effective_result["contentType"] = content_type
            effective_result["componentType"] = predicted_type
            effective_result["componentId"] = ""
            effective_result["visualForm"] = "border_frame"
            effective_result["specificVisualForm"] = "border_frame"

        if node.type == "Title" and predicted_type != "Title":
            content_type = "title"
            predicted_type = "Title"
            effective_result["contentType"] = content_type
            effective_result["componentType"] = predicted_type

        effective_result = guard_incompatible_component_result(node, effective_result, self.library)
        ranked = self.rank_records(node, effective_result, top_k=top_k, crop_features=crop_features)
        if ranked:
            node.component_id = ranked[0]["componentId"]
            node.candidates = ranked

        text_evidence = str(effective_result.get("textEvidence") or effective_result.get("ocrEvidence") or "")[:240]
        structure_evidence = str(effective_result.get("structureEvidence") or "")[:240]
        if not structure_evidence:
            structure_evidence = build_structure_evidence(node, ranked)[:240]

        node.features["contentClassifier"] = {
            "mode": self.mode,
            "contentType": content_type,
            "componentType": predicted_type or node.type,
            "confidence": round(confidence, 4),
            "text": text,
            "paddleOcrText": str(paddle_ocr.get("text") or "")[:240],
            "paddleOcrItems": paddle_ocr.get("items") or [],
            "textEvidence": text_evidence,
            "visualEvidence": str(effective_result.get("visualEvidence") or "")[:240],
            "structureEvidence": structure_evidence,
            "reason": str(effective_result.get("reason") or "")[:240],
            "llmComponentId": result.get("componentId"),
            "effectiveLlmComponentId": effective_result.get("componentId"),
            "rejectedComponentId": effective_result.get("rejectedComponentId"),
            "rejectedComponentCategory": effective_result.get("rejectedComponentCategory"),
            "rejectedReason": effective_result.get("rejectedReason"),
            "llmCandidateNo": result.get("candidateNo"),
            "llmVisualForm": result.get("visualForm"),
            "localVisualForm": effective_result.get("localVisualForm"),
            "localVisualSignature": effective_result.get("localVisualSignature"),
            "visualSignature": effective_result.get("visualSignature"),
            "palette": effective_result.get("palette") or effective_result.get("dominantColors") or [],
            "dominantColors": effective_result.get("dominantColors") or effective_result.get("palette") or [],
            "dominantColor": effective_result.get("dominantColor"),
            "extractedLineData": effective_result.get("extractedLineData"),
            "modelSource": effective_result.get("modelSource"),
            "rawModelOutput": str(result.get("rawModelOutput") or "")[:500],
        }

    def rank_records(
        self,
        node: Node,
        result: Dict[str, object],
        top_k: int,
        crop_features: Optional[Dict[str, object]] = None,
    ) -> List[Dict[str, object]]:
        content_type = normalize_content_type(str(result.get("contentType") or result.get("visualForm") or ""))
        llm_component_id = str(result.get("componentId") or "")
        target_visual_form = select_target_visual_form(result)
        text = str(result.get("text") or result.get("ocrText") or "")
        llm_confidence = numeric_score(result.get("confidence"))
        evidence_text = build_candidate_evidence_text(result)
        candidate_score_by_id = candidate_scores_by_component_id(result)
        rejected_by_id = rejected_near_misses_by_component_id(result)
        llm_decision_strength = llm_component_decision_strength(result)
        categories = CONTENT_TO_CATEGORIES.get(content_type) or TYPE_TO_CATEGORIES.get(node.type, [])
        if normalize_visual_form(target_visual_form) == "vertical_bar_line_overlay" and "Bars" not in categories:
            categories = [*categories, "Bars"]
        if normalize_visual_form(target_visual_form) in {"percent_area_chart", "area_chart"} and "Areas" not in categories:
            categories = [*categories, "Areas"]
        llm_record = self.library.by_key.get(llm_component_id)
        strict_categories = strict_allowed_categories_for_node(node.type, content_type)
        if strict_categories:
            allowed = set(strict_categories)
            candidate_score_by_id = {
                component_id: score
                for component_id, score in candidate_score_by_id.items()
                if (self.library.by_key.get(component_id) and self.library.by_key[component_id].category in allowed)
            }
            if llm_record and llm_record.category not in allowed:
                llm_component_id = ""
                llm_record = None
                llm_decision_strength = 0.0
        if llm_record and llm_record.category not in categories:
            categories = [llm_record.category, *categories]
        records = self.library.records

        scored = []
        base_by_id = {str(item.get("componentId")): item for item in node.candidates or []}
        for record in records:
            base = float(base_by_id.get(record.key, {}).get("score") or 0.32)
            visual_score = self.visual_library.score(record.key, crop_features or {}) if self.visual_library.enabled else None
            if visual_score is None:
                raw_visual = base_by_id.get(record.key, {}).get("visualScore")
                visual_score = float(raw_visual) if raw_visual is not None else None

            score = 0.04 + 0.04 * base
            if visual_score is not None:
                score += 0.24 * visual_score
            llm_candidate_score = candidate_score_by_id.get(record.key)
            if llm_candidate_score is not None:
                score += 0.92 * llm_candidate_score
            if record.key in rejected_by_id:
                score -= 0.42
            if record.category in (categories or []):
                score += 0.06
            if strict_categories and record.category not in strict_categories:
                score -= 1.1
            score += 0.35 * keyword_score(record, content_type, text)
            score += aspect_score(record, node)
            if visual_score is not None and visual_score >= 0.78:
                score += 0.16
            elif visual_score is not None and visual_score < 0.42:
                score -= 0.18
            profile = self.component_profiles.get(record.key, {})
            profile_score = profile_match_score(
                profile,
                content_type,
                text,
                target_visual_form=target_visual_form,
                evidence_text=evidence_text,
            )
            score += 0.18 * profile_score
            score += 0.55 * line_component_visual_score(record.key, content_type, target_visual_form, result)
            attribute_gate = component_attribute_gate_score(record, profile, target_visual_form, result, self.visual_library)
            score += 0.72 * attribute_gate
            score += 0.32 * component_visual_gate_score(record.key, content_type, target_visual_form, result)
            if content_type in {"border", "panel"} and record.category == "Borders":
                shell_visual = max(0.0, float(visual_score or 0.0))
                score = (
                    0.08
                    + 1.22 * shell_visual
                    + 0.05 * base
                    + 0.10 * profile_score
                    + 0.08 * attribute_gate
                    + aspect_score(record, node)
                )
            if record.key == llm_component_id:
                score += 0.64 + 0.34 * llm_confidence + 0.42 * llm_decision_strength + 0.16 * profile_score
                if visual_score is not None:
                    score += 0.12 * visual_score
                if llm_candidate_score is not None:
                    score += 0.5 * llm_candidate_score
                if local_visual_form_is_strong(target_visual_form, result.get("localVisualSignature") if isinstance(result.get("localVisualSignature"), dict) else {}):
                    if attribute_gate < 0.16:
                        score -= 1.65
                    elif attribute_gate < 0.32:
                        score -= 1.25
                    elif attribute_gate < 0.55:
                        score -= 1.05
                    elif attribute_gate < 0.7:
                        score -= 0.45
            if target_visual_form and profile_score >= 0.86:
                score += 0.14
            elif target_visual_form and profile_score < 0.5:
                score -= 0.08
            raw_score = score
            display_score = normalized_display_score(raw_score, upper=0.99)
            display_profile_score = normalized_display_score(profile_score, upper=1.0)
            scored.append(
                {
                    "componentId": record.key,
                    "title": record.title,
                    "category": record.category,
                    "schema": record.schema,
                    "score": round(display_score, 4),
                    "rawScore": round(display_score, 4),
                    "_rankScore": raw_score,
                    "_profileRankScore": profile_score,
                    "visualScore": round(visual_score, 4) if visual_score is not None else None,
                    "profileScore": round(display_profile_score, 4),
                    "attributeGateScore": round(attribute_gate, 4),
                    "profileVisualForm": profile.get("visualForm") if profile else None,
                    "targetVisualForm": target_visual_form or None,
                    "evidenceText": evidence_text[:160],
                    "baseScore": round(base, 4),
                    "llmComponentId": llm_component_id or None,
                    "llmCandidateScore": round(llm_candidate_score, 4) if llm_candidate_score is not None else None,
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
                    "rawScore": 0.95,
                    "_rankScore": 0.95,
                    "_profileRankScore": 0.95,
                    "visualScore": None,
                    "baseScore": None,
                    "llmComponentId": llm_component_id,
                    "matchMode": self.mode,
                    "contentType": content_type,
                    "contentText": text[:80],
                }
            )
        scored.sort(key=lambda item: float(item.get("_rankScore") or item.get("score") or 0), reverse=True)
        if llm_component_id in self.library.by_key:
            for index, item in enumerate(scored):
                if item.get("componentId") != llm_component_id:
                    continue
                llm_item_raw = float(item.get("_rankScore") or 0)
                best_raw = float(scored[0].get("_rankScore") or 0)
                llm_attribute_gate = float(item.get("attributeGateScore") or 0)
                local_strong_conflict = (
                    local_visual_form_is_strong(
                        target_visual_form,
                        result.get("localVisualSignature") if isinstance(result.get("localVisualSignature"), dict) else {},
                    )
                    and llm_attribute_gate < 0.55
                )
                llm_profile_score = float(item.get("_profileRankScore") or item.get("profileScore") or 0)
                should_lock_llm = (
                    not local_strong_conflict
                    and (
                        llm_decision_strength >= 0.72
                        or (llm_confidence >= 0.82 and llm_profile_score >= 0.62)
                        or llm_item_raw >= best_raw - 0.08
                    )
                )
                if not should_lock_llm:
                    item["decisionSource"] = "vlm_candidate_unlocked"
                    break
                item["decisionSource"] = "vlm_final"
                item["_rankScore"] = max(float(item.get("_rankScore") or 0), float(scored[0].get("_rankScore") or 0) + 0.001)
                item["score"] = round(normalized_display_score(float(item["_rankScore"]), upper=0.99), 4)
                item["rawScore"] = item["score"]
                if index > 0:
                    scored.insert(0, scored.pop(index))
                break
        return [public_candidate_scores(item) for item in scored[:top_k]]


def normalized_display_score(value: Any, upper: float = 1.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(upper, score))


def public_candidate_scores(item: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in item.items() if not str(key).startswith("_")}


def crop_node(image: Image.Image, bbox: BBox) -> Image.Image:
    width, height = image.size
    pad = max(2, int(round(min(bbox.w, bbox.h) * 0.04)))
    left = max(0, int(round(bbox.x)) - pad)
    top = max(0, int(round(bbox.y)) - pad)
    right = min(width, int(round(bbox.right)) + pad)
    bottom = min(height, int(round(bbox.bottom)) + pad)
    return image.crop((left, top, right, bottom))


def build_candidate_evidence_text(result: Dict[str, object]) -> str:
    return " ".join(
        str(result.get(key) or "")
        for key in ["text", "ocrText", "textEvidence", "visualEvidence", "reason"]
    )


STRICT_CATEGORY_NODE_TYPES = {"Border", "Panel", "Title", "Table", "Map", "Filter", "Image"}


def strict_allowed_categories_for_node(node_type: str, content_type: str = "") -> List[str]:
    if node_type not in STRICT_CATEGORY_NODE_TYPES:
        return []
    if node_type == "Border":
        return ["Borders"]
    if node_type == "Image":
        return ["Biz", "Three", "Decorates"]
    categories = list(CONTENT_TO_CATEGORIES.get(content_type) or TYPE_TO_CATEGORIES.get(node_type, []))
    if node_type == "Title" and "Decorates" not in categories:
        categories.append("Decorates")
    return categories


def guard_incompatible_component_result(
    node: Node,
    result: Dict[str, object],
    library: ComponentLibrary,
) -> Dict[str, object]:
    content_type = normalize_content_type(str(result.get("contentType") or result.get("visualForm") or ""))
    allowed = strict_allowed_categories_for_node(node.type, content_type)
    if not allowed:
        return result

    cleaned = dict(result)
    allowed_set = set(allowed)
    scores = cleaned.get("candidateScores")
    if isinstance(scores, list):
        cleaned["candidateScores"] = [
            item
            for item in scores
            if not isinstance(item, dict)
            or not _component_category_known_and_blocked(str(item.get("componentId") or ""), allowed_set, library)
        ]

    component_id = str(cleaned.get("componentId") or "").strip()
    record = library.by_key.get(component_id)
    if not record or record.category in allowed_set:
        return cleaned

    reason = (
        f"检测类型 {node.type} 只允许 {', '.join(allowed)}；"
        f"已忽略模型候选 {record.key}({record.category})。"
    )
    previous = str(cleaned.get("structureEvidence") or "")
    cleaned["structureEvidence"] = " | ".join(item for item in [previous, reason] if item)
    cleaned["rejectedComponentId"] = record.key
    cleaned["rejectedComponentCategory"] = record.category
    cleaned["rejectedReason"] = reason
    cleaned["componentId"] = ""
    cleaned["candidateNo"] = ""
    return cleaned


def _component_category_known_and_blocked(
    component_id: str,
    allowed_categories: set[str],
    library: ComponentLibrary,
) -> bool:
    record = library.by_key.get(component_id)
    return bool(record and record.category not in allowed_categories)


def build_structure_evidence(node: Node, ranked: List[Dict[str, object]]) -> str:
    top = ranked[0] if ranked else {}
    box = node.bbox
    parts = [
        f"detector={node.type}",
        f"bbox=({round(box.x, 1)},{round(box.y, 1)},{round(box.w, 1)},{round(box.h, 1)})",
        f"parent={node.parent_id or '-'}",
    ]
    if top:
        parts.append(f"candidateCategory={top.get('category') or '-'}")
    return "; ".join(parts)


def build_context_image(image: Image.Image, node: Node) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    box = node.bbox
    draw.rectangle([box.x, box.y, box.right, box.bottom], outline=(255, 36, 66, 255), width=8)
    label = f"{node.node_id} {node.type} L{node.level}"
    label_w = max(160, len(label) * 9)
    label_y = max(0, int(box.y) - 34)
    draw.rectangle([box.x, label_y, box.x + label_w, label_y + 30], fill=(255, 36, 66, 210))
    draw.text((box.x + 8, label_y + 7), label, fill=(255, 255, 255, 255))
    canvas.thumbnail((1280, 720), Image.LANCZOS)
    return canvas


def build_structure_context(node: Node, nodes: List[Node]) -> Dict[str, object]:
    by_id = {item.node_id: item for item in nodes}
    parent = by_id.get(node.parent_id or "")
    siblings = [
        item for item in nodes
        if item.node_id != node.node_id and item.parent_id == node.parent_id and item.type not in {"Screen", "Region", "Content"}
    ]
    siblings = sorted(siblings, key=lambda item: (item.bbox.y, item.bbox.x))[:10]
    return {
        "nodeId": node.node_id,
        "detectorType": node.type,
        "level": node.level,
        "parent": node_summary(parent) if parent else None,
        "siblings": [node_summary(item) for item in siblings],
        "position": position_bucket(node.bbox),
        "aspectRatio": round(node.bbox.w / max(node.bbox.h, 1), 3),
        "area": round(node.bbox.area, 2),
    }


def node_summary(node: Optional[Node]) -> Optional[Dict[str, object]]:
    if node is None:
        return None
    candidate = node.candidates[0] if node.candidates else {}
    return {
        "nodeId": node.node_id,
        "type": node.type,
        "level": node.level,
        "componentId": node.component_id or candidate.get("componentId"),
        "bbox": node.bbox.to_dict(),
    }


def position_bucket(bbox: BBox) -> str:
    cx, cy = bbox.center
    horizontal = "left" if cx < 640 else "center" if cx < 1280 else "right"
    vertical = "top" if cy < 360 else "middle" if cy < 720 else "bottom"
    return f"{vertical}-{horizontal}"


def run_paddle_ocr(image_path: str) -> Dict[str, object]:
    if os.getenv("SCREEN_PARSER_PADDLE_OCR", "true").lower() in {"0", "false", "off", "no"}:
        return {"enabled": False, "items": [], "fullText": "", "error": "disabled"}
    try:
        engine = get_paddle_ocr_engine()
        result = engine.predict(image_path)
        payload = result[0] if isinstance(result, list) and result else result
        items = normalize_paddle_result(payload)
        return {
            "enabled": True,
            "items": items,
            "fullText": " ".join(item["text"] for item in items if item.get("text")),
            "error": None,
        }
    except Exception as exc:
        return {"enabled": False, "items": [], "fullText": "", "error": str(exc)[:240]}


def get_paddle_ocr_engine():
    global _PADDLE_OCR_ENGINE
    if _PADDLE_OCR_ENGINE is None:
        from paddleocr import PaddleOCR

        _PADDLE_OCR_ENGINE = PaddleOCR(
            lang=os.getenv("SCREEN_PARSER_PADDLE_OCR_LANG", "ch"),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _PADDLE_OCR_ENGINE


def normalize_paddle_result(payload: object) -> List[Dict[str, object]]:
    if hasattr(payload, "get"):
        rec_texts = safe_list(payload.get("rec_texts"))
        rec_scores = safe_list(payload.get("rec_scores"))
        rec_boxes = safe_list(payload.get("rec_boxes"))
        raw_polys = payload.get("rec_polys")
        if raw_polys is None:
            raw_polys = payload.get("dt_polys")
        rec_polys = safe_list(raw_polys)
    elif isinstance(payload, dict):
        rec_texts = safe_list(payload.get("rec_texts"))
        rec_scores = safe_list(payload.get("rec_scores"))
        rec_boxes = safe_list(payload.get("rec_boxes"))
        raw_polys = payload.get("rec_polys")
        if raw_polys is None:
            raw_polys = payload.get("dt_polys")
        rec_polys = safe_list(raw_polys)
    else:
        return []

    items: List[Dict[str, object]] = []
    for index, text in enumerate(rec_texts):
        text = str(text or "").strip()
        if not text:
            continue
        score = float(rec_scores[index]) if index < len(rec_scores) else 0.0
        bbox = paddle_box_to_bbox(rec_boxes[index] if index < len(rec_boxes) else None)
        if bbox is None and index < len(rec_polys):
            bbox = paddle_poly_to_bbox(rec_polys[index])
        if bbox is None:
            continue
        items.append({"text": text, "score": round(score, 4), "bbox": bbox})
    return items


def safe_list(value: object) -> List[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    try:
        return list(value)  # type: ignore[arg-type]
    except TypeError:
        return [value]


def paddle_box_to_bbox(box: object) -> Optional[Dict[str, float]]:
    if box is None:
        return None
    values = np.array(box).astype(float).reshape(-1).tolist()
    if len(values) < 4:
        return None
    x1, y1, x2, y2 = values[:4]
    return {"x": round(x1, 2), "y": round(y1, 2), "w": round(max(0.0, x2 - x1), 2), "h": round(max(0.0, y2 - y1), 2)}


def paddle_poly_to_bbox(poly: object) -> Optional[Dict[str, float]]:
    if poly is None:
        return None
    points = np.array(poly).astype(float).reshape(-1, 2)
    if points.size == 0:
        return None
    x1 = float(np.min(points[:, 0]))
    y1 = float(np.min(points[:, 1]))
    x2 = float(np.max(points[:, 0]))
    y2 = float(np.max(points[:, 1]))
    return {"x": round(x1, 2), "y": round(y1, 2), "w": round(max(0.0, x2 - x1), 2), "h": round(max(0.0, y2 - y1), 2)}


def ocr_for_node(paddle_ocr: Dict[str, object], bbox: BBox) -> Dict[str, object]:
    items = []
    for item in paddle_ocr.get("items") or []:
        item_bbox = item.get("bbox") if isinstance(item, dict) else None
        if not isinstance(item_bbox, dict):
            continue
        overlap = bbox_overlap_ratio(bbox, item_bbox)
        center_inside = bbox_contains_center(bbox, item_bbox)
        if overlap >= 0.18 or center_inside:
            copied = dict(item)
            copied["overlap"] = round(overlap, 4)
            items.append(copied)
    items.sort(key=lambda item: (float((item.get("bbox") or {}).get("y", 0)), float((item.get("bbox") or {}).get("x", 0))))
    return {
        "text": " ".join(str(item.get("text") or "") for item in items).strip()[:500],
        "items": items[:20],
        "enabled": bool(paddle_ocr.get("enabled")),
    }


def bbox_overlap_ratio(node_bbox: BBox, text_bbox: Dict[str, float]) -> float:
    tx = float(text_bbox.get("x", 0))
    ty = float(text_bbox.get("y", 0))
    tw = float(text_bbox.get("w", 0))
    th = float(text_bbox.get("h", 0))
    ix1 = max(node_bbox.x, tx)
    iy1 = max(node_bbox.y, ty)
    ix2 = min(node_bbox.right, tx + tw)
    iy2 = min(node_bbox.bottom, ty + th)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return intersection / max(1.0, tw * th)


def bbox_contains_center(node_bbox: BBox, text_bbox: Dict[str, float]) -> bool:
    cx = float(text_bbox.get("x", 0)) + float(text_bbox.get("w", 0)) / 2.0
    cy = float(text_bbox.get("y", 0)) + float(text_bbox.get("h", 0)) / 2.0
    return node_bbox.x <= cx <= node_bbox.right and node_bbox.y <= cy <= node_bbox.bottom


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
    elif node.type == "Image":
        content_type = "image"
        confidence = 0.62
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


def apply_ocr_semantic_cues(result: Dict[str, object], text: str) -> Dict[str, object]:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return result

    content_type = str(result.get("contentType") or "")
    confidence = float(result.get("confidence") or 0.0)
    percent_count = len(re.findall(r"\d+(?:\.\d+)?%", normalized))
    number_count = len(re.findall(r"\d+(?:\.\d+)?", normalized))

    table_tokens = ["任务名称", "责任单位", "任务类型", "创建时间", "完成时间", "任务状态"]
    if sum(1 for token in table_tokens if token in normalized) >= 3:
        content_type = "table"
        confidence = max(confidence, 0.9)
    elif len(re.findall(r"\d{2}-\d{2}", normalized)) >= 3:
        content_type = "line_chart"
        confidence = max(confidence, 0.86)
    elif "学历分布" in normalized or any(token in normalized for token in ["高中以下", "高中", "大专", "本科", "硕士"]):
        content_type = "bar_chart"
        confidence = max(confidence, 0.88)
    elif any(token in normalized for token in ["服务分布", "平台分布"]) and percent_count >= 3:
        content_type = "pie_chart"
        confidence = max(confidence, 0.86)
    elif any(token in normalized for token in ["排行", "Top5", "TOP5"]) and number_count >= 4:
        content_type = "bar_chart"
        confidence = max(confidence, 0.84)
    elif "调用分布指数" in normalized and number_count >= 4:
        content_type = "bar_chart"
        confidence = max(confidence, 0.84)
    elif "指标" in normalized and number_count >= 4:
        content_type = "bar_chart"
        confidence = max(confidence, 0.84)
    elif percent_count >= 3 and content_type in {"scatter_chart", "chart", "decorate"}:
        content_type = "pie_chart"
        confidence = max(confidence, 0.78)

    updated = dict(result)
    updated["contentType"] = content_type
    updated["componentType"] = CONTENT_TO_TYPE.get(content_type, str(result.get("componentType") or "Chart"))
    updated["confidence"] = confidence
    updated["text"] = normalized
    updated["reason"] = f"{result.get('reason', '')}; OCR semantic cue -> {content_type}".strip("; ")
    return updated


def add_crop_visual_signature(result: Dict[str, object], crop: Image.Image) -> Dict[str, object]:
    signature = infer_crop_visual_signature(
        crop,
        str(result.get("contentType") or ""),
        text=str(result.get("text") or result.get("ocrText") or ""),
    )
    if not signature:
        return result

    updated = dict(result)
    visual_form = str(signature.get("visualForm") or "")
    evidence = str(signature.get("visualEvidence") or "")
    if visual_form:
        updated["visualForm"] = visual_form
        inferred_content_type = content_type_for_visual_form(visual_form)
        if inferred_content_type and normalize_content_type(str(updated.get("contentType") or "")) in {"", "chart", "scatter_chart", "decorate"}:
            updated["contentType"] = inferred_content_type
            updated["componentType"] = CONTENT_TO_TYPE.get(inferred_content_type, str(updated.get("componentType") or "Chart"))
            updated["confidence"] = max(float(updated.get("confidence") or 0.0), 0.82)
    if evidence:
        previous = str(updated.get("visualEvidence") or "")
        updated["visualEvidence"] = " | ".join(item for item in [previous, evidence] if item)
        updated["reason"] = f"{updated.get('reason', '')}; visual signature -> {visual_form}".strip("; ")
    updated["visualSignature"] = signature
    palette = signature.get("palette")
    if isinstance(palette, list) and palette:
        updated["palette"] = palette
        updated["dominantColors"] = palette
        updated["dominantColor"] = palette[0]
    return updated


def infer_crop_visual_signature(crop: Image.Image, content_type: str, text: str = "") -> Dict[str, object]:
    rgb = np.array(crop.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    height, width = gray.shape[:2]
    area = float(max(1, width * height))
    edges = cv2.Canny(gray, 45, 135)

    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    mask = ((value > 45) & (saturation > 35)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    normalized_type = normalize_content_type(content_type)
    if normalized_type in {"border", "panel"}:
        mask = cv2.bitwise_and(mask, border_shell_mask(height, width))
    palette = dominant_hex_palette(rgb, hsv, mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        if contour_area >= area * 0.0025 and w >= 4 and h >= 4:
            boxes.append((x, y, w, h, contour_area))

    tall_boxes = [box for box in boxes if box[3] >= height * 0.22 and box[3] > box[2] * 1.15]
    vertical_segments = vertical_bar_segments(mask, hsv)
    segment_hues = len({segment["hueGroup"] for segment in vertical_segments if segment.get("hueGroup")})
    color_count = dominant_hue_count(hsv, mask)
    circle_count_value = circle_count(gray)
    slanted_edges = slanted_line_count(edges)
    colored_edges = cv2.bitwise_and(edges, edges, mask=mask)
    colored_slanted_edges = slanted_line_count(colored_edges)
    ellipse_caps = ellipse_cap_count(edges, width, height)
    bar_boxes = tall_boxes or segment_boxes(vertical_segments)
    cylinder_score = cylinder_bar_score(gray, hsv, edges, bar_boxes, ellipse_caps, colored_slanted_edges)
    prism_score = prism_bar_score(edges, bar_boxes, color_count, segment_hues, ellipse_caps, colored_slanted_edges)
    ring_score = max(donut_ring_score(gray, mask), local_donut_ring_score(mask))
    table_score = table_grid_score(edges, width, height)
    red_area_score = red_filled_area_score(hsv, mask)
    line_signature: Dict[str, object] = {}

    evidence_parts = [
        f"colors={color_count}",
        f"tallBars={max(len(tall_boxes), len(vertical_segments))}",
        f"segmentHues={segment_hues}",
        f"ellipseCaps={ellipse_caps}",
        f"slantedEdges={slanted_edges}",
        f"coloredSlanted={colored_slanted_edges}",
        f"cylinderScore={cylinder_score:.2f}",
        f"prismScore={prism_score:.2f}",
        f"ring={ring_score:.2f}",
        f"grid={table_score:.2f}",
        f"redArea={red_area_score:.2f}",
    ]

    visual_form = ""
    percent_count = len(re.findall(r"\d+(?:\.\d+)?%", text))
    chart_like_type = normalized_type in {"bar_chart", "chart", "scatter_chart", "decorate", ""}
    if chart_like_type:
        bar_count = max(len(tall_boxes), len(vertical_segments))
        if percent_count >= 3 and ellipse_caps >= 8 and cylinder_score >= 0.46 and color_count <= 5:
            visual_form = "liquid_vertical_bar"
            evidence_parts.extend(["liquid", "transparent_tube", "percentage_label", "round_cap", "bottom_label"])
        elif cylinder_score >= 0.58 and bar_count >= 2:
            visual_form = "gradient_cylinder_bar"
            evidence_parts.extend(["cylinder", "elliptical_cap", "gradient", "base", "top_label", "bottom_label"])
        elif prism_score >= 0.5 and bar_count >= 2:
            visual_form = "isometric_prism_bar"
            evidence_parts.extend(["isometric", "prism", "slanted_facet", "multicolor", "base", "top_label", "bottom_label"])
        elif ellipse_caps >= 16 and color_count >= 4:
            visual_form = "gradient_cylinder_bar"
            evidence_parts.extend(["cylinder", "elliptical_cap", "gradient", "base", "top_label", "bottom_label"])
        elif ellipse_caps >= 6 and color_count >= 5:
            visual_form = "isometric_prism_bar"
            evidence_parts.extend(["isometric", "prism", "multicolor", "base", "top_label", "bottom_label"])
        elif ellipse_caps >= 3 and bar_count >= 3:
            visual_form = "cylinder_vertical_bar"
            evidence_parts.extend(["cylinder", "base", "top_label", "bottom_label"])
        elif bar_count >= 3 and tube_like_bar_score(gray, hsv, bar_boxes) >= 0.42:
            visual_form = "liquid_vertical_bar"
            evidence_parts.extend(["liquid", "gradient", "base", "bottom_label"])
        elif bar_count >= 3:
            visual_form = "vertical_bar"
            evidence_parts.extend(["vertical_bar", "top_label", "bottom_label"])
    if not visual_form and normalized_type in {"pie_chart", "chart", "scatter_chart", "decorate", ""}:
        legend_like = percent_count >= 3 and (ellipse_caps < 15 or slanted_edges < 22)
        if ring_score >= 0.18 or (legend_like and color_count >= 3):
            visual_form = "donut_pie"
            evidence_parts.extend(["ring", "donut", "legend"])
        elif ring_score >= 0.12 and slanted_edges > 7:
            visual_form = "pie3d_ring"
            evidence_parts.extend(["ring", "3d", "perspective"])
        elif circle_count_value >= 1 or (percent_count >= 3 and color_count >= 3):
            visual_form = "pie3d_exploded" if slanted_edges > 7 else "flat_pie"
            evidence_parts.extend(["exploded" if slanted_edges > 7 else "flat_pie", "percent_label"])
    if not visual_form and normalized_type == "line_chart":
        line_signature = line_chart_signature(gray, hsv, mask, edges, text)
        visual_form = str(line_signature.get("visualForm") or "line_chart")
        evidence_parts.extend(line_signature.get("evidence") or [])
        evidence_parts.append(f"lineSeries={line_signature.get('seriesCount')}")
        evidence_parts.append(f"areaFill={float(line_signature.get('areaFillScore') or 0):.2f}")
        evidence_parts.append(f"barLine={float(line_signature.get('barLineScore') or 0):.2f}")
    elif normalized_type == "table":
        visual_form = "table_grid"
        evidence_parts.extend(["grid", "table", "header", "rows"])
    elif normalized_type == "title":
        visual_form = "title_text"
        evidence_parts.extend(["title", "text"])
    elif normalized_type in {"border", "panel"}:
        visual_form = "border_frame"
        evidence_parts.extend(["border", "frame"])

    return {
        "visualForm": visual_form,
        "visualEvidence": " ".join(item for item in evidence_parts if item),
        "palette": palette,
        "metrics": {
            "colorCount": color_count,
            "tallBarCount": len(tall_boxes),
            "ellipseCapCount": ellipse_caps,
            "slantedLineCount": slanted_edges,
            "coloredSlantedLineCount": colored_slanted_edges,
            "cylinderScore": round(cylinder_score, 4),
            "prismScore": round(prism_score, 4),
            "ringScore": round(ring_score, 4),
            "tableGridScore": round(table_score, 4),
            "redAreaScore": round(red_area_score, 4),
            "lineSeriesCount": line_signature.get("seriesCount"),
            "lineAreaFillScore": line_signature.get("areaFillScore"),
            "lineBarComboScore": line_signature.get("barLineScore"),
            "lineSlopedSegmentCount": line_signature.get("slopedSegmentCount"),
        },
        "seriesCount": line_signature.get("seriesCount"),
        "areaFillScore": line_signature.get("areaFillScore"),
        "barLineScore": line_signature.get("barLineScore"),
        "slopedSegmentCount": line_signature.get("slopedSegmentCount"),
    }


def dominant_hex_palette(rgb: np.ndarray, hsv: np.ndarray, mask: np.ndarray, limit: int = 8) -> List[str]:
    if rgb.size == 0 or hsv.size == 0:
        return []
    selected_mask = mask.astype(bool)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    selected_mask &= saturation > 32
    selected_mask &= value > 48
    selected_mask &= value < 250
    if int(np.count_nonzero(selected_mask)) < 24:
        return []

    pixels = rgb[selected_mask]
    hsv_pixels = hsv[selected_mask]
    bins: Dict[Tuple[int, int, int], List[int]] = {}
    for index, hsv_pixel in enumerate(hsv_pixels):
        hue_bin = int(hsv_pixel[0] // 10)
        sat_bin = int(hsv_pixel[1] // 48)
        val_bin = int(hsv_pixel[2] // 48)
        bins.setdefault((hue_bin, sat_bin, val_bin), []).append(index)

    colors: List[Tuple[float, int, str]] = []
    for indexes in bins.values():
        if len(indexes) < max(12, int(len(pixels) * 0.008)):
            continue
        cluster = pixels[indexes]
        hsv_cluster = hsv_pixels[indexes]
        r, g, b = np.median(cluster, axis=0).astype(int).tolist()
        if max(r, g, b) - min(r, g, b) < 14:
            continue
        median_saturation = float(np.median(hsv_cluster[:, 1])) / 255.0
        median_value = float(np.median(hsv_cluster[:, 2])) / 255.0
        area_ratio = len(indexes) / max(1, len(pixels))
        salience = 0.55 * median_saturation + 0.35 * median_value + 0.10 * min(1.0, area_ratio * 12.0)
        if median_value < 0.24:
            salience *= 0.42
        colors.append((salience, len(indexes), f"#{r:02x}{g:02x}{b:02x}"))

    seen = set()
    out: List[str] = []
    for _salience, _count, color in sorted(colors, key=lambda item: (item[0], item[1]), reverse=True):
        if color in seen:
            continue
        seen.add(color)
        out.append(color)
        if len(out) >= limit:
            break
    return out


def line_chart_signature(
    gray: np.ndarray,
    hsv: np.ndarray,
    mask: np.ndarray,
    edges: np.ndarray,
    text: str = "",
) -> Dict[str, object]:
    height, width = gray.shape[:2]
    area = float(max(1, width * height))
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]

    # Keep saturated chart strokes/fills, while dropping most gray grid/border lines.
    color_mask = ((value > 55) & (saturation > 38)).astype(np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    line_edges = cv2.bitwise_and(edges, edges, mask=color_mask)
    segments = cv2.HoughLinesP(
        line_edges,
        1,
        np.pi / 180,
        threshold=max(10, int(min(width, height) * 0.08)),
        minLineLength=max(14, int(width * 0.08)),
        maxLineGap=max(5, int(width * 0.035)),
    )

    sloped_segments: List[Tuple[int, int, int, int, float, str]] = []
    if segments is not None:
        for item in segments[:, 0, :]:
            x1, y1, x2, y2 = [int(v) for v in item]
            dx = x2 - x1
            dy = y2 - y1
            length = float((dx * dx + dy * dy) ** 0.5)
            if length < max(12, width * 0.055):
                continue
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            angle = min(angle, 180 - angle)
            if angle < 5 or angle > 78:
                continue
            sample_hue = median_hue_on_segment(hsv, x1, y1, x2, y2)
            sloped_segments.append((x1, y1, x2, y2, length, hue_group(sample_hue)))

    hue_groups = {segment[5] for segment in sloped_segments if segment[5]}
    colored_hues = dominant_hue_count(hsv, color_mask)
    bright_series = bright_line_series_groups(hsv)
    series_count = max(1 if sloped_segments else 0, min(4, len(bright_series) or len(hue_groups) or colored_hues))

    fill_score = line_area_fill_score(hsv, color_mask, sloped_segments)
    bar_line_score = line_bar_combo_score(color_mask, hsv, sloped_segments)
    normalized_text = str(text or "")
    percent_axis_ticks = len(re.findall(r"\b(?:0(?:\.0)?|0\.[2468]|1(?:\.0)?)\b", normalized_text))
    percent_axis = "%" in normalized_text or percent_axis_ticks >= 4
    has_area = fill_score >= 0.055
    has_many_fills = fill_score >= 0.13 and series_count >= 2
    is_combo = bar_line_score >= 0.34

    warm_single_area = has_area and len(bright_series) <= 1 and ("red" in bright_series or "orange" in bright_series)

    if is_combo:
        visual_form = "vertical_bar_line_overlay"
        evidence = ["bar_line_combo", "vertical_bar", "line", "axis"]
    elif percent_axis and has_area:
        visual_form = "percent_area_chart"
        evidence = ["percent", "area", "stacked", "axis"]
    elif has_many_fills and not warm_single_area:
        visual_form = "double_line_gradient_area"
        evidence = ["double_line", "gradient", "area", "axis"]
    elif has_area:
        visual_form = "line_gradient_area"
        evidence = ["single_line" if series_count <= 1 else "line", "gradient", "area", "axis"]
    elif series_count <= 1:
        visual_form = "line_chart_gradient"
        evidence = ["single_line", "gradient", "polyline", "axis"]
    else:
        visual_form = "line_chart"
        evidence = ["double_line", "polyline", "axis"]

    return {
        "visualForm": visual_form,
        "seriesCount": series_count,
        "areaFillScore": round(fill_score, 4),
        "barLineScore": round(bar_line_score, 4),
        "slopedSegmentCount": len(sloped_segments),
        "brightLineSeries": sorted(bright_series),
        "evidence": evidence,
    }


def bright_line_series_groups(hsv: np.ndarray) -> set[str]:
    # Count only the high-saturation, high-brightness stroke colors. This avoids
    # treating dark grid lines, diagonal background texture, or glow shadows as
    # independent data series.
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    selected = (saturation > 90) & (value > 135)
    total = int(np.count_nonzero(selected))
    if total < 80:
        return set()

    groups: Dict[str, int] = {}
    for raw_group in ["red", "orange", "yellow", "green", "cyan", "blue", "purple"]:
        if raw_group == "red":
            group_mask = selected & ((hue < 12) | (hue >= 168))
        elif raw_group == "orange":
            group_mask = selected & (hue >= 12) & (hue < 28)
        elif raw_group == "yellow":
            group_mask = selected & (hue >= 28) & (hue < 45)
        elif raw_group == "green":
            group_mask = selected & (hue >= 45) & (hue < 78)
        elif raw_group == "cyan":
            group_mask = selected & (hue >= 78) & (hue < 96)
        elif raw_group == "blue":
            group_mask = selected & (hue >= 96) & (hue < 135)
        else:
            group_mask = selected & (hue >= 135) & (hue < 168)
        count = int(np.count_nonzero(group_mask))
        if count >= max(70, int(total * 0.08)):
            groups[raw_group] = count

    # Red/orange are often the same antialiased line in dashboard screenshots.
    if "red" in groups and "orange" in groups:
        groups["red"] += groups.pop("orange")
    return set(groups.keys())


def extract_line_dataset_from_crop(crop: Image.Image, bbox: BBox, node_ocr: Dict[str, object]) -> Optional[Dict[str, object]]:
    items = [item for item in (node_ocr.get("items") or []) if isinstance(item, dict)]
    if not items:
        return None

    date_items = []
    tick_items = []
    for item in items:
        text = str(item.get("text") or "")
        item_bbox = item.get("bbox") if isinstance(item.get("bbox"), dict) else {}
        local = {
            "x": float(item_bbox.get("x") or 0) - bbox.x,
            "y": float(item_bbox.get("y") or 0) - bbox.y,
            "w": float(item_bbox.get("w") or 0),
            "h": float(item_bbox.get("h") or 0),
        }
        local["cx"] = local["x"] + local["w"] / 2.0
        local["cy"] = local["y"] + local["h"] / 2.0
        if DATE_LABEL_RE.fullmatch(text.strip()):
            date_items.append({"text": text.strip(), **local})
        elif re.fullmatch(r"\d+(?:\.\d+)?", text.strip()):
            value = float(text)
            if value >= 0:
                tick_items.append({"value": value, **local})

    date_items.sort(key=lambda item: float(item["cx"]))
    tick_items = sorted(tick_items, key=lambda item: float(item["cy"]))
    if len(date_items) < 2 or len(tick_items) < 2:
        return None

    rgb = np.array(crop.convert("RGB"))
    hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    # Prefer the warm, bright stroke. The semi-transparent area fill is darker,
    # so this isolates the curve path instead of the filled polygon.
    line_mask = (((hue < 28) | (hue >= 168)) & (saturation > 80) & (value > 115)).astype(np.uint8)
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    ys = np.array([float(item["cy"]) for item in tick_items], dtype=np.float32)
    values = np.array([float(item["value"]) for item in tick_items], dtype=np.float32)
    if len(np.unique(ys.astype(int))) < 2 or len(np.unique(values.astype(int))) < 2:
        return None
    slope, intercept = np.polyfit(ys, values, 1)

    height, width = line_mask.shape[:2]
    rows: List[Dict[str, object]] = []
    search = max(5, int(width * 0.018))
    for item in date_items[:24]:
        cx = int(round(float(item["cx"])))
        x1 = max(0, cx - search)
        x2 = min(width, cx + search + 1)
        roi = line_mask[:, x1:x2]
        points = np.argwhere(roi > 0)
        if points.size == 0:
            # Some labels sit between sampled curve points; search a little wider
            # before giving up so peaks/troughs are not flattened by OCR ticks.
            x1 = max(0, cx - search * 3)
            x2 = min(width, cx + search * 3 + 1)
            roi = line_mask[:, x1:x2]
            points = np.argwhere(roi > 0)
        if points.size == 0:
            continue
        point_ys = points[:, 0].astype(np.float32)
        # The bright stroke is thin; median is more stable than min when the
        # glow creates a few stray pixels.
        y = float(np.median(point_ys))
        sampled_value = float(slope * y + intercept)
        rows.append({
            "name": str(item["text"]),
            "value": round(max(0.0, sampled_value), 2),
            "raw": "visual-line-sample",
        })

    if len(rows) < 2:
        return None
    return {
        "dimensions": ["name", "value"],
        "source": rows,
        "method": "visual_line_path",
    }


def median_hue_on_segment(hsv: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    height, width = hsv.shape[:2]
    steps = max(6, int(max(abs(x2 - x1), abs(y2 - y1))))
    xs = np.linspace(x1, x2, steps).round().astype(np.int32)
    ys = np.linspace(y1, y2, steps).round().astype(np.int32)
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)
    selected = hsv[ys, xs]
    saturated = selected[(selected[:, 1] > 35) & (selected[:, 2] > 50)]
    if saturated.size == 0:
        return -1.0
    return float(np.median(saturated[:, 0]))


def line_area_fill_score(
    hsv: np.ndarray,
    color_mask: np.ndarray,
    sloped_segments: List[Tuple[int, int, int, int, float, str]],
) -> float:
    height, width = color_mask.shape[:2]
    if not sloped_segments:
        return 0.0

    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    # Area fills are usually broad, low-to-mid saturation color regions below a line.
    fill_mask = ((value > 38) & (saturation > 24)).astype(np.uint8)
    fill_mask = cv2.morphologyEx(fill_mask, cv2.MORPH_CLOSE, np.ones((7, 5), np.uint8))
    fill_mask = cv2.morphologyEx(fill_mask, cv2.MORPH_OPEN, np.ones((5, 3), np.uint8))

    chart_band = np.zeros_like(fill_mask)
    for x1, y1, x2, y2, _length, _hue in sloped_segments:
        top = max(0, min(y1, y2) - int(height * 0.04))
        bottom = min(height, max(y1, y2) + int(height * 0.32))
        left = max(0, min(x1, x2) - int(width * 0.035))
        right = min(width, max(x1, x2) + int(width * 0.035))
        chart_band[top:bottom, left:right] = 1

    score = float(np.count_nonzero(fill_mask & chart_band)) / float(max(1, np.count_nonzero(chart_band)))
    return min(1.0, score)


def line_bar_combo_score(
    color_mask: np.ndarray,
    hsv: np.ndarray,
    sloped_segments: List[Tuple[int, int, int, int, float, str]],
) -> float:
    if not sloped_segments:
        return 0.0
    segments = vertical_bar_segments(color_mask, hsv)
    if len(segments) < 3:
        return 0.0
    height, width = color_mask.shape[:2]
    bar_area = sum(float(segment.get("area") or 0) for segment in segments)
    bar_score = min(1.0, bar_area / float(max(1, width * height)) * 18.0)
    return min(1.0, 0.18 * len(segments) + 0.55 * bar_score)


def dominant_hue_count(hsv: np.ndarray, mask: np.ndarray) -> int:
    selected = hsv[:, :, 0][mask.astype(bool)]
    if selected.size < 20:
        return 0
    hist, _ = np.histogram(selected, bins=18, range=(0, 180))
    threshold = max(8, int(selected.size * 0.015))
    return int(np.count_nonzero(hist >= threshold))


def vertical_bar_segments(mask: np.ndarray, hsv: np.ndarray) -> List[Dict[str, object]]:
    height, width = mask.shape[:2]
    if width <= 0 or height <= 0:
        return []
    column_density = np.count_nonzero(mask, axis=0).astype(np.float32) / float(max(1, height))
    threshold = max(0.06, float(np.percentile(column_density, 72)) * 0.58)
    active = column_density >= threshold
    segments = []
    start = None
    for index, value in enumerate(active.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            end = index
            if end - start >= max(3, int(width * 0.015)):
                roi = mask[:, start:end]
                ys = np.where(np.count_nonzero(roi, axis=1) > 0)[0]
                if ys.size:
                    top, bottom = int(ys.min()), int(ys.max())
                    seg_h = bottom - top + 1
                    seg_w = end - start
                    if seg_h >= height * 0.18 and seg_h > seg_w * 1.05:
                        selected = roi.astype(bool)
                        hue_values = hsv[:, start:end, 0][selected]
                        hue = float(np.median(hue_values)) if hue_values.size else -1.0
                        segments.append(
                            {
                                "x": start,
                                "y": top,
                                "w": seg_w,
                                "h": seg_h,
                                "area": float(np.count_nonzero(roi)),
                                "hueGroup": hue_group(hue),
                            }
                        )
            start = None
    return merge_nearby_segments(segments, width)


def merge_nearby_segments(segments: List[Dict[str, object]], width: int) -> List[Dict[str, object]]:
    if not segments:
        return []
    merged = [dict(segments[0])]
    for segment in segments[1:]:
        last = merged[-1]
        gap = int(segment["x"]) - (int(last["x"]) + int(last["w"]))
        if gap <= max(2, int(width * 0.012)) and segment.get("hueGroup") == last.get("hueGroup"):
            right = max(int(last["x"]) + int(last["w"]), int(segment["x"]) + int(segment["w"]))
            bottom = max(int(last["y"]) + int(last["h"]), int(segment["y"]) + int(segment["h"]))
            last["x"] = min(int(last["x"]), int(segment["x"]))
            last["y"] = min(int(last["y"]), int(segment["y"]))
            last["w"] = right - int(last["x"])
            last["h"] = bottom - int(last["y"])
            last["area"] = float(last.get("area") or 0) + float(segment.get("area") or 0)
        else:
            merged.append(dict(segment))
    return merged


def hue_group(hue: float) -> str:
    if hue < 0:
        return ""
    if hue < 12 or hue >= 168:
        return "red"
    if hue < 28:
        return "orange"
    if hue < 45:
        return "yellow"
    if hue < 78:
        return "green"
    if hue < 96:
        return "cyan"
    if hue < 135:
        return "blue"
    return "purple"


def segment_boxes(segments: List[Dict[str, object]]) -> List[Tuple[int, int, int, int, float]]:
    return [
        (
            int(segment.get("x") or 0),
            int(segment.get("y") or 0),
            int(segment.get("w") or 0),
            int(segment.get("h") or 0),
            float(segment.get("area") or 0),
        )
        for segment in segments
    ]


def slanted_line_count(edges: np.ndarray) -> int:
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=24, minLineLength=10, maxLineGap=5)
    if lines is None:
        return 0
    count = 0
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = [float(v) for v in line]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        angle = min(angle, 180 - angle)
        if 16 <= angle <= 74:
            count += 1
    return count


def cylinder_bar_score(
    gray: np.ndarray,
    hsv: np.ndarray,
    edges: np.ndarray,
    boxes: List[Tuple[int, int, int, int, float]],
    ellipse_caps: int,
    colored_slanted_edges: int,
) -> float:
    if not boxes:
        return 0.0
    height, width = gray.shape[:2]
    bar_count = len(boxes)
    cap_score = min(1.0, ellipse_caps / max(3.0, bar_count * 0.75))
    gradient_score = vertical_gradient_score(hsv, boxes)
    upright_score = upright_bar_score(boxes, width, height)
    low_facet_score = max(0.0, 1.0 - colored_slanted_edges / max(4.0, bar_count * 1.2))
    base_score = local_ellipse_base_score(edges, boxes, width, height)
    return float(min(1.0, 0.32 * cap_score + 0.26 * gradient_score + 0.18 * upright_score + 0.14 * low_facet_score + 0.10 * base_score))


def prism_bar_score(
    edges: np.ndarray,
    boxes: List[Tuple[int, int, int, int, float]],
    color_count: int,
    segment_hues: int,
    ellipse_caps: int,
    colored_slanted_edges: int,
) -> float:
    if not boxes:
        return 0.0
    bar_count = len(boxes)
    hue_score = min(1.0, max(color_count, segment_hues) / 5.0)
    facet_score = min(1.0, colored_slanted_edges / max(3.0, bar_count * 0.8))
    cap_penalty = min(0.24, max(0.0, ellipse_caps - colored_slanted_edges) * 0.025)
    edge_density = float(np.count_nonzero(edges)) / float(max(edges.size, 1))
    detail_score = min(1.0, edge_density * 18.0)
    return float(max(0.0, min(1.0, 0.34 * facet_score + 0.24 * hue_score + 0.22 * detail_score + 0.20 * min(1.0, bar_count / 5.0) - cap_penalty)))


def vertical_gradient_score(hsv: np.ndarray, boxes: List[Tuple[int, int, int, int, float]]) -> float:
    scores: List[float] = []
    for x, y, w, h, _ in boxes:
        if w < 3 or h < 8:
            continue
        roi = hsv[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        saturation = roi[:, :, 1]
        value = roi[:, :, 2].astype(np.float32)
        active = (saturation > 35) & (value > 45)
        if np.count_nonzero(active) < max(12, int(active.size * 0.12)):
            continue
        row_values = []
        for row in range(active.shape[0]):
            row_mask = active[row]
            if np.count_nonzero(row_mask) >= max(2, int(w * 0.18)):
                row_values.append(float(np.median(value[row][row_mask])))
        if len(row_values) < 6:
            continue
        span = max(row_values) - min(row_values)
        smoothness = 1.0 - min(1.0, float(np.std(np.diff(row_values))) / 38.0)
        scores.append(min(1.0, span / 90.0) * max(0.0, smoothness))
    return float(sum(scores) / max(1, len(scores))) if scores else 0.0


def upright_bar_score(boxes: List[Tuple[int, int, int, int, float]], width: int, height: int) -> float:
    scores = []
    for _x, _y, w, h, _ in boxes:
        if h <= 0 or w <= 0:
            continue
        aspect = h / float(max(w, 1))
        scores.append(min(1.0, max(0.0, (aspect - 1.0) / 3.2)))
    return float(sum(scores) / max(1, len(scores))) if scores else 0.0


def local_ellipse_base_score(edges: np.ndarray, boxes: List[Tuple[int, int, int, int, float]], width: int, height: int) -> float:
    if not boxes:
        return 0.0
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hits = 0
    for bx, by, bw, bh, _ in boxes:
        bar_cx = bx + bw / 2.0
        bar_bottom = by + bh
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < max(8, bw * 0.75) or h < 3:
                continue
            if not (w >= h * 1.45 and h <= max(16, height * 0.16)):
                continue
            cap_cx = x + w / 2.0
            near_x = abs(cap_cx - bar_cx) <= max(10.0, bw * 1.15)
            near_y = abs(y + h / 2.0 - bar_bottom) <= max(12.0, height * 0.12)
            if near_x and near_y:
                hits += 1
                break
    return min(1.0, hits / max(1.0, len(boxes) * 0.55))


def ellipse_cap_count(edges: np.ndarray, width: int, height: int) -> int:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 8 or h < 3:
            continue
        if w >= h * 1.45 and h <= max(16, height * 0.16) and w <= width * 0.22:
            count += 1
    return count


def donut_ring_score(gray: np.ndarray, mask: np.ndarray) -> float:
    h, w = gray.shape[:2]
    cx1, cx2 = int(w * 0.34), int(w * 0.66)
    cy1, cy2 = int(h * 0.34), int(h * 0.66)
    center = mask[cy1:cy2, cx1:cx2]
    if center.size == 0:
        return 0.0
    outer = mask[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
    center_density = float(np.count_nonzero(center)) / float(center.size)
    outer_density = float(np.count_nonzero(outer)) / float(max(outer.size, 1))
    return max(0.0, min(1.0, outer_density - center_density))


def local_donut_ring_score(mask: np.ndarray) -> float:
    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = 0.0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.12 or h < height * 0.18:
            continue
        aspect = w / float(max(h, 1))
        if not 0.55 <= aspect <= 1.7:
            continue
        roi = mask[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        cx1, cx2 = int(w * 0.34), int(w * 0.66)
        cy1, cy2 = int(h * 0.34), int(h * 0.66)
        center = roi[cy1:cy2, cx1:cx2]
        outer_density = float(np.count_nonzero(roi)) / float(max(roi.size, 1))
        center_density = float(np.count_nonzero(center)) / float(max(center.size, 1))
        contour_area = cv2.contourArea(contour)
        bbox_area = float(max(1, w * h))
        ring = max(0.0, outer_density - center_density)
        compactness = max(0.0, min(1.0, contour_area / bbox_area))
        best = max(best, ring * (0.55 + 0.45 * compactness))
    return float(min(1.0, best))


def table_grid_score(edges: np.ndarray, width: int, height: int) -> float:
    area = float(max(1, width * height))
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, width // 8), 1)))
    vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, height // 8))))
    h_density = float(np.count_nonzero(horizontal)) / area
    v_density = float(np.count_nonzero(vertical)) / area
    return max(0.0, min(1.0, (h_density * 18.0 + v_density * 22.0) / 2.0))


def red_filled_area_score(hsv: np.ndarray, mask: np.ndarray) -> float:
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    red = (((hue <= 10) | (hue >= 165)) & (sat > 45) & (val > 40) & mask.astype(bool))
    return float(np.count_nonzero(red)) / float(max(1, np.count_nonzero(mask)))


def tube_like_bar_score(gray: np.ndarray, hsv: np.ndarray, boxes: List[Tuple[int, int, int, int, float]]) -> float:
    if not boxes:
        return 0.0
    scores = []
    for x, y, w, h, _ in boxes:
        roi_gray = gray[y:y + h, x:x + w]
        roi_hsv = hsv[y:y + h, x:x + w]
        if roi_gray.size == 0:
            continue
        edge = cv2.Canny(roi_gray, 45, 135)
        edge_density = float(np.count_nonzero(edge)) / float(max(edge.size, 1))
        fill_density = float(np.count_nonzero((roi_hsv[:, :, 2] > 80) & (roi_hsv[:, :, 1] > 45))) / float(max(roi_gray.size, 1))
        scores.append(min(1.0, 0.55 * edge_density * 8.0 + 0.45 * fill_density))
    return float(sum(scores) / max(1, len(scores)))


def should_call_llm(node: Node, local_result: Dict[str, object], force: bool = False) -> bool:
    if node.type in {"Screen", "Region", "Content"}:
        return False
    if node.type in {"Panel", "Border", "Decorate"}:
        return False
    content_type = normalize_content_type(str(local_result.get("contentType") or ""))
    # Component IDs must be selected from the ai-schema-view library. Local
    # rules are only coarse detectors, so every meaningful content component
    # should be verified by the VLM even when OCR made the local confidence high.
    if force and node.type in {"Chart", "Table", "Map", "MetricCard", "Filter", "Image"}:
        return True
    if node.type in {"Chart", "Table", "Map", "MetricCard", "Filter", "Image"}:
        return True
    if content_type in {
        "bar_chart",
        "line_chart",
        "area_chart",
        "pie_chart",
        "scatter_chart",
        "funnel_chart",
        "wordcloud",
        "chart",
        "table",
        "map",
        "metric_card",
        "filter",
        "image",
        "ai_shield",
        "title",
    }:
        return True
    if node.type == "Title":
        confidence = float(local_result.get("confidence") or 0.0)
        return confidence < 0.74
    confidence = float(local_result.get("confidence") or 0.0)
    return confidence < 0.82


def should_call_local_qwen(node: Node, local_result: Dict[str, object], force: bool = False) -> bool:
    if node.type in {"Screen", "Region", "Content"}:
        return False
    if force:
        return True
    if node.type in {"Chart", "Table", "Map", "MetricCard", "Filter", "Image", "Title", "Border", "Panel", "Decorate"}:
        return True
    return should_call_llm(node, local_result, force=False)


def node_processing_priority(node: Node) -> Tuple[int, float, float]:
    priority = {
        "Chart": 0,
        "Table": 1,
        "Map": 2,
        "MetricCard": 3,
        "Filter": 4,
        "Image": 5,
        "Title": 8,
        "Border": 9,
        "Panel": 10,
        "Decorate": 11,
        "Screen": 99,
        "Region": 99,
        "Content": 99,
    }
    return (priority.get(node.type, 20), node.bbox.y, node.bbox.x)


def build_prompt(
    node: Node,
    candidates: List[ComponentRecord],
    paddle_ocr: Optional[Dict[str, object]] = None,
    structure_context: Optional[Dict[str, object]] = None,
    component_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
    crop_features: Optional[Dict[str, object]] = None,
) -> str:
    candidate_lines = []
    component_profiles = component_profiles or {}
    for index, record in enumerate(candidates, start=1):
        profile = component_profiles.get(record.key, {})
        candidate_lines.append(
            {
                "candidateNo": f"C{index}",
                "componentId": record.key,
                "title": record.title,
                "category": record.category,
                "categoryName": record.category_name,
                "profileContentType": profile.get("contentType"),
                "profileVisualForm": profile.get("visualForm"),
                "profileLayout": profile.get("layout"),
                "profileKeywords": profile.get("semanticKeywords", [])[:10] if isinstance(profile.get("semanticKeywords"), list) else [],
                "profileDistinguishingFeatures": profile.get("distinguishingFeatures", [])[:8] if isinstance(profile.get("distinguishingFeatures"), list) else [],
                "profileNegativeMatches": profile.get("negativeMatches", [])[:6] if isinstance(profile.get("negativeMatches"), list) else [],
                "description": record.description[:280],
            }
        )
    paddle_ocr = paddle_ocr or {}
    structure_context = structure_context or {}
    crop_features = crop_features or {}
    return json.dumps(
        {
            "task": (
                "Match a detected dashboard component to the best component-library candidate. "
                "You must use BOTH visual structure and text semantics. First read the text in the crop yourself, "
                "then compare it with the provided PaddleOCR text. Use the full-screen context image with a red box "
                "to understand where the component sits in the hierarchy. Do not choose only by color/style; "
                "choose by visible chart/table/map/metric/title/filter structure plus the meaning of the text."
            ),
            "imageInputs": [
                "image1: detected component crop",
                "image2: full dashboard context with the target component in a red box",
                "image3 when present: numbered candidate sheet from the component library",
            ],
            "detectorType": node.type,
            "nodeId": node.node_id,
            "bbox": node.bbox.to_dict(),
            "layoutContext": structure_context,
            "paddleOcr": {
                "text": paddle_ocr.get("text", ""),
                "items": paddle_ocr.get("items", []),
            },
            "cropVisualSignature": compact_crop_signature(crop_features),
            "fineDetailChecklist": {
                "barCharts": [
                    "liquidBar: transparent tube/frame, orange/colored liquid fill, wavy liquid surface, percentage labels; not a normal prism/cylinder.",
                    "ColorPrismBar: black background/back pillars plus colorful 3D prism bars, diamond or slanted polygon top/bottom facets, visible prism edges.",
                    "PrismaticBar: 3D prism facets without the distinctive black background pillars of ColorPrismBar; choose it only when the bar body has angular/slanted facets.",
                    "clor: gradient cylinder bars, elliptical top caps, round/ellipse base rings, smooth vertical gradient body; no diamond prism facets.",
                    "CylinderBar: cylindrical/elliptical top caps and rounded column feel; no diamond prism facets.",
                    "BarCommon/VChartBarCommon: ordinary flat bars, no prism/cylinder/liquid tube details.",
                    "BarLine: bars and line overlay in the same plot; must visibly contain both.",
                    "CapsuleChart: horizontal capsule/progress style rather than vertical chart columns.",
                ],
                "lineAreaCharts": [
                    "LineCommon/VChartLine: pure line chart; no filled area under the curve.",
                    "LineGradientSingle/LineGradients: gradient glow/fill or stylized line; distinguish one series vs multiple series.",
                    "VChartArea: filled area chart; visible area under curve.",
                    "VChartPercentArea: stacked/percent area behavior, percentage-like composition.",
                    "BarLine: line plus bar columns; do not choose if only a line exists.",
                ],
                "pieCharts": [
                    "PieCommon/VChartPie: flat pie or donut; no 3D thickness.",
                    "PieCircle: single center/ring percent indicator, not multi-category pie.",
                    "Pie3DExploded: 3D thick slices and separated/exploded pieces.",
                    "Pie3DRingRegion/Pie3DRingUser: 3D donut/ring with central hole; distinguish from solid 3D pie.",
                    "Pie3DMultiLayer/Pie3DTwoBlue/Pie3DTwoCyan: layered/two-tone 3D pie style; check number of layers and color scheme.",
                ],
                "tablesLists": [
                    "TableScrollBoard: scrolling table/list with headers and rows, often dense operational data.",
                    "TablesBasic: plain grid table.",
                    "AlarmList/AlarmStatus: alarm/status-specific list or bubbles; look for status levels, badges, warning labels.",
                ],
                "nonCharts": [
                    "title1: short banner heading with decorative blue title background.",
                    "TextCommon/TextGradient/TextBarrage: text-only blocks, not data charts.",
                    "Borders/Decorates: frames and ornaments; no actual data series.",
                    "AIShield/AIRobot/KeySecurity3D: custom business illustrations; central shield/robot/3D security scene.",
                    "Inputs*: interactive control shapes such as input/select/date/tab/pagination.",
                ],
            },
            "candidateComponentIds": candidate_lines,
            "allowedContentTypes": sorted(CONTENT_TO_TYPE.keys()),
            "decisionRules": [
                "The component library spans all ai-schema-view packages: Charts, VChart, Informations, Tables, Decorates, Photos, Icons and Customs. Do not assume the component is a chart just because detectorType says Chart.",
                "If the crop is a title, text block, metric card, table/list, input/filter, border/decorate, image/icon, or custom business visual, choose that library component instead of a chart.",
                "If PaddleOCR text includes title-like phrases, metric labels, column headers, legends, or filter words, use them as strong evidence.",
                "Use text to identify content family and data parameters. Use visible geometry to select the exact componentId.",
                "If OCR text conflicts with the visual guess, explain the conflict and prefer the candidate matching both structure and text.",
                "You must distinguish fine-grained component forms, not only broad categories. For example: liquid vertical bar vs normal bar vs cylinder bar vs prism bar; donut/ring pie vs exploded 3D pie; normal table vs alarm/status list.",
                "Before selecting componentId, compare the target crop against at least three visually similar candidates and reject the near-misses using concrete small details.",
                "Do not copy the previous detector or OCR guess. The final componentId must be your own visual+text judgment against the candidate sheet and profiles.",
                "Use profileVisualForm and profileDistinguishingFeatures as the component library ground truth. Choose the candidate whose preview/profile would look closest after data is injected.",
                "Score the strongest candidate components across multiple dimensions: shapeGeometry, perspective3D, colorStyle, baseAndCap, textDataFit, layoutFit, profileFit, negativeMismatch. Do not let a single word such as cylinder/bar/pie decide the result.",
                "Return candidateScores for at most 8 candidates: the final chosen component plus the strongest visually similar near-misses. Keep every evidence field under 24 words so the JSON stays valid.",
                "For line and area charts, distinguish single line vs multiple lines, filled area vs pure line, straight/polyline vs smooth curve, gradient fill, axis tick labels, legend count, and whether bars are combined with a line. OCR axis ticks are labels, not the dataset itself; infer data shape from the visible curve when possible.",
                "For bar charts, distinguish ColorPrismBar/PrismaticBar/clor/CylinderBar/liquidBar/normal bars by actual shape attributes: elliptical caps and round bases imply cylinder; diamond/slanted polygon facets imply prism; liquid tubes imply liquidBar; color count alone is not enough.",
                "For pies, distinguish flat pie/donut from 3D exploded/ring/multilayer pies by thickness, perspective ellipse, separated slices, inner hole, and layer count.",
                "For tables/lists, identify headers/rows and choose table/list components instead of generic charts.",
                "For metric cards, look for large numbers plus label text.",
                "For titles, prioritize short heading text and title bar layout.",
                "For maps/network diagrams, consider central labels, node labels, geography words, and spatial structure.",
            ],
                "returnJsonSchema": {
                "contentType": "one of allowedContentTypes, e.g. bar_chart/table/metric_card/title/map/filter/image",
                "componentType": "Panel|Title|Chart|Table|Map|MetricCard|Border|Decorate|Filter|Image",
                "componentId": "best candidate componentId or empty string",
                "candidateNo": "candidateNo such as C1/C2, or empty string",
                "visualForm": "fine-grained form visible in image1, e.g. liquid_vertical_bar/donut_pie/cylinder_bar/prism_bar/line_chart/table_grid/border_frame",
                "candidateScores": [
                    {
                        "candidateNo": "C1",
                        "componentId": "candidate componentId",
                        "visualMatchScore": "0-1 overall visual match after checking all dimensions",
                        "shapeGeometry": "0-1",
                        "perspective3D": "0-1",
                        "colorStyle": "0-1",
                        "baseAndCap": "0-1",
                        "textDataFit": "0-1",
                        "layoutFit": "0-1",
                        "profileFit": "0-1",
                        "negativeMismatch": "0-1 where 1 means no important mismatch; lower when candidate has details absent from crop",
                        "evidence": "short concrete evidence for this candidate"
                    }
                ],
                "confidence": "0-1",
                "text": "all important visible text you can read from image1, corrected using PaddleOCR when useful",
                "textEvidence": "which words/phrases influenced the decision",
                "visualEvidence": "visual structure evidence: chart/table/map/title/card/filter/panel cues",
                "structureEvidence": "hierarchy/position/parent/sibling evidence from layoutContext and image2",
                "rejectedNearMisses": [
                    {
                        "componentId": "near miss componentId",
                        "whyRejected": "specific detail absent or conflicting"
                    }
                ],
                "reason": "short final reason combining text + visual structure + hierarchy",
            },
        },
        ensure_ascii=False,
    )


def compact_crop_signature(crop_features: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(crop_features, dict) or not crop_features:
        return {}
    structural = crop_features.get("structural") if isinstance(crop_features.get("structural"), dict) else {}
    keys = [
        "aspectRatio",
        "edgeDensity",
        "horizontalDensity",
        "verticalDensity",
        "dominantColor",
        "brightness",
        "contrast",
    ]
    signature = {key: crop_features.get(key) for key in keys if key in crop_features}
    if structural:
        signature["structural"] = {
            key: structural.get(key)
            for key in [
                "barLikeScore",
                "lineLikeScore",
                "circleLikeScore",
                "tableLikeScore",
                "largeConnectedComponents",
                "verticalComponentCount",
                "horizontalComponentCount",
            ]
            if key in structural
        }
    return signature


def build_candidate_sheet(candidates: List[ComponentRecord], visual_library: VisualReferenceLibrary) -> Optional[Image.Image]:
    if not candidates or not visual_library.enabled:
        return None

    tile_w, tile_h = 220, 154
    cols = 4
    rows = int(np.ceil(len(candidates) / cols))
    sheet = Image.new("RGB", (cols * tile_w, max(1, rows) * tile_h), (10, 16, 28))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    has_image = False

    for index, record in enumerate(candidates):
        col = index % cols
        row = index // cols
        x = col * tile_w
        y = row * tile_h
        draw.rectangle([x, y, x + tile_w - 1, y + tile_h - 1], outline=(65, 145, 190), width=1)
        label = f"C{index + 1} {record.key}"
        draw.text((x + 8, y + 6), label[:34], font=font, fill=(235, 245, 255))

        image_path = resolve_reference_image_path(record, visual_library)
        if not image_path:
            draw.text((x + 8, y + 58), record.title[:28], font=font, fill=(180, 205, 225))
            continue

        try:
            ref = Image.open(image_path).convert("RGBA")
        except Exception:
            draw.text((x + 8, y + 58), record.title[:28], font=font, fill=(180, 205, 225))
            continue

        has_image = True
        ref.thumbnail((tile_w - 18, tile_h - 42), Image.LANCZOS)
        px = x + (tile_w - ref.width) // 2
        py = y + 34 + (tile_h - 42 - ref.height) // 2
        background = Image.new("RGBA", ref.size, (12, 22, 38, 255))
        background.alpha_composite(ref)
        sheet.paste(background.convert("RGB"), (px, py))

    return sheet if has_image else None


def resolve_reference_image_path(record: ComponentRecord, visual_library: VisualReferenceLibrary) -> Optional[Path]:
    reference = visual_library.by_component_id.get(record.key)
    paths = []
    if reference and reference.image_path:
        paths.append(Path(reference.image_path))
    paths.extend(
        [
            PROJECT_ROOT / "data" / "component-reference" / "images" / f"{record.key}.png",
            PROJECT_ROOT / "data" / "component-reference" / "images" / f"{record.key}.jpg",
            PROJECT_ROOT / "data" / "component-reference" / "images" / f"{record.key}.jpeg",
        ]
    )
    if reference and reference.image_path:
        paths.append(PROJECT_ROOT / "data" / "component-reference" / "images" / Path(reference.image_path).name)

    for path in paths:
        if path.exists():
            return path
    return None


def resolve_llm_candidate(result: Dict[str, object], candidates: List[Dict[str, object]]) -> Dict[str, object]:
    resolved = dict(result)
    component_id = str(resolved.get("componentId") or "")
    candidate_no = str(resolved.get("candidateNo") or resolved.get("candidateIndex") or "").strip()
    if re.fullmatch(r"C\d+", component_id.strip(), flags=re.I):
        candidate_no = component_id
        component_id = ""
    if component_id:
        return resolved

    match = re.search(r"\d+", candidate_no)
    if not match:
        return resolved
    index = int(match.group(0)) - 1
    if 0 <= index < len(candidates):
        resolved["componentId"] = candidates[index].get("componentId") or ""
        resolved["candidateNo"] = f"C{index + 1}"
    return resolved


def resolve_llm_candidate_from_records(result: Dict[str, object], candidates: List[ComponentRecord]) -> Dict[str, object]:
    resolved = dict(result)
    component_id = str(resolved.get("componentId") or "")
    candidate_no = str(resolved.get("candidateNo") or resolved.get("candidateIndex") or "").strip()
    if re.fullmatch(r"C\d+", component_id.strip(), flags=re.I):
        candidate_no = component_id
        component_id = ""
    if component_id:
        return resolved

    match = re.search(r"\d+", candidate_no)
    if not match:
        return resolved
    index = int(match.group(0)) - 1
    if 0 <= index < len(candidates):
        resolved["componentId"] = candidates[index].key
        resolved["candidateNo"] = f"C{index + 1}"
    return resolved


def attach_candidate_scores(result: Dict[str, object], candidates: List[ComponentRecord]) -> None:
    scores = result.get("candidateScores")
    if not isinstance(scores, list):
        return
    for item in scores:
        if not isinstance(item, dict):
            continue
        component_id = str(item.get("componentId") or "")
        candidate_no = str(item.get("candidateNo") or "").strip()
        if not component_id:
            match = re.search(r"\d+", candidate_no)
            if match:
                index = int(match.group(0)) - 1
                if 0 <= index < len(candidates):
                    component_id = candidates[index].key
        if component_id:
            item["componentId"] = component_id


def candidate_scores_by_component_id(result: Dict[str, object]) -> Dict[str, float]:
    scores = result.get("candidateScores")
    if not isinstance(scores, list):
        return {}
    by_id: Dict[str, float] = {}
    dimension_keys = [
        "shapeGeometry",
        "perspective3D",
        "colorStyle",
        "baseAndCap",
        "textDataFit",
        "layoutFit",
        "profileFit",
        "negativeMismatch",
    ]
    for item in scores:
        if not isinstance(item, dict):
            continue
        component_id = str(item.get("componentId") or "")
        if not component_id:
            continue
        visual = numeric_score(item.get("visualMatchScore"))
        dimensions = [numeric_score(item.get(key)) for key in dimension_keys if item.get(key) is not None]
        dimension_score = sum(dimensions) / len(dimensions) if dimensions else 0.0
        final = 0.48 * visual + 0.52 * dimension_score if dimensions else visual
        by_id[component_id] = max(by_id.get(component_id, 0.0), final)
    return by_id


def rejected_near_misses_by_component_id(result: Dict[str, object]) -> Dict[str, str]:
    items = result.get("rejectedNearMisses")
    if not isinstance(items, list):
        return {}
    rejected: Dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        component_id = str(item.get("componentId") or "").strip()
        if not component_id:
            continue
        rejected[component_id] = str(item.get("whyRejected") or "")[:120]
    return rejected


def llm_component_decision_strength(result: Dict[str, object]) -> float:
    component_id = str(result.get("componentId") or "")
    if not component_id:
        return 0.0
    scores = candidate_scores_by_component_id(result)
    chosen_score = scores.get(component_id, 0.0)
    alternatives = sorted((score for key, score in scores.items() if key != component_id), reverse=True)
    gap = chosen_score - (alternatives[0] if alternatives else 0.0)
    confidence = numeric_score(result.get("confidence"))
    rejected_count = len(rejected_near_misses_by_component_id(result))
    evidence_bonus = min(0.1, 0.025 * rejected_count)
    return max(0.0, min(1.0, 0.5 * chosen_score + 0.32 * confidence + 0.18 * max(0.0, gap) + evidence_bonus))


def numeric_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


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
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None


def merge_llm_and_local(llm_result: Dict[str, object], local_result: Dict[str, object]) -> Dict[str, object]:
    merged = dict(local_result)
    merged.update({key: value for key, value in llm_result.items() if value not in [None, ""]})
    merged["localContentType"] = local_result.get("contentType")
    merged["localVisualForm"] = local_result.get("visualForm")
    merged["localVisualSignature"] = local_result.get("visualSignature")
    merged["localReason"] = local_result.get("reason")
    return merged


def select_target_visual_form(result: Dict[str, object]) -> str:
    llm_form = str(result.get("visualForm") or result.get("specificVisualForm") or "")
    local_form = str(result.get("localVisualForm") or "")
    local_signature = result.get("localVisualSignature") if isinstance(result.get("localVisualSignature"), dict) else {}
    if not local_signature:
        local_signature = result.get("visualSignature") if isinstance(result.get("visualSignature"), dict) else {}
    if local_form and local_visual_form_is_strong(local_form, local_signature):
        return local_form
    return llm_form or local_form


def local_visual_form_is_strong(visual_form: str, signature: Dict[str, object]) -> bool:
    form = normalize_visual_form(visual_form)
    metrics = signature.get("metrics") if isinstance(signature.get("metrics"), dict) else {}
    if not form:
        return False
    if form == "liquid_vertical_bar":
        return float(metrics.get("ellipseCapCount") or 0) >= 6
    if form == "gradient_cylinder_bar":
        return float(metrics.get("cylinderScore") or 0) >= 0.5 or float(metrics.get("ellipseCapCount") or 0) >= 12
    if form in {"isometric_prism_bar", "prismatic_vertical_bar"}:
        return float(metrics.get("prismScore") or 0) >= 0.45 or float(metrics.get("colorCount") or 0) >= 5
    if "pie" in form or "donut" in form or "ring" in form:
        return float(metrics.get("ringScore") or 0) >= 0.08 or float(metrics.get("slantedLineCount") or 0) >= 6
    return True


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
        "img": "image",
        "photo": "image",
        "picture": "image",
        "visual": "image",
        "shield": "ai_shield",
    }
    key = aliases.get(key, key)
    return key if key in CONTENT_TO_TYPE else "chart" if "chart" in key else key


def normalize_component_type(value: str) -> str:
    value = value.strip()
    if value in {"Region", "Panel", "Content", "Title", "Chart", "Table", "Map", "MetricCard", "Border", "Decorate", "Filter", "Image"}:
        return value
    lowered = value.lower()
    for content_type, component_type in CONTENT_TO_TYPE.items():
        if content_type in lowered:
            return component_type
    return ""


def should_update_type(
    current: str,
    predicted: str,
    confidence: float,
    raw_type: str = "",
    content_type: str = "",
    text: str = "",
) -> bool:
    if predicted == current:
        return False
    raw_lower = raw_type.lower()
    if ("border" in raw_lower or "panel" in raw_lower) and predicted not in {"Border", "Panel"}:
        return False
    if (current == "Image" or "image" in raw_lower) and predicted != "Image":
        return False
    title_like = current == "Title" or "title" in raw_lower or "text" in raw_lower
    if title_like and predicted != "Title":
        if confidence < 0.9:
            return False
        non_title_content = content_type not in {"", "title", "text", "decorate", "panel", "border"}
        text_evidence = len(text.strip()) >= 18 or "\n" in text or "|" in text
        if not (non_title_content and text_evidence):
            return False
    if current in {"Panel", "Border"}:
        return False
    if predicted in {"Panel", "Border"} and current not in {"Panel", "Border"}:
        return False
    return confidence >= 0.52


def level_for_type(node_type: str) -> int:
    return {
        "Panel": 2,
        "Border": 2,
        "Content": 3,
        "Title": 3,
        "Decorate": 3,
        "Filter": 4,
        "Chart": 4,
        "Table": 4,
        "Map": 4,
        "MetricCard": 4,
        "Image": 4,
    }.get(node_type, 4)


def keyword_score(record: ComponentRecord, content_type: str, text: str) -> float:
    haystack = f"{record.key} {record.title} {record.category} {record.category_name} {record.description} {text}".lower()
    score = 0.0
    for token in CONTENT_KEYWORDS.get(content_type, []):
        if token.lower() in haystack:
            score += 0.055
    return min(0.24, score)


def line_component_visual_score(
    component_id: str,
    content_type: str,
    visual_form: str,
    result: Dict[str, object],
) -> float:
    normalized_type = normalize_content_type(content_type)
    normalized_form = normalize_visual_form(visual_form)
    if normalized_type not in {"line_chart", "area_chart"} and normalized_form not in {
        "line_chart",
        "line_chart_gradient",
        "line_gradient_area",
        "double_line_gradient_area",
        "percent_area_chart",
        "vertical_bar_line_overlay",
    }:
        return 0.0

    signature = result.get("visualSignature") if isinstance(result.get("visualSignature"), dict) else {}
    metrics = signature.get("metrics") if isinstance(signature.get("metrics"), dict) else {}
    line_metrics = {}
    if isinstance(signature, dict):
        # New line-specific metrics are kept at the top level of the signature.
        line_metrics = signature
    series_count = int(float(line_metrics.get("seriesCount") or 0))
    area_fill = float(line_metrics.get("areaFillScore") or metrics.get("redAreaScore") or 0)
    bar_line = float(line_metrics.get("barLineScore") or 0)

    score = 0.0
    cid = component_id.lower()
    if normalized_form == "vertical_bar_line_overlay":
        score += 0.36 if cid == "barline" else -0.16
    elif normalized_form == "double_line_gradient_area":
        score += 0.34 if component_id == "LineGradients" else -0.1
        if component_id == "LineGradientSingle" and series_count >= 2:
            score -= 0.14
    elif normalized_form == "line_gradient_area":
        score += 0.34 if component_id == "LineGradientSingle" else -0.08
        if component_id == "LineGradients" and series_count >= 2 and area_fill >= 0.1:
            score += 0.12
    elif normalized_form == "line_chart_gradient":
        score += 0.32 if component_id == "LineLinearSingle" else -0.06
        if component_id in {"LineGradientSingle", "LineGradients"} and area_fill < 0.045:
            score -= 0.08
    elif normalized_form == "percent_area_chart":
        score += 0.38 if component_id == "VChartPercentArea" else -0.08
        if component_id == "VChartArea":
            score += 0.08
    elif normalized_form == "area_chart":
        score += 0.3 if component_id == "VChartArea" else -0.06
    elif normalized_form == "line_chart":
        if component_id in {"LineCommon", "VChartLine"}:
            score += 0.24
        if component_id in {"LineGradientSingle", "LineGradients"} and area_fill < 0.045:
            score -= 0.08

    if series_count <= 1 and component_id in {"LineGradientSingle", "LineLinearSingle"}:
        score += 0.06
    if series_count >= 2 and component_id in {"LineCommon", "LineGradients", "VChartLine"}:
        score += 0.05
    if area_fill >= 0.07 and component_id in {"LineGradientSingle", "LineGradients", "VChartArea", "VChartPercentArea"}:
        score += 0.08
    if bar_line >= 0.34 and component_id == "BarLine":
        score += 0.12
    return score


def component_visual_gate_score(
    component_id: str,
    content_type: str,
    visual_form: str,
    result: Dict[str, object],
) -> float:
    normalized_type = normalize_content_type(content_type)
    normalized_form = normalize_visual_form(visual_form)
    text = " ".join(
        str(result.get(key) or "")
        for key in ["text", "ocrText", "paddleOcrText", "textEvidence", "visualEvidence"]
    )
    cid = component_id

    if normalized_type == "bar_chart":
        # Fine-grained bar matching is handled by component_attribute_gate_score,
        # which uses each component's profile and reference attributes rather
        # than hard-coded component IDs.
        return 0.0

    if normalized_type in {"line_chart", "area_chart"}:
        if normalized_form in {"line_gradient_area", "line_chart_gradient", "double_line_gradient_area", "percent_area_chart", "area_chart"}:
            if cid in {"VChartScatter", "ScatterCommon"}:
                return -0.62
            if cid in {"ColorPrismBar", "PrismaticBar", "CylinderBar", "VChartBarCommon", "BarCommon", "CapsuleChart", "liquidBar"}:
                return -0.48

    if normalized_type == "pie_chart":
        if normalized_form == "donut_pie":
            if cid in {"VChartPie", "PieCommon", "PieCircle"}:
                return 0.22
            if cid.startswith("Pie3D"):
                return -0.12
        if normalized_form == "pie3d_exploded":
            if cid == "Pie3DExploded":
                return 0.36
            if cid in {"VChartPie", "PieCommon", "PieCircle"}:
                return -0.18
        if normalized_form == "pie3d_ring":
            if cid in {"Pie3DRingRegion", "Pie3DRingUser"}:
                return 0.34

    if normalized_type == "table":
        has_many_headers = sum(1 for token in ["任务名称", "责任单位", "任务类型", "创建时间", "完成时间", "任务状态"] if token in text) >= 3
        if has_many_headers:
            if cid in {"TableScrollBoard", "TablesBasic"}:
                return 0.34
            if cid == "AlarmList":
                return -0.22
            if cid == "TableList":
                return -0.08
        elif cid in {"AlarmList", "TableScrollBoard", "TablesBasic"}:
            return 0.08

    return 0.0


def component_attribute_gate_score(
    record: ComponentRecord,
    profile: Dict[str, Any],
    target_visual_form: str,
    result: Dict[str, object],
    visual_library: VisualReferenceLibrary,
) -> float:
    target = target_visual_attributes(target_visual_form, result)
    candidate = candidate_visual_attributes(record, profile, visual_library)
    if not target.primary and not target.attrs:
        return 0.0

    score = 0.0
    if target.primary and candidate.primary:
        if target.primary == candidate.primary:
            score += 0.58
        elif visual_form_compatible(candidate.primary, target.primary):
            score += 0.34
        elif form_family(candidate.primary) == form_family(target.primary):
            score += 0.08
        else:
            score -= 0.28

    shared = target.attrs & candidate.attrs
    score += min(0.32, 0.055 * len(shared))
    score += min(0.18, 0.045 * len(target.required & candidate.attrs))

    for left, right in [
        ("cylinder", "prism"),
        ("liquid", "solid"),
        ("horizontal", "vertical"),
        ("table", "chart"),
        ("line", "bar"),
        ("pie", "bar"),
        ("pie", "line"),
    ]:
        if left in target.required and right in candidate.strong:
            score -= 0.3
        if right in target.required and left in candidate.strong:
            score -= 0.3

    missing_required = target.required - candidate.attrs
    score -= min(0.36, 0.12 * len(missing_required))

    if "gradient" in target.attrs and "gradient" not in candidate.attrs and {"cylinder", "prism"} & target.attrs:
        score -= 0.08
    if "round_base" in target.required and "round_base" not in candidate.attrs and "cylinder" not in candidate.attrs:
        score -= 0.12
    if "facet" in target.required and "facet" not in candidate.attrs and "prism" not in candidate.attrs:
        score -= 0.14

    return float(max(-0.75, min(0.9, score)))


@dataclass(frozen=True)
class VisualAttributes:
    primary: str
    attrs: set[str]
    strong: set[str]
    required: set[str]


def target_visual_attributes(visual_form: str, result: Dict[str, object]) -> VisualAttributes:
    signature = result.get("visualSignature") if isinstance(result.get("visualSignature"), dict) else {}
    metrics = signature.get("metrics") if isinstance(signature.get("metrics"), dict) else {}
    text = " ".join(
        str(result.get(key) or "")
        for key in ["visualEvidence", "reason", "textEvidence", "text", "ocrText"]
    )
    forms = [normalize_visual_form(visual_form), normalize_visual_form(str(signature.get("visualForm") or ""))]
    primary = next((item for item in forms if item), "")
    attrs = visual_attrs_from_forms(forms) | visual_attrs_from_text(text)
    strong: set[str] = set()
    required: set[str] = set()

    cylinder_score = float(metrics.get("cylinderScore") or 0)
    prism_score = float(metrics.get("prismScore") or 0)
    ellipse_caps = float(metrics.get("ellipseCapCount") or 0)
    colored_slanted = float(metrics.get("coloredSlantedLineCount") or 0)
    if cylinder_score >= 0.58 and cylinder_score >= prism_score + 0.08:
        attrs.update({"cylinder", "round_cap", "round_base"})
        strong.add("cylinder")
        required.update({"cylinder", "round_cap"})
        if "gradient" in attrs or "gradient_cylinder" in primary:
            required.add("gradient")
        if "round_base" in attrs or "gradient_cylinder" in primary:
            required.add("round_base")
    if prism_score >= 0.55 and colored_slanted >= 3:
        attrs.update({"prism", "facet"})
        strong.add("prism")
        required.update({"prism", "facet"})
    if ellipse_caps >= 3:
        attrs.add("round_cap")
    if colored_slanted >= 3:
        attrs.add("facet")

    return VisualAttributes(primary=primary, attrs=attrs, strong=strong, required=required)


def content_type_for_visual_form(visual_form: str) -> str:
    form = normalize_visual_form(visual_form)
    if not form:
        return ""
    if any(token in form for token in ["bar", "cylinder", "prism", "liquid"]):
        return "bar_chart"
    if any(token in form for token in ["pie", "donut", "ring"]):
        return "pie_chart"
    if any(token in form for token in ["line", "area"]):
        return "line_chart"
    if "table" in form or "grid" in form:
        return "table"
    if "scatter" in form:
        return "scatter_chart"
    if "map" in form:
        return "map"
    if any(token in form for token in ["image", "photo", "picture", "shield", "robot", "illustration", "visual"]):
        return "image"
    return ""


def candidate_visual_attributes(
    record: ComponentRecord,
    profile: Dict[str, Any],
    visual_library: VisualReferenceLibrary,
) -> VisualAttributes:
    reference = visual_library.by_component_id.get(record.key) if visual_library.enabled else None
    structural = reference.features.get("structural", {}) if reference and isinstance(reference.features, dict) else {}
    forms = [
        normalize_visual_form(str(profile.get("visualForm") or "")),
        normalize_visual_form(str(structural.get("profileVisualForm") or "")),
        normalize_visual_form(str(structural.get("primaryForm") or "")),
    ]
    forms.extend(normalize_visual_form(str(item)) for item in (structural.get("forms") or []) if item)
    text = " ".join(
        str(value or "")
        for value in [
            record.key,
            record.title,
            record.category,
            record.description,
            " ".join(profile.get("distinguishingFeatures") or []) if isinstance(profile.get("distinguishingFeatures"), list) else "",
        ]
    )
    attrs = visual_attrs_from_forms(forms) | visual_attrs_from_text(text)
    strong = visual_attrs_from_forms(forms[:2])
    primary = next((item for item in forms if item), "")
    required = set(strong)
    if "cylinder" in strong:
        required.add("round_cap")
    if "prism" in strong:
        required.add("facet")
    return VisualAttributes(primary=primary, attrs=attrs, strong=strong, required=required)


def visual_attrs_from_forms(forms: Iterable[str]) -> set[str]:
    attrs: set[str] = set()
    for raw in forms:
        form = normalize_visual_form(str(raw or ""))
        if not form:
            continue
        if "cylinder" in form or "round_column" in form:
            attrs.update({"bar", "cylinder", "round_cap", "solid"})
        if "gradient_cylinder" in form or "3d_cylinder" in form:
            attrs.add("gradient")
            attrs.add("round_base")
        if "prism" in form or "prismatic" in form or "isometric" in form:
            attrs.update({"bar", "prism", "facet", "solid"})
        if "liquid" in form:
            attrs.update({"bar", "liquid", "round_cap"})
        if "stacked" in form:
            attrs.update({"bar", "stacked", "solid"})
        if "capsule" in form:
            attrs.update({"bar", "capsule", "horizontal", "round_cap"})
        if "crossrange" in form or "horizontal_bar" in form:
            attrs.update({"bar", "horizontal"})
        if "vertical_bar" in form or form.endswith("_bar"):
            attrs.update({"bar", "vertical"})
        if "line" in form:
            attrs.add("line")
        if "area" in form:
            attrs.update({"line", "area", "gradient"})
        if "pie" in form or "donut" in form or "ring" in form:
            attrs.add("pie")
        if "donut" in form or "ring" in form:
            attrs.add("ring")
        if "pie3d" in form or "3d" in form:
            attrs.add("3d")
        if "table" in form or "grid" in form:
            attrs.add("table")
        if "scatter" in form:
            attrs.add("scatter")
        if "funnel" in form:
            attrs.add("funnel")
        if "map" in form:
            attrs.add("map")
    return attrs


def visual_attrs_from_text(text: str) -> set[str]:
    lowered = str(text or "").lower()
    lowered = re.sub(r"\b[a-z_]*(?:score|count|caps|edges|bars|hues|area|grid|ring|slanted)=[0-9.]+", " ", lowered)
    attrs: set[str] = set()
    aliases = {
        "cylinder": ["圆柱", "cylinder", "cylindrical", "elliptical top", "ellipse cap", "椭圆顶", "圆形顶盖"],
        "round_cap": ["椭圆", "圆形顶", "elliptical", "round cap", "top cap"],
        "round_base": ["环形光效", "圆环底座", "round base", "base ring", "ellipse base"],
        "prism": ["棱柱", "prism", "prismatic", "isometric"],
        "facet": ["棱面", "斜切", "diamond", "facet", "polygon", "slanted"],
        "gradient": ["渐变", "gradient"],
        "liquid": ["液体", "水波", "liquid", "wavy"],
        "horizontal": ["横向", "horizontal"],
        "vertical": ["纵向", "vertical"],
        "stacked": ["堆叠", "stacked"],
        "line": ["折线", "line"],
        "area": ["面积", "area"],
        "table": ["表格", "列表", "table", "grid"],
        "pie": ["饼", "pie"],
        "ring": ["环形", "donut", "ring"],
    }
    for attr, values in aliases.items():
        if any(value in lowered for value in values):
            attrs.add(attr)
    if {"cylinder", "prism", "liquid"}.isdisjoint(attrs) and "bar" in lowered:
        attrs.add("bar")
    if attrs & {"cylinder", "prism", "liquid", "stacked"}:
        attrs.add("bar")
    return attrs


def aspect_score(record: ComponentRecord, node: Node) -> float:
    aspect = float(node.features.get("aspectRatio", node.bbox.w / max(node.bbox.h, 1.0)))
    description = record.description
    score = 0.0
    if aspect >= 2.2 and ("横向" in description or "延展" in description):
        score += 0.045
    if aspect < 1.2 and ("中心" in description or "均衡" in description or "圆" in description):
        score += 0.045
    return score
