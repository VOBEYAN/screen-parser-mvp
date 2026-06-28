from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from PIL import Image

from .component_library import ComponentLibrary
from .schemas import ComponentRecord


SYSTEM_PROMPT = "你是 ai-schema-view 大屏组件识别助手。你只根据图片判断最匹配的组件，并严格输出 JSON。"
USER_PROMPT = (
    "请从 ai-schema-view 的 95 个组件库中识别图片里的组件，只返回 JSON。"
    "JSON 字段包含 componentId、visualForm、confidence、nearMisses。"
)

CHART_CONTENT_BY_CATEGORY = {
    "Bars": "bar_chart",
    "Lines": "line_chart",
    "Areas": "area_chart",
    "Pies": "pie_chart",
    "Scatters": "scatter_chart",
    "Funnels": "funnel_chart",
    "WordClouds": "wordcloud",
    "FlowChart": "chart",
}

TYPE_BY_CATEGORY = {
    "Tables": "Table",
    "Maps": "Map",
    "Inputs": "Filter",
    "Title": "Title",
    "Texts": "Title",
    "Borders": "Border",
    "Decorates": "Decorate",
    "Three": "Decorate",
    "Biz": "Decorate",
    "Mores": "MetricCard",
}

MAP_COMPONENT_IDS = {"ChinaMap", "MapAmap", "MapBase"}
IMAGE_COMPONENT_IDS = {"AIRobot", "AIShield", "KeySecurity3D", "ThreeEarth01"}


class LocalQwenVLComponentRecognizer:
    """Lazy local Qwen3-VL + LoRA component recognizer."""

    def __init__(
        self,
        model_path: str,
        adapter_path: str,
        library: ComponentLibrary,
        device: str = "auto",
        image_size: int = 224,
        max_new_tokens: int = 96,
    ):
        self.model_path = str(Path(model_path).expanduser())
        self.adapter_path = str(Path(adapter_path).expanduser())
        self.library = library
        self.device_name = device
        self.image_size = image_size
        self.max_new_tokens = max_new_tokens
        self._lock = threading.RLock()
        self._loaded = False
        self._load_error: Optional[str] = None
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._process_vision_info: Any = None
        self._device: Any = None
        self._records_by_lower = {record.key.lower(): record for record in library.records}

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def status(self) -> Dict[str, Any]:
        return {
            "configured": bool(self.model_path and self.adapter_path),
            "loaded": self._loaded,
            "loadError": self._load_error,
            "modelPath": self.model_path,
            "adapterPath": self.adapter_path,
            "device": str(self._device or self.device_name),
            "imageSize": self.image_size,
        }

    def classify(self, crop: Image.Image) -> Dict[str, Any]:
        with self._lock:
            self._ensure_loaded()
            crop = crop.convert("RGB")
            if self.image_size > 0:
                crop = crop.resize((self.image_size, self.image_size))

            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": crop},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = self._process_vision_info(messages)
            inputs = self._processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")
            inputs = {key: value.to(self._device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with self._torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            new_tokens = generated[:, inputs["input_ids"].shape[1] :]
            answer = self._processor.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            return self._normalize_response(answer)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._load_error:
            raise RuntimeError(self._load_error)
        try:
            model_path = Path(self.model_path)
            adapter_path = Path(self.adapter_path)
            if not model_path.exists():
                raise FileNotFoundError(f"local Qwen3-VL model not found: {model_path}")
            if not adapter_path.exists():
                raise FileNotFoundError(f"local Qwen3-VL adapter not found: {adapter_path}")

            import torch
            from peft import PeftModel
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            self._torch = torch
            self._process_vision_info = process_vision_info
            self._device = self._resolve_device(torch)
            dtype = torch.float16 if str(self._device) in {"mps", "cuda"} else torch.float32
            self._processor = AutoProcessor.from_pretrained(str(model_path))
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            self._model = PeftModel.from_pretrained(base_model, str(adapter_path))
            self._model.eval().to(self._device)
            self._loaded = True
        except Exception as exc:
            self._load_error = str(exc)[:500]
            raise

    def _resolve_device(self, torch: Any) -> Any:
        if self.device_name == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(self.device_name)

    def _normalize_response(self, answer: str) -> Dict[str, Any]:
        parsed = parse_json_object(answer) or {}
        component_id = resolve_component_id(str(parsed.get("componentId") or ""), self.library.records)
        record = self.library.by_key.get(component_id) if component_id else None
        visual_form = str(parsed.get("visualForm") or "").strip()
        confidence = numeric_score(parsed.get("confidence"), default=0.78 if component_id else 0.0)
        near_misses = normalize_near_misses(parsed.get("nearMisses"), self.library.records)
        candidate_scores = build_candidate_scores(component_id, near_misses, confidence)
        result = {
            "componentId": component_id,
            "visualForm": visual_form,
            "confidence": confidence,
            "nearMisses": near_misses,
            "candidateScores": candidate_scores,
            "rejectedNearMisses": [
                {"componentId": item, "whyRejected": "local Qwen3-VL LoRA ranked it as a near miss"}
                for item in near_misses
            ],
            "text": str(parsed.get("text") or ""),
            "textEvidence": "",
            "visualEvidence": f"local Qwen3-VL LoRA output: {visual_form}" if visual_form else "local Qwen3-VL LoRA output",
            "reason": "local Qwen3-VL LoRA component recognition",
            "modelSource": "local_qwen3_vl_lora",
            "rawModelOutput": answer[:500],
        }
        if record:
            result["contentType"] = infer_content_type(record, visual_form)
            result["componentType"] = infer_component_type(record, visual_form)
        return result


def parse_json_object(content: str) -> Optional[Dict[str, Any]]:
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


def resolve_component_id(value: str, records: Iterable[ComponentRecord]) -> str:
    normalized = value.strip()
    if not normalized:
        return ""
    by_key = {record.key: record.key for record in records}
    if normalized in by_key:
        return normalized
    by_lower = {record.key.lower(): record.key for record in records}
    return by_lower.get(normalized.lower(), "")


def normalize_near_misses(value: Any, records: Iterable[ComponentRecord]) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        component_id = resolve_component_id(str(item or ""), records)
        if component_id and component_id not in normalized:
            normalized.append(component_id)
    return normalized[:5]


def build_candidate_scores(component_id: str, near_misses: list[str], confidence: float) -> list[Dict[str, Any]]:
    scores = []
    if component_id:
        scores.append(candidate_score(component_id, confidence, "chosen by local Qwen3-VL LoRA"))
    for index, near_miss in enumerate(near_misses):
        score = max(0.08, confidence - 0.24 - index * 0.08)
        scores.append(candidate_score(near_miss, score, "near miss from local Qwen3-VL LoRA"))
    return scores


def candidate_score(component_id: str, score: float, evidence: str) -> Dict[str, Any]:
    score = max(0.0, min(1.0, score))
    return {
        "componentId": component_id,
        "visualMatchScore": round(score, 4),
        "shapeGeometry": round(score, 4),
        "perspective3D": round(score, 4),
        "colorStyle": round(score, 4),
        "baseAndCap": round(score, 4),
        "textDataFit": round(score, 4),
        "layoutFit": round(score, 4),
        "profileFit": round(score, 4),
        "negativeMismatch": round(score, 4),
        "evidence": evidence,
    }


def numeric_score(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return max(0.0, min(1.0, score))


def infer_content_type(record: ComponentRecord, visual_form: str) -> str:
    if record.key in MAP_COMPONENT_IDS:
        return "map"
    if record.key in IMAGE_COMPONENT_IDS:
        return "image"
    if record.category in CHART_CONTENT_BY_CATEGORY:
        return CHART_CONTENT_BY_CATEGORY[record.category]
    if record.category == "Tables":
        return "table"
    if record.category == "Maps":
        return "map"
    if record.category == "Inputs":
        return "filter"
    if record.category in {"Title", "Texts"}:
        return "title"
    if record.category == "Borders":
        return "border"
    if record.category == "Decorates":
        return "decorate"
    if record.category in {"Biz", "Three"}:
        return "image"
    form = visual_form.lower()
    if "bar" in form:
        return "bar_chart"
    if "line" in form:
        return "line_chart"
    if "pie" in form or "ring" in form:
        return "pie_chart"
    return "chart" if infer_component_type(record, visual_form) == "Chart" else "decorate"


def infer_component_type(record: ComponentRecord, visual_form: str) -> str:
    if record.key in MAP_COMPONENT_IDS:
        return "Map"
    if record.key in IMAGE_COMPONENT_IDS:
        return "Image"
    if record.category in CHART_CONTENT_BY_CATEGORY:
        return "Chart"
    if record.category in TYPE_BY_CATEGORY:
        return TYPE_BY_CATEGORY[record.category]
    form = visual_form.lower()
    if "border" in form or "frame" in form:
        return "Border"
    if any(token in form for token in ["bar", "line", "pie", "chart"]):
        return "Chart"
    return "Decorate"
