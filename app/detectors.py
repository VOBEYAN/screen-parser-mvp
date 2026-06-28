from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .geometry import iou
from .schemas import BBox, Detection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_MODELS = (
    PROJECT_ROOT / "models" / "yolo_screen_structure_rich_v5_design1_hardcase_local.pt",
    PROJECT_ROOT / "models" / "yolo_screen_structure_rich_v5_hardcase_local.pt",
    PROJECT_ROOT / "models" / "yolo_screen_structure_rich_v5.pt",
    PROJECT_ROOT / "models" / "yolo_screen_structure_rich_v5_quick_local.pt",
    PROJECT_ROOT / "models" / "yolo_screen_structure_chart_hard_v3.pt",
    PROJECT_ROOT / "models" / "yolo_screen_structure_title_hard_v2.pt",
    PROJECT_ROOT / "models" / "yolo_screen_structure_local_v1.pt",
)


class Detector:
    def detect(self, image_path: str) -> List[Detection]:
        raise NotImplementedError


class YoloDetector(Detector):
    def __init__(self, model_path: str, conf_threshold: float = 0.08):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.model_path = model_path
        self.conf_threshold = conf_threshold

    def detect(self, image_path: str) -> List[Detection]:
        results = self.model(image_path, verbose=False, conf=self.conf_threshold)
        detections: List[Detection] = []
        index = 0
        for result in results:
            names = result.names or {}
            for box in result.boxes:
                xyxy = box.xyxy.cpu().numpy()[0]
                cls_id = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                raw_class_name = str(names.get(cls_id, cls_id))
                node_type = normalize_detection_type(raw_class_name)
                detections.append(
                    Detection(
                        detection_id=f"det_{index:04d}",
                        bbox=BBox(x1, y1, x2 - x1, y2 - y1),
                        class_name=node_type,
                        component_id=raw_class_name if raw_class_name != node_type else None,
                        confidence=conf,
                        source="yolo",
                        features={"rawClassName": raw_class_name},
                    )
                )
                index += 1
        detections = self._dedupe(detections)
        detections.extend(detect_luminous_bar_charts(image_path, detections, start_index=len(detections)))
        return self._dedupe(detections)

    def _dedupe(self, detections: List[Detection]) -> List[Detection]:
        ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
        kept: List[Detection] = []
        for detection in ordered:
            duplicate = False
            for other in kept:
                if detection.class_name == other.class_name and iou(detection.bbox, other.bbox) > 0.86:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(detection)
        kept.sort(key=lambda item: item.detection_id)
        return kept


class HeuristicDetector(Detector):
    def __init__(self, max_detections: int = 90):
        self.max_detections = max_detections

    def detect(self, image_path: str) -> List[Detection]:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        _, bright = cv2.threshold(gray, 28, 255, cv2.THRESH_BINARY)
        masks = [
            cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)),
            cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)),
        ]

        raw_boxes: List[Tuple[BBox, float]] = []
        image_area = float(width * height)
        min_area = max(900.0, image_area * 0.00035)
        max_area = image_area * 0.92

        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w < 14 or h < 12:
                    continue
                area = float(w * h)
                if area < min_area or area > max_area:
                    continue
                if w > width * 0.98 and h > height * 0.98:
                    continue
                bbox = BBox(float(x), float(y), float(w), float(h))
                fill = min(1.0, cv2.contourArea(contour) / max(area, 1.0))
                confidence = 0.32 + min(0.35, area / image_area * 3.0) + min(0.25, fill * 0.28)
                raw_boxes.append((bbox, float(min(0.92, confidence))))

        boxes = self._dedupe(raw_boxes)
        detections: List[Detection] = []
        for index, (bbox, confidence) in enumerate(boxes[: self.max_detections]):
            features = extract_visual_features(image, bbox)
            class_name = infer_component_type(bbox, width, height, features)
            detections.append(
                Detection(
                    detection_id=f"det_{index:04d}",
                    bbox=bbox,
                    class_name=class_name,
                    confidence=round(confidence, 4),
                    source="opencv",
                    features=features,
                )
            )
        return detections

    def _dedupe(self, boxes: List[Tuple[BBox, float]]) -> List[Tuple[BBox, float]]:
        boxes = sorted(boxes, key=lambda item: (item[1], item[0].area), reverse=True)
        kept: List[Tuple[BBox, float]] = []
        for bbox, confidence in boxes:
            duplicate = False
            for other, _ in kept:
                same_shape = abs(bbox.area - other.area) / max(bbox.area, other.area, 1.0) < 0.08
                if iou(bbox, other) > 0.82 and same_shape:
                    duplicate = True
                    break
            if not duplicate:
                kept.append((bbox, confidence))
        return kept


def build_detector(model_path: Optional[str] = None, conf_threshold: Optional[float] = None) -> Detector:
    resolved_model = resolve_yolo_model(model_path)
    if resolved_model:
        return YoloDetector(str(resolved_model), conf_threshold=0.08 if conf_threshold is None else conf_threshold)
    return HeuristicDetector()


def resolve_yolo_model(model_path: Optional[str] = None) -> Optional[Path]:
    if model_path:
        path = Path(model_path)
        candidates = [path] if path.is_absolute() else [Path.cwd() / path, PROJECT_ROOT / path]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    for candidate in DEFAULT_YOLO_MODELS:
        if candidate.exists():
            return candidate
    return None


def detect_luminous_bar_charts(image_path: str, existing: List[Detection], start_index: int = 0) -> List[Detection]:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return []

    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Cyan/green luminous dashboard bars are easy to miss as objects, but have a strong color signature.
    mask = cv2.inRange(hsv, np.array([70, 60, 90], dtype=np.uint8), np.array([105, 255, 255], dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 13), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bar_boxes: List[BBox] = []
    image_area = float(width * height)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 5 or h < 24:
            continue
        if h / max(w, 1) < 1.35:
            continue
        if w > width * 0.08 or h > height * 0.38:
            continue
        area_ratio = (w * h) / image_area
        if area_ratio < 0.00003 or area_ratio > 0.018:
            continue
        bar_boxes.append(BBox(float(x), float(y), float(w), float(h)))

    groups = group_vertical_bars(bar_boxes, width, height)
    detections: List[Detection] = []
    next_index = start_index
    for group in groups:
        bbox = chart_bbox_from_bars(group, width, height)
        if bbox.w < 90 or bbox.h < 70:
            continue
        if bbox.area / image_area > 0.16:
            continue
        if overlaps_existing_chart(bbox, existing):
            continue
        confidence = min(0.72, 0.38 + 0.05 * len(group))
        detections.append(
            Detection(
                detection_id=f"det_bar_{next_index:04d}",
                bbox=bbox,
                class_name="Chart",
                confidence=round(confidence, 4),
                source="bar_chart_vision",
                features={
                    "rawClassName": "Chart",
                    "detector": "luminous_bar_chart",
                    "barCount": len(group),
                },
            )
        )
        next_index += 1
    return detections


def group_vertical_bars(bar_boxes: List[BBox], image_width: int, image_height: int) -> List[List[BBox]]:
    groups: List[List[BBox]] = []
    for box in sorted(bar_boxes, key=lambda item: (item.y + item.h * 0.5, item.x)):
        cx, cy = box.center
        matched = False
        for group in groups:
            xs = [member.x for member in group] + [member.right for member in group]
            ys = [member.y for member in group] + [member.bottom for member in group]
            group_box = BBox(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            _, group_cy = group_box.center
            x_near = box.x <= group_box.right + image_width * 0.08 and box.right >= group_box.x - image_width * 0.08
            y_near = abs(cy - group_cy) <= max(image_height * 0.09, group_box.h * 0.75, box.h * 0.9)
            if x_near and y_near:
                group.append(box)
                matched = True
                break
        if not matched:
            groups.append([box])

    filtered = []
    for group in groups:
        if len(group) < 3:
            continue
        xs = [member.x for member in group] + [member.right for member in group]
        ys = [member.y for member in group] + [member.bottom for member in group]
        group_w = max(xs) - min(xs)
        group_h = max(ys) - min(ys)
        if group_w >= image_width * 0.08 and group_h >= image_height * 0.06:
            filtered.append(group)
    return filtered


def chart_bbox_from_bars(group: List[BBox], image_width: int, image_height: int) -> BBox:
    xs = [member.x for member in group] + [member.right for member in group]
    ys = [member.y for member in group] + [member.bottom for member in group]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pad_x = max(24.0, (x2 - x1) * 0.18)
    top_pad = max(28.0, (y2 - y1) * 0.30)
    bottom_pad = max(24.0, (y2 - y1) * 0.18)
    x1 = max(0.0, x1 - pad_x)
    x2 = min(float(image_width), x2 + pad_x)
    y1 = max(0.0, y1 - top_pad)
    y2 = min(float(image_height), y2 + bottom_pad)
    return BBox(x1, y1, x2 - x1, y2 - y1)


def overlaps_existing_chart(bbox: BBox, existing: List[Detection]) -> bool:
    for detection in existing:
        if detection.class_name == "Chart" and iou(bbox, detection.bbox) > 0.38:
            return True
    return False


def extract_visual_features(image: np.ndarray, bbox: BBox) -> dict:
    height, width = image.shape[:2]
    x1 = max(0, int(bbox.x))
    y1 = max(0, int(bbox.y))
    x2 = min(width, int(bbox.right))
    y2 = min(height, int(bbox.bottom))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return {"dominantColor": "unknown", "edgeDensity": 0.0, "lineDensity": 0.0}

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 18))
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
    line_density = float(np.count_nonzero(horizontal) + np.count_nonzero(vertical)) / float(edges.size)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat_mask = hsv[:, :, 1] > 35
    if np.count_nonzero(sat_mask) > 0:
        hue = float(np.median(hsv[:, :, 0][sat_mask]))
    else:
        hue = -1.0

    return {
        "dominantColor": hue_to_label(hue),
        "edgeDensity": round(edge_density, 4),
        "lineDensity": round(line_density, 4),
        "areaRatio": round(bbox.area / float(width * height), 5),
        "aspectRatio": round(bbox.w / max(bbox.h, 1.0), 4),
    }


def hue_to_label(hue: float) -> str:
    if hue < 0:
        return "gray"
    if 90 <= hue <= 135:
        return "blue"
    if 70 <= hue < 90:
        return "cyan"
    if 35 <= hue < 70:
        return "green"
    if 10 <= hue < 35:
        return "yellow"
    if hue < 10 or hue > 165:
        return "red"
    if 135 < hue <= 165:
        return "purple"
    return "blue"


def infer_component_type(bbox: BBox, image_width: int, image_height: int, features: dict) -> str:
    area_ratio = bbox.area / float(image_width * image_height)
    aspect = bbox.w / max(bbox.h, 1.0)
    edge_density = float(features.get("edgeDensity", 0.0))
    line_density = float(features.get("lineDensity", 0.0))

    if bbox.h <= image_height * 0.075 and aspect > 3.0:
        return "Title"
    if area_ratio > 0.055 and aspect > 0.65:
        return "Panel"
    if line_density > 0.018 and aspect > 1.2:
        return "Table"
    if area_ratio > 0.025 and edge_density < 0.035:
        return "MetricCard"
    if area_ratio > 0.015:
        return "Chart"
    if aspect > 4.0:
        return "Title"
    return "Decorate"


def normalize_detection_type(class_name: str) -> str:
    value = str(class_name).lower()
    valid_types = {"Region", "Panel", "Content", "Title", "Chart", "Table", "Map", "MetricCard", "Border", "Decorate", "Filter", "Image"}
    if class_name in valid_types:
        return class_name
    if "table" in value:
        return "Table"
    if "map" in value:
        return "Map"
    if any(token in value for token in ["image", "img", "photo", "picture", "shield", "robot", "earth", "3d", "visual"]):
        return "Image"
    if "title" in value or "text" in value:
        return "Title"
    if "border" in value:
        return "Border"
    if "content" in value:
        return "Content"
    if "panel" in value or "region" in value:
        return "Panel"
    if "metric" in value or "number" in value or "card" in value:
        return "MetricCard"
    if "filter" in value or "input" in value or "select" in value:
        return "Filter"
    if (
        "chart" in value
        or "bar" in value
        or "line" in value
        or "pie" in value
        or "radar" in value
        or "sankey" in value
        or "scatter" in value
        or "area" in value
        or "funnel" in value
        or "heatmap" in value
        or "wordcloud" in value
        or "graph" in value
    ):
        return "Chart"
    if "clock" in value or "decorate" in value or "circle" in value or "pipeline" in value:
        return "Decorate"
    return class_name if class_name in valid_types else "Decorate"
