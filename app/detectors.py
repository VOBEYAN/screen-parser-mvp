from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .geometry import iou, overlap_ratio
from .schemas import BBox, Detection


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
        return detections


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


def build_detector(model_path: Optional[str] = None) -> Detector:
    if model_path and Path(model_path).exists():
        return HybridDetector(YoloDetector(model_path))
    return HeuristicDetector()


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
    if "table" in value:
        return "Table"
    if "map" in value:
        return "Map"
    if "title" in value or "text" in value:
        return "Title"
    if "border" in value:
        return "Border"
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
    return class_name if class_name in {"Panel", "Title", "Chart", "Table", "Map", "MetricCard", "Border", "Decorate", "Filter"} else "Decorate"


class HybridDetector(Detector):
    def __init__(self, yolo_detector: YoloDetector, fallback_detector: Optional[Detector] = None, max_supplemental: int = 18):
        self.yolo_detector = yolo_detector
        self.fallback_detector = fallback_detector or HeuristicDetector()
        self.max_supplemental = max_supplemental

    def detect(self, image_path: str) -> List[Detection]:
        yolo_detections = self.yolo_detector.detect(image_path)
        if len(yolo_detections) >= 8:
            detections = self._dedupe_detections(yolo_detections)
            fallback_detections = self.fallback_detector.detect(image_path)
            detections = self._add_supplemental_fallbacks(image_path, detections, fallback_detections)
            return self._renumber(detections[:100])

        fallback_detections = self.fallback_detector.detect(image_path)
        merged: List[Detection] = []
        for detection in sorted(yolo_detections + fallback_detections, key=lambda item: (item.source == "yolo", item.confidence), reverse=True):
            if any(iou(detection.bbox, existing.bbox) > 0.78 for existing in merged):
                continue
            merged.append(detection)
        return self._renumber(merged[:100])

    def _dedupe_detections(self, detections: List[Detection]) -> List[Detection]:
        kept: List[Detection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            duplicate = False
            for existing in kept:
                if iou(detection.bbox, existing.bbox) > 0.78:
                    duplicate = True
                    break
                if detection.class_name == existing.class_name and overlap_ratio(detection.bbox, existing.bbox) > 0.68:
                    duplicate = True
                    break
            if duplicate:
                continue
            kept.append(detection)
        return kept

    def _add_supplemental_fallbacks(
        self,
        image_path: str,
        detections: List[Detection],
        fallback_detections: List[Detection],
    ) -> List[Detection]:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            return detections

        height, width = image.shape[:2]
        image_area = float(width * height)
        merged = list(detections)
        added = 0
        for candidate in sorted(fallback_detections, key=lambda item: item.confidence, reverse=True):
            if added >= self.max_supplemental:
                break
            if not self._is_useful_supplement(candidate, merged, image_area):
                continue
            candidate.source = "opencv_supplement"
            candidate.confidence = min(candidate.confidence, 0.49)
            merged.append(candidate)
            added += 1
        return self._dedupe_detections(merged)

    def _is_useful_supplement(self, candidate: Detection, existing: List[Detection], image_area: float) -> bool:
        if candidate.class_name not in {"Panel", "Chart", "Table", "MetricCard", "Title"}:
            return False

        area_ratio = candidate.bbox.area / max(image_area, 1.0)
        aspect = candidate.bbox.w / max(candidate.bbox.h, 1.0)
        if candidate.class_name == "Panel" and not (0.012 <= area_ratio <= 0.45):
            return False
        if candidate.class_name in {"Chart", "Table", "MetricCard"} and not (0.0015 <= area_ratio <= 0.18):
            return False
        if candidate.class_name == "Title" and not (0.00045 <= area_ratio <= 0.035 and 2.8 <= aspect <= 22.0 and candidate.bbox.h >= 14):
            return False

        for item in existing:
            same_type = candidate.class_name == item.class_name
            same_family = candidate.class_name in {"Chart", "Table", "MetricCard"} and item.class_name in {"Chart", "Table", "MetricCard", "Map"}
            if iou(candidate.bbox, item.bbox) > 0.72:
                return False
            if same_type and overlap_ratio(candidate.bbox, item.bbox) > 0.52:
                return False
            if same_family and overlap_ratio(candidate.bbox, item.bbox) > 0.72:
                return False
            if candidate.class_name == "Panel" and item.class_name in {"Panel", "Border"} and overlap_ratio(candidate.bbox, item.bbox) > 0.60:
                return False
        return True

    def _renumber(self, detections: List[Detection]) -> List[Detection]:
        for index, detection in enumerate(detections):
            detection.detection_id = f"det_{index:04d}"
        return detections
