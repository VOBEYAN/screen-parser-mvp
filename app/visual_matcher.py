from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from .schemas import BBox


REFERENCE_FEATURES_FILE = "reference_features.json"


@dataclass
class VisualReference:
    component_id: str
    title: str
    category: str
    image_path: str
    features: Dict[str, object]


class VisualReferenceLibrary:
    def __init__(self, references: Iterable[VisualReference]):
        self.references = list(references)
        self.by_component_id: Dict[str, VisualReference] = {
            item.component_id: item for item in self.references
        }

    @classmethod
    def from_path(cls, reference_path: Optional[str]) -> "VisualReferenceLibrary":
        if not reference_path:
            return cls([])

        path = Path(reference_path)
        json_path = path / REFERENCE_FEATURES_FILE if path.is_dir() else path
        if not json_path.exists():
            return cls([])

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        components = payload.get("components", payload if isinstance(payload, list) else [])
        references: List[VisualReference] = []
        for item in components:
            component_id = item.get("componentId") or item.get("component_id")
            features = item.get("features") or item.get("feature") or {}
            if not component_id or not features:
                continue
            references.append(
                VisualReference(
                    component_id=str(component_id),
                    title=str(item.get("title", "")),
                    category=str(item.get("category", "")),
                    image_path=str(item.get("imagePath", "")),
                    features=features,
                )
            )
        return cls(references)

    @property
    def enabled(self) -> bool:
        return bool(self.references)

    def score(self, component_id: str, crop_features: Dict[str, object]) -> Optional[float]:
        reference = self.by_component_id.get(component_id)
        if not reference:
            return None
        return visual_similarity(crop_features, reference.features)


def load_bgr_image(image_path: str) -> np.ndarray:
    path = Path(image_path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return ensure_bgr(image)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        bgr = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        background = np.array([18.0, 24.0, 34.0], dtype=np.float32)
        return np.clip(bgr * alpha + background * (1.0 - alpha), 0, 255).astype(np.uint8)
    if image.shape[2] == 3:
        return image
    raise ValueError("Unsupported image shape for visual matching")


def extract_image_features_from_path(image_path: str) -> Dict[str, object]:
    return extract_image_features(load_bgr_image(image_path))


def extract_crop_features(image: np.ndarray, bbox: BBox) -> Dict[str, object]:
    height, width = image.shape[:2]
    x1 = max(0, int(round(bbox.x)))
    y1 = max(0, int(round(bbox.y)))
    x2 = min(width, int(round(bbox.right)))
    y2 = min(height, int(round(bbox.bottom)))
    if x2 <= x1 or y2 <= y1:
        return {}
    return extract_image_features(image[y1:y2, x1:x2])


def extract_image_features(image: np.ndarray) -> Dict[str, object]:
    bgr = ensure_bgr(image)
    if bgr.size == 0:
        return {}

    height, width = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    edges = cv2.Canny(gray, 45, 135)
    edge_density = float(np.count_nonzero(edges)) / float(max(edges.size, 1))

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, width // 18), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, height // 18)))
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_density = float(np.count_nonzero(horizontal)) / float(max(edges.size, 1))
    vertical_density = float(np.count_nonzero(vertical)) / float(max(edges.size, 1))

    content_mask = build_content_mask(gray, hsv, edges)
    color_hist = hsv_histogram(hsv, content_mask)
    dominant_hue = median_hue(hsv, content_mask)

    edge_profile = normalized_vector(
        cv2.resize(edges, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    )
    gray_profile = normalized_vector(
        cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    )
    hash_bits = perceptual_hash(gray)

    return {
        "width": int(width),
        "height": int(height),
        "aspectRatio": round(float(width) / float(max(height, 1)), 5),
        "brightness": round(float(np.mean(gray)), 5),
        "contrast": round(float(np.std(gray)), 5),
        "edgeDensity": round(edge_density, 6),
        "horizontalDensity": round(horizontal_density, 6),
        "verticalDensity": round(vertical_density, 6),
        "dominantColor": hue_to_label(dominant_hue),
        "dominantHue": round(dominant_hue, 4),
        "colorHist": round_list(color_hist),
        "edgeProfile": round_list(edge_profile),
        "grayProfile": round_list(gray_profile),
        "hash": hash_bits,
    }


def build_content_mask(gray: np.ndarray, hsv: np.ndarray, edges: np.ndarray) -> np.ndarray:
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    mask = ((value > 28) & (saturation > 22)) | (edges > 0) | (gray > 50)
    if np.count_nonzero(mask) < max(16, int(mask.size * 0.01)):
        return np.ones(gray.shape, dtype=np.uint8)
    return mask.astype(np.uint8)


def hsv_histogram(hsv: np.ndarray, mask: np.ndarray) -> List[float]:
    hist_h = cv2.calcHist([hsv], [0], mask, [24], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], mask, [8], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], mask, [8], [0, 256])
    hist = np.concatenate([hist_h.reshape(-1), hist_s.reshape(-1), hist_v.reshape(-1)]).astype(np.float32)
    return normalized_vector(hist)


def median_hue(hsv: np.ndarray, mask: np.ndarray) -> float:
    selected = hsv[:, :, 0][mask.astype(bool)]
    if selected.size == 0:
        return -1.0
    return float(np.median(selected))


def perceptual_hash(gray: np.ndarray) -> List[int]:
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low = dct[:8, :8].reshape(-1)
    values = low[1:]
    median = float(np.median(values))
    return [1 if float(value) > median else 0 for value in values]


def normalized_vector(values: np.ndarray) -> List[float]:
    array = values.astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-8:
        return [0.0 for _ in array]
    return (array / norm).tolist()


def round_list(values: Iterable[float], digits: int = 6) -> List[float]:
    return [round(float(value), digits) for value in values]


def visual_similarity(a: Dict[str, object], b: Dict[str, object]) -> float:
    if not a or not b:
        return 0.0

    edge_score = cosine_score(a.get("edgeProfile"), b.get("edgeProfile"))
    gray_score = cosine_score(a.get("grayProfile"), b.get("grayProfile"))
    hash_score = hamming_score(a.get("hash"), b.get("hash"))
    aspect_score = aspect_similarity(
        float(a.get("aspectRatio", 1.0)),
        float(b.get("aspectRatio", 1.0)),
    )
    stats_score = stats_similarity(a, b)

    score = (
        0.38 * edge_score
        + 0.22 * gray_score
        + 0.18 * aspect_score
        + 0.14 * stats_score
        + 0.08 * hash_score
    )
    return float(max(0.0, min(1.0, score)))


def cosine_score(a: object, b: object) -> float:
    if not isinstance(a, list) or not isinstance(b, list) or not a or not b:
        return 0.0
    length = min(len(a), len(b))
    left = np.array(a[:length], dtype=np.float32)
    right = np.array(b[:length], dtype=np.float32)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-8:
        return 0.0
    return float(max(0.0, min(1.0, float(np.dot(left, right)) / denom)))


def hamming_score(a: object, b: object) -> float:
    if not isinstance(a, list) or not isinstance(b, list) or not a or not b:
        return 0.0
    length = min(len(a), len(b))
    distance = sum(1 for index in range(length) if int(a[index]) != int(b[index]))
    return 1.0 - float(distance) / float(max(length, 1))


def aspect_similarity(a: float, b: float) -> float:
    a = max(0.05, a)
    b = max(0.05, b)
    distance = abs(np.log(a / b))
    return float(max(0.0, 1.0 - min(distance, np.log(6.0)) / np.log(6.0)))


def stats_similarity(a: Dict[str, object], b: Dict[str, object]) -> float:
    brightness = scalar_similarity(a, b, "brightness", 255.0)
    contrast = scalar_similarity(a, b, "contrast", 128.0)
    edge = scalar_similarity(a, b, "edgeDensity", 0.35)
    horizontal = scalar_similarity(a, b, "horizontalDensity", 0.18)
    vertical = scalar_similarity(a, b, "verticalDensity", 0.18)
    return 0.05 * brightness + 0.17 * contrast + 0.34 * edge + 0.24 * horizontal + 0.2 * vertical


def scalar_similarity(a: Dict[str, object], b: Dict[str, object], key: str, scale: float) -> float:
    left = float(a.get(key, 0.0))
    right = float(b.get(key, 0.0))
    return float(max(0.0, 1.0 - min(abs(left - right) / max(scale, 1e-6), 1.0)))


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
