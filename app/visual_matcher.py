from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    bgr = trim_visual_content(bgr)
    bgr = isolate_data_region(bgr)

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
    structural = extract_structural_features(gray, hsv, edges, content_mask)

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
        "structural": structural,
    }


def trim_visual_content(image: np.ndarray) -> np.ndarray:
    bgr = ensure_bgr(image)
    if bgr.size == 0:
        return bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 35, 120)
    mask = build_content_mask(gray, hsv, edges).astype(bool)
    if np.count_nonzero(mask) < max(32, int(mask.size * 0.015)):
        return bgr
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    pad_x = max(4, int(w * 0.025))
    pad_y = max(4, int(h * 0.025))
    x1 = max(0, int(xs.min()) - pad_x)
    y1 = max(0, int(ys.min()) - pad_y)
    x2 = min(w, int(xs.max()) + pad_x + 1)
    y2 = min(h, int(ys.max()) + pad_y + 1)
    if (x2 - x1) < 12 or (y2 - y1) < 12:
        return bgr
    if (x2 - x1) * (y2 - y1) > bgr.shape[0] * bgr.shape[1] * 0.96:
        return bgr
    return bgr[y1:y2, x1:x2]


def isolate_data_region(image: np.ndarray) -> np.ndarray:
    bgr = ensure_bgr(image)
    if bgr.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    if h < 48 or w < 48:
        return bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 35, 120)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]

    data_mask = ((value > 46) & (saturation > 34)).astype(np.uint8)
    data_mask = cv2.morphologyEx(data_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    data_mask = cv2.morphologyEx(data_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Drop the outer frame and title strip when a detected node is a whole
    # panel. Component-library previews are mostly the inner chart/table body,
    # so matching against the panel chrome makes visually similar components
    # collapse into the same dark rectangle.
    border_x = max(2, int(w * 0.035))
    border_y = max(2, int(h * 0.035))
    data_mask[:border_y, :] = 0
    data_mask[h - border_y :, :] = 0
    data_mask[:, :border_x] = 0
    data_mask[:, w - border_x :] = 0
    if h >= 90:
        data_mask[: int(h * 0.12), :] = 0

    if np.count_nonzero(data_mask) < max(24, int(h * w * 0.004)):
        edge_mask = (edges > 0).astype(np.uint8)
        edge_mask[:border_y, :] = 0
        edge_mask[h - border_y :, :] = 0
        edge_mask[:, :border_x] = 0
        edge_mask[:, w - border_x :] = 0
        if h >= 90:
            edge_mask[: int(h * 0.10), :] = 0
        data_mask = edge_mask

    if np.count_nonzero(data_mask) < max(24, int(h * w * 0.003)):
        return bgr

    ys, xs = np.where(data_mask > 0)
    if xs.size == 0 or ys.size == 0:
        return bgr
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad_x = max(5, int(w * 0.055))
    pad_y = max(5, int(h * 0.07))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w < 24 or crop_h < 24:
        return bgr
    if crop_w * crop_h > h * w * 0.94:
        return bgr
    return bgr[y1:y2, x1:x2]


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
    structure_score = structural_similarity(
        dict(a.get("structural") or {}),
        dict(b.get("structural") or {}),
    )
    compatibility = structural_compatibility(
        dict(a.get("structural") or {}),
        dict(b.get("structural") or {}),
    )

    score = (
        0.24 * edge_score
        + 0.12 * gray_score
        + 0.14 * aspect_score
        + 0.10 * stats_score
        + 0.05 * hash_score
        + 0.35 * structure_score
    )
    score *= compatibility
    return float(max(0.0, min(1.0, score)))


def extract_structural_features(
    gray: np.ndarray,
    hsv: np.ndarray,
    edges: np.ndarray,
    content_mask: np.ndarray,
) -> Dict[str, object]:
    height, width = gray.shape[:2]
    area = float(max(1, width * height))
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    hue = hsv[:, :, 0]

    color_mask = ((value > 48) & (saturation > 35)).astype(np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bright_mask = ((value > 95) & (saturation > 55)).astype(np.uint8)

    contour_bars = colored_tall_contours(color_mask, hsv)
    bar_segments = vertical_bar_segments(color_mask, hsv)
    bar_boxes = segment_boxes(bar_segments) + segment_boxes(contour_bars)
    ellipse_caps = ellipse_cap_count(edges, width, height)
    colored_edges = cv2.bitwise_and(edges, edges, mask=color_mask)
    sloped_segments = sloped_line_segments(colored_edges, hsv, width, height)
    colored_slanted_count = len(sloped_segments)
    bright_series = bright_line_series_groups(hsv)
    table_score = table_grid_score(edges, width, height)
    ring_score = max(donut_ring_score(gray, color_mask), local_donut_ring_score(color_mask))
    circle_score = circularity_score(color_mask, width, height)
    red_fill = color_area_ratio(hsv, color_mask, "red")
    color_count = dominant_hue_count(hsv, color_mask)
    bar_area_ratio = sum(float(item.get("area") or 0) for item in bar_segments) / area
    bright_ratio = float(np.count_nonzero(bright_mask)) / area
    content_ratio = float(np.count_nonzero(content_mask)) / area
    line_area = line_area_fill_score(hsv, color_mask, sloped_segments) if len(sloped_segments) >= 2 else 0.0
    bar_line = line_bar_combo_score(color_mask, hsv, sloped_segments) if len(sloped_segments) >= 2 else 0.0
    cylinder_score = cylinder_bar_score(gray, hsv, edges, bar_boxes, ellipse_caps, colored_slanted_count)
    prism_score = prism_bar_score(edges, bar_boxes, color_count, ellipse_caps, colored_slanted_count)

    tall_bar_count = max(len(bar_segments), len(contour_bars))
    sloped_count = len(sloped_segments)
    series_count = max(0, min(5, (len(bright_series) if sloped_segments else 0) or len({item[5] for item in sloped_segments if item[5]})))
    aspect = width / float(max(height, 1))
    prism_like = prism_score >= 0.5 and (tall_bar_count >= 2 or (aspect >= 1.7 and bar_area_ratio >= 0.025))
    cylinder_like = cylinder_score >= 0.55 and (tall_bar_count >= 2 or bar_area_ratio >= 0.08)
    pie_like = circle_score >= 0.18 and tall_bar_count < 3 and red_fill < 0.18 and aspect <= 2.25

    forms: List[str] = []
    if table_score >= 0.28:
        forms.append("table_grid")
    if ring_score >= 0.13 and pie_like:
        forms.append("donut_pie")
    elif pie_like:
        forms.append("pie")
    if tall_bar_count >= 3 or prism_like or cylinder_like:
        if cylinder_like and cylinder_score >= prism_score:
            forms.append("cylinder_vertical_bar")
        elif prism_like:
            forms.append("isometric_prism_bar")
        elif cylinder_like:
            forms.append("cylinder_vertical_bar")
        elif tube_like_bar_score(gray, hsv, bar_boxes) >= 0.42:
            forms.append("liquid_vertical_bar")
        else:
            forms.append("vertical_bar")
    if not pie_like and ((sloped_count >= 2 and tall_bar_count < 3) or (bar_line >= 0.32 and sloped_count >= 6)):
        if bar_line >= 0.32 and sloped_count >= 6:
            forms.append("vertical_bar_line_overlay")
        elif line_area >= 0.08:
            forms.append("line_gradient_area")
        else:
            forms.append("line_chart")

    primary = choose_primary_form(forms, {
        "table_grid": table_score,
        "donut_pie": ring_score,
        "pie": circle_score,
        "isometric_prism_bar": max(prism_score, min(1.0, 0.18 * tall_bar_count + 0.08 * color_count + 0.06 * colored_slanted_count)),
        "cylinder_vertical_bar": max(cylinder_score, min(1.0, 0.18 * tall_bar_count + 0.1 * ellipse_caps)),
        "liquid_vertical_bar": min(1.0, 0.18 * tall_bar_count + tube_like_bar_score(gray, hsv, bar_boxes)),
        "vertical_bar": min(1.0, 0.18 * tall_bar_count + bar_area_ratio * 8.0),
        "vertical_bar_line_overlay": bar_line,
        "line_gradient_area": max(line_area, 0.12 * sloped_count),
        "line_chart": min(1.0, 0.12 * sloped_count + 0.12 * series_count),
    })

    vector = [
        min(1.0, tall_bar_count / 10.0),
        min(1.0, ellipse_caps / 10.0),
        min(1.0, sloped_count / 18.0),
        min(1.0, series_count / 4.0),
        min(1.0, color_count / 8.0),
        min(1.0, table_score),
        min(1.0, ring_score * 3.0),
        min(1.0, circle_score * 2.0),
        min(1.0, line_area * 4.0),
        min(1.0, bar_line * 3.0),
        min(1.0, bar_area_ratio * 10.0),
        min(1.0, bright_ratio * 20.0),
        min(1.0, red_fill * 4.0),
        min(1.0, content_ratio * 3.0),
    ]

    return {
        "primaryForm": primary,
        "forms": forms,
        "vector": round_list(vector),
        "tallBarCount": int(tall_bar_count),
        "ellipseCapCount": int(ellipse_caps),
        "slopedSegmentCount": int(sloped_count),
        "coloredSlantedLineCount": int(colored_slanted_count),
        "lineSeriesCount": int(series_count),
        "colorCount": int(color_count),
        "cylinderScore": round(float(cylinder_score), 5),
        "prismScore": round(float(prism_score), 5),
        "tableGridScore": round(float(table_score), 5),
        "ringScore": round(float(ring_score), 5),
        "circleScore": round(float(circle_score), 5),
        "lineAreaFillScore": round(float(line_area), 5),
        "lineBarComboScore": round(float(bar_line), 5),
        "barAreaRatio": round(float(bar_area_ratio), 5),
    }


def structural_similarity(a: Dict[str, object], b: Dict[str, object]) -> float:
    if not a or not b:
        return 0.45
    vector_score = cosine_score(a.get("vector"), b.get("vector"))
    primary_a = str(a.get("primaryForm") or "")
    primary_b = str(b.get("primaryForm") or "")
    forms_a = set(str(item) for item in (a.get("forms") or []) if item)
    forms_b = set(str(item) for item in (b.get("forms") or []) if item)
    form_score = 0.0
    if primary_a and primary_b and primary_a == primary_b:
        form_score = 1.0
    elif forms_a and forms_b and forms_a.intersection(forms_b):
        form_score = 0.78
    elif same_form_family(primary_a, primary_b):
        form_score = 0.52
    metric_score = structural_metric_similarity(a, b)
    return float(max(0.0, min(1.0, 0.42 * vector_score + 0.38 * form_score + 0.20 * metric_score)))


def structural_compatibility(a: Dict[str, object], b: Dict[str, object]) -> float:
    primary_a = str(a.get("primaryForm") or "")
    primary_b = str(b.get("primaryForm") or "")
    if not primary_a or not primary_b:
        return 1.0
    if primary_a == primary_b or same_form_family(primary_a, primary_b):
        return 1.0
    families = {form_family(primary_a), form_family(primary_b)}
    if families == {"line", "bar"} and float(a.get("lineBarComboScore") or 0) >= 0.28:
        return 0.86
    if "table" in families:
        return 0.42
    if "pie" in families and len(families) > 1:
        return 0.48
    if families == {"line", "scatter"}:
        return 0.72
    return 0.62


def structural_metric_similarity(a: Dict[str, object], b: Dict[str, object]) -> float:
    specs: List[Tuple[str, float]] = [
        ("tallBarCount", 10.0),
        ("ellipseCapCount", 10.0),
        ("slopedSegmentCount", 18.0),
        ("lineSeriesCount", 4.0),
        ("colorCount", 8.0),
        ("tableGridScore", 1.0),
        ("ringScore", 0.5),
        ("circleScore", 0.8),
        ("lineAreaFillScore", 0.4),
        ("lineBarComboScore", 0.5),
        ("barAreaRatio", 0.18),
    ]
    scores = [scalar_similarity(a, b, key, scale) for key, scale in specs]
    return float(sum(scores) / max(1, len(scores)))


def same_form_family(a: str, b: str) -> bool:
    return bool(a and b and form_family(a) == form_family(b))


def form_family(form: str) -> str:
    if "table" in form:
        return "table"
    if "pie" in form or "ring" in form or "donut" in form:
        return "pie"
    if "line" in form or "area" in form:
        return "line"
    if "bar" in form or "prism" in form or "cylinder" in form or "liquid" in form:
        return "bar"
    if "scatter" in form:
        return "scatter"
    return form


def choose_primary_form(forms: List[str], scores: Dict[str, float]) -> str:
    if not forms:
        return ""
    priority = {
        "table_grid": 8,
        "donut_pie": 7,
        "pie": 6,
        "vertical_bar_line_overlay": 6,
        "isometric_prism_bar": 5,
        "cylinder_vertical_bar": 5,
        "liquid_vertical_bar": 5,
        "line_gradient_area": 4,
        "line_chart": 4,
        "vertical_bar": 3,
    }
    return max(forms, key=lambda item: (float(scores.get(item) or 0.0), priority.get(item, 0)))


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
    threshold = max(0.06, float(np.percentile(column_density, 74)) * 0.6)
    active = column_density >= threshold
    segments: List[Dict[str, object]] = []
    start: Optional[int] = None
    for index, value in enumerate(active.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            end = index
            if end - start >= max(3, int(width * 0.014)):
                roi = mask[:, start:end]
                ys = np.where(np.count_nonzero(roi, axis=1) > 0)[0]
                if ys.size:
                    top, bottom = int(ys.min()), int(ys.max())
                    seg_h = bottom - top + 1
                    seg_w = end - start
                    if seg_h >= height * 0.16 and seg_h > seg_w * 0.9:
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


def colored_tall_contours(mask: np.ndarray, hsv: np.ndarray) -> List[Dict[str, object]]:
    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments: List[Dict[str, object]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        if area < width * height * 0.006:
            continue
        if h < height * 0.20 or h <= w * 0.85:
            continue
        roi = mask[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        fill = float(np.count_nonzero(roi)) / float(roi.size)
        if fill < 0.10:
            continue
        hue_values = hsv[y:y + h, x:x + w, 0][roi.astype(bool)]
        hue = float(np.median(hue_values)) if hue_values.size else -1.0
        segments.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": float(np.count_nonzero(roi)),
                "hueGroup": hue_group(hue),
            }
        )
    return sorted(segments, key=lambda item: int(item["x"]))


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


def ellipse_cap_count(edges: np.ndarray, width: int, height: int) -> int:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for contour in contours:
        x, _y, w, h = cv2.boundingRect(contour)
        if w < 8 or h < 3:
            continue
        if w >= h * 1.45 and h <= max(16, height * 0.16) and w <= width * 0.24 and x <= width:
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
    upright_score = upright_bar_score(boxes)
    low_facet_score = max(0.0, 1.0 - colored_slanted_edges / max(4.0, bar_count * 1.2))
    base_score = local_ellipse_base_score(edges, boxes, width, height)
    return float(min(1.0, 0.32 * cap_score + 0.26 * gradient_score + 0.18 * upright_score + 0.14 * low_facet_score + 0.10 * base_score))


def prism_bar_score(
    edges: np.ndarray,
    boxes: List[Tuple[int, int, int, int, float]],
    color_count: int,
    ellipse_caps: int,
    colored_slanted_edges: int,
) -> float:
    if not boxes:
        return 0.0
    bar_count = len(boxes)
    hue_score = min(1.0, color_count / 5.0)
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


def upright_bar_score(boxes: List[Tuple[int, int, int, int, float]]) -> float:
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


def sloped_line_segments(edges: np.ndarray, hsv: np.ndarray, width: int, height: int) -> List[Tuple[int, int, int, int, float, str]]:
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(12, int(min(width, height) * 0.06)),
        minLineLength=max(10, int(width * 0.055)),
        maxLineGap=max(5, int(width * 0.035)),
    )
    if lines is None:
        return []
    segments: List[Tuple[int, int, int, int, float, str]] = []
    for item in lines.reshape(-1, 4):
        x1, y1, x2, y2 = [int(v) for v in item]
        dx = x2 - x1
        dy = y2 - y1
        length = float((dx * dx + dy * dy) ** 0.5)
        if length < max(10, width * 0.05):
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        angle = min(angle, 180 - angle)
        if 8 <= angle <= 78:
            segments.append((x1, y1, x2, y2, length, median_hue_group_on_segment(hsv, x1, y1, x2, y2)))
    return segments


def median_hue_group_on_segment(hsv: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> str:
    height, width = hsv.shape[:2]
    steps = max(6, int(max(abs(x2 - x1), abs(y2 - y1))))
    xs = np.linspace(x1, x2, steps).round().astype(np.int32)
    ys = np.linspace(y1, y2, steps).round().astype(np.int32)
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)
    selected = hsv[ys, xs]
    saturated = selected[(selected[:, 1] > 35) & (selected[:, 2] > 50)]
    if saturated.size == 0:
        return ""
    return hue_group(float(np.median(saturated[:, 0])))


def bright_line_series_groups(hsv: np.ndarray) -> set[str]:
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    selected = (saturation > 88) & (value > 128)
    total = int(np.count_nonzero(selected))
    if total < 60:
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
        if count >= max(45, int(total * 0.07)):
            groups[raw_group] = count
    if "red" in groups and "orange" in groups:
        groups["red"] += groups.pop("orange")
    return set(groups.keys())


def table_grid_score(edges: np.ndarray, width: int, height: int) -> float:
    area = float(max(1, width * height))
    horizontal = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, width // 8), 1)),
    )
    vertical = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, height // 8))),
    )
    h_density = float(np.count_nonzero(horizontal)) / area
    v_density = float(np.count_nonzero(vertical)) / area
    return max(0.0, min(1.0, (h_density * 18.0 + v_density * 22.0) / 2.0))


def donut_ring_score(gray: np.ndarray, mask: np.ndarray) -> float:
    h, w = gray.shape[:2]
    cx1, cx2 = int(w * 0.34), int(w * 0.66)
    cy1, cy2 = int(h * 0.34), int(h * 0.66)
    center = mask[cy1:cy2, cx1:cx2]
    outer = mask[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
    if center.size == 0 or outer.size == 0:
        return 0.0
    center_density = float(np.count_nonzero(center)) / float(center.size)
    outer_density = float(np.count_nonzero(outer)) / float(outer.size)
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


def circularity_score(mask: np.ndarray, width: int, height: int) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < width * height * 0.015:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / float(max(h, 1))
        if not 0.45 <= aspect <= 2.2:
            continue
        circularity = 4.0 * np.pi * area / float(perimeter * perimeter)
        bbox_share = area / float(max(1, w * h))
        best = max(best, min(1.0, circularity * 0.75 + bbox_share * 0.25))
    return float(best)


def color_area_ratio(hsv: np.ndarray, mask: np.ndarray, color: str) -> float:
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    base = mask.astype(bool) & (sat > 45) & (val > 40)
    if color == "red":
        selected = base & ((hue <= 10) | (hue >= 165))
    else:
        selected = base
    return float(np.count_nonzero(selected)) / float(max(1, np.count_nonzero(mask)))


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
    return min(1.0, float(np.count_nonzero(fill_mask & chart_band)) / float(max(1, np.count_nonzero(chart_band))))


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


def tube_like_bar_score(gray: np.ndarray, hsv: np.ndarray, boxes: List[Tuple[int, int, int, int, float]]) -> float:
    if not boxes:
        return 0.0
    scores = []
    for x, y, w, h, _area in boxes:
        roi_gray = gray[y:y + h, x:x + w]
        roi_hsv = hsv[y:y + h, x:x + w]
        if roi_gray.size == 0:
            continue
        edge = cv2.Canny(roi_gray, 45, 135)
        edge_density = float(np.count_nonzero(edge)) / float(max(edge.size, 1))
        fill_density = float(np.count_nonzero((roi_hsv[:, :, 2] > 80) & (roi_hsv[:, :, 1] > 45))) / float(max(roi_gray.size, 1))
        scores.append(min(1.0, 0.55 * edge_density * 8.0 + 0.45 * fill_density))
    return float(sum(scores) / max(1, len(scores)))


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
