from __future__ import annotations

from .schemas import BBox


def intersection(a: BBox, b: BBox) -> BBox:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.right, b.right)
    y2 = min(a.bottom, b.bottom)
    return BBox(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


def iou(a: BBox, b: BBox) -> float:
    inter = intersection(a, b).area
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def overlap_ratio(a: BBox, b: BBox) -> float:
    inter = intersection(a, b).area
    base = min(a.area, b.area)
    return inter / base if base > 0 else 0.0


def containment(parent: BBox, child: BBox) -> float:
    inter = intersection(parent, child).area
    return inter / child.area if child.area > 0 else 0.0


def center_inside(parent: BBox, child: BBox) -> bool:
    cx, cy = child.center
    return parent.x <= cx <= parent.right and parent.y <= cy <= parent.bottom


def normalized_center_distance(a: BBox, b: BBox) -> float:
    ax, ay = a.center
    bx, by = b.center
    dx = ax - bx
    dy = ay - by
    denom = max(a.w + b.w, a.h + b.h, 1.0)
    return ((dx * dx + dy * dy) ** 0.5) / denom

