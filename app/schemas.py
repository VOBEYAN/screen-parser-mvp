from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0

    def to_dict(self) -> Dict[str, float]:
        return {"x": round(self.x, 2), "y": round(self.y, 2), "w": round(self.w, 2), "h": round(self.h, 2)}


@dataclass
class Detection:
    detection_id: str
    bbox: BBox
    class_name: str
    confidence: float
    source: str = "heuristic"
    component_id: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["bbox"] = self.bbox.to_dict()
        return data


@dataclass
class Node:
    node_id: str
    bbox: BBox
    type: str
    level: int
    confidence: float
    parent_id: Optional[str] = None
    detection_id: Optional[str] = None
    component_id: Optional[str] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["bbox"] = self.bbox.to_dict()
        return data


@dataclass
class Relation:
    source: str
    target: str
    type: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OverlapIssue:
    source: str
    target: str
    iou: float
    overlap_ratio: float
    severity: str
    intersection: BBox

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["intersection"] = self.intersection.to_dict()
        data["overlapRatio"] = data.pop("overlap_ratio")
        return data


@dataclass
class ComponentRecord:
    key: str
    title: str
    category: str
    category_name: str
    chart_frame: str
    chart_key: str
    con_key: str
    schema: str
    description: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)
