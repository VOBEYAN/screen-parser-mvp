from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from .schemas import ComponentRecord


class ComponentLibrary:
    def __init__(
        self,
        records: Iterable[ComponentRecord],
        option_blueprints: Optional[Dict[str, Dict[str, Any]]] = None,
        option_shapes: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.records = list(records)
        self.by_key: Dict[str, ComponentRecord] = {record.key: record for record in self.records}
        self.option_blueprints = option_blueprints or {}
        self.option_shapes = option_shapes or {}

    @classmethod
    def from_catalog(cls, catalog_path: Union[str, Path]) -> "ComponentLibrary":
        path = Path(catalog_path)
        records: List[ComponentRecord] = []
        if not path.exists():
            return cls(records)

        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("|") or line.startswith("| ---") or line.startswith("| key "):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) < 9:
                continue
            records.append(
                ComponentRecord(
                    key=cells[0],
                    title=cells[1],
                    category=cells[2],
                    category_name=cells[3],
                    chart_frame=cells[4],
                    chart_key=cells[5],
                    con_key=cells[6],
                    schema=cells[7],
                    description=cells[8],
                )
            )
        option_blueprints, option_shapes = load_modeling_source(path)
        return cls(records, option_blueprints=option_blueprints, option_shapes=option_shapes)

    def categories(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self.records:
            counts[record.category] = counts.get(record.category, 0) + 1
        return counts

    def filter_by_categories(self, categories: Iterable[str]) -> List[ComponentRecord]:
        category_set = set(categories)
        return [record for record in self.records if record.category in category_set]

    def option_blueprint(self, component_id: str) -> Dict[str, Any]:
        return copy.deepcopy(self.option_blueprints.get(component_id) or {})

    def option_shape(self, component_id: str) -> Dict[str, Any]:
        return copy.deepcopy(self.option_shapes.get(component_id) or {})


def load_modeling_source(catalog_path: Path) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    modeling_path = catalog_path.parent.parent / "json" / "modeling-source.json"
    if not modeling_path.exists():
        return {}, {}
    try:
        payload = json.loads(modeling_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, {}

    blueprints: Dict[str, Dict[str, Any]] = {}
    shapes: Dict[str, Dict[str, Any]] = {}
    for item in payload.get("componentList") or []:
        if not isinstance(item, dict):
            continue
        chart_config = item.get("chartConfig") if isinstance(item.get("chartConfig"), dict) else {}
        key = str(item.get("key") or chart_config.get("key") or "").strip()
        option = item.get("option") if isinstance(item.get("option"), dict) else {}
        if not key:
            continue
        blueprints[key] = copy.deepcopy(option)
        shapes[key] = analyze_option_shape(option)
    return blueprints, shapes


def analyze_option_shape(option: Dict[str, Any]) -> Dict[str, Any]:
    shape: Dict[str, Any] = {
        "optionKeys": sorted(str(key) for key in option.keys()),
    }
    if "dataset" not in option:
        shape["datasetKind"] = "none"
        return shape

    dataset = option.get("dataset")
    shape.update(dataset_shape(dataset))
    return shape


def dataset_shape(dataset: Any) -> Dict[str, Any]:
    if dataset is None:
        return {"datasetKind": "null"}
    if isinstance(dataset, str):
        return {"datasetKind": "string"}
    if isinstance(dataset, bool):
        return {"datasetKind": "boolean"}
    if isinstance(dataset, (int, float)):
        return {"datasetKind": "number"}
    if isinstance(dataset, list):
        sample = next((item for item in dataset if item is not None), None)
        shape = {
            "datasetKind": "array",
            "arrayItemKind": type_name(sample),
        }
        if isinstance(sample, dict):
            shape["arrayItemKeys"] = sorted(str(key) for key in sample.keys())
        return shape
    if isinstance(dataset, dict):
        shape: Dict[str, Any] = {
            "datasetKind": "object",
            "datasetKeys": sorted(str(key) for key in dataset.keys()),
        }
        if isinstance(dataset.get("dimensions"), list):
            shape["dimensions"] = dimension_keys(dataset.get("dimensions") or [])
        if isinstance(dataset.get("source"), list):
            shape["datasetKind"] = "object.source"
            shape["sourceItemKeys"] = first_dict_keys(dataset.get("source") or [])
        elif isinstance(dataset.get("values"), list):
            shape["datasetKind"] = "object.values"
            shape["valueItemKeys"] = first_dict_keys(dataset.get("values") or [])
        elif isinstance(dataset.get("nodes"), list):
            shape["datasetKind"] = "object.nodes"
            shape["nodeItemKeys"] = first_dict_keys(dataset.get("nodes") or [])
        elif any(isinstance(dataset.get(key), list) for key in ["productNodes", "reportNodes", "platformNodes"]):
            shape["datasetKind"] = "object.bizNodes"
            shape["bizNodeKeys"] = sorted(
                key for key in ["productNodes", "reportNodes", "platformNodes"] if isinstance(dataset.get(key), list)
            )
        return shape
    return {"datasetKind": type_name(dataset)}


def dimension_keys(dimensions: List[Any]) -> List[str]:
    keys: List[str] = []
    for item in dimensions:
        if isinstance(item, dict):
            key = item.get("key") or item.get("name") or item.get("title")
        else:
            key = item
        if key is not None:
            keys.append(str(key))
    return keys


def first_dict_keys(items: List[Any]) -> List[str]:
    for item in items:
        if isinstance(item, dict):
            return sorted(str(key) for key in item.keys())
    return []


def type_name(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__
