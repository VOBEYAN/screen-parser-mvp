from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Union

from .schemas import ComponentRecord


class ComponentLibrary:
    def __init__(self, records: Iterable[ComponentRecord]):
        self.records = list(records)
        self.by_key: Dict[str, ComponentRecord] = {record.key: record for record in self.records}

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
        return cls(records)

    def categories(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self.records:
            counts[record.category] = counts.get(record.category, 0) + 1
        return counts

    def filter_by_categories(self, categories: Iterable[str]) -> List[ComponentRecord]:
        category_set = set(categories)
        return [record for record in self.records if record.category in category_set]
