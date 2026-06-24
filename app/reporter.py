from __future__ import annotations

import json
import shutil
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from .schemas import BBox, Detection, Node, OverlapIssue, Relation


class ReportWriter:
    def __init__(self, artifact_root: str = "artifacts"):
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        image_path: str,
        image_size: Tuple[int, int],
        detections: List[Detection],
        nodes: List[Node],
        relations: List[Relation],
        overlaps: List[OverlapIssue],
        extras: Optional[Dict[str, object]] = None,
    ) -> Dict[str, str]:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        run_dir = self.artifact_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        image_copy = run_dir / Path(image_path).name
        shutil.copy2(image_path, image_copy)

        evidence_path = run_dir / "evidence.png"
        self._draw_evidence(image_path, nodes, overlaps, evidence_path)

        result = {
            "runId": run_id,
            "imageMeta": {"path": str(image_copy), "width": image_size[0], "height": image_size[1]},
            "detections": [item.to_dict() for item in detections],
            "nodes": [item.to_dict() for item in nodes],
            "relations": [item.to_dict() for item in relations],
            "overlaps": [item.to_dict() for item in overlaps],
            "warnings": self._warnings(overlaps),
            "evidenceImage": str(evidence_path),
        }
        if extras:
            result.update(extras)

        result_path = run_dir / "result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        report_path = run_dir / "report.md"
        report_path.write_text(self._markdown(result), encoding="utf-8")

        html_path = run_dir / "report.html"
        html_path.write_text(self._html(result), encoding="utf-8")

        return {
            "runId": run_id,
            "runDir": str(run_dir),
            "resultJson": str(result_path),
            "reportMd": str(report_path),
            "reportHtml": str(html_path),
            "evidenceImage": str(evidence_path),
        }

    def _draw_evidence(self, image_path: str, nodes: List[Node], overlaps: List[OverlapIssue], output_path: Path) -> None:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")

        colors = {
            "Panel": (76, 201, 240, 220),
            "Region": (125, 211, 252, 170),
            "Content": (148, 163, 184, 190),
            "Title": (255, 206, 86, 230),
            "Border": (90, 160, 255, 220),
            "Chart": (88, 214, 141, 230),
            "Table": (255, 159, 64, 230),
            "Map": (153, 102, 255, 230),
            "MetricCard": (255, 99, 132, 230),
            "Decorate": (201, 203, 207, 210),
        }

        for node in nodes:
            if node.type == "Screen":
                continue
            color = colors.get(node.type, (255, 255, 255, 220))
            self._rect(draw, node.bbox, outline=color, width=3)
            component_label = node.component_id or node.type
            label = f"{node.node_id} {component_label} {node.confidence:.2f}"
            draw.rectangle([node.bbox.x, max(0, node.bbox.y - 18), node.bbox.x + min(360, len(label) * 7), node.bbox.y], fill=(0, 0, 0, 160))
            draw.text((node.bbox.x + 3, max(0, node.bbox.y - 16)), label, fill=(255, 255, 255, 255))

        for issue in overlaps:
            self._rect(draw, issue.intersection, outline=(255, 0, 0, 255), fill=(255, 0, 0, 70), width=4)

        image.save(output_path)

    def _rect(self, draw: ImageDraw.ImageDraw, bbox: BBox, outline, width: int = 2, fill=None) -> None:
        draw.rectangle([bbox.x, bbox.y, bbox.right, bbox.bottom], outline=outline, width=width, fill=fill)

    def _warnings(self, overlaps: List[OverlapIssue]) -> List[str]:
        if not overlaps:
            return []
        errors = sum(1 for issue in overlaps if issue.severity == "error")
        warnings = len(overlaps) - errors
        return [f"Detected {len(overlaps)} sibling overlap issue(s), including {errors} error and {warnings} warning."]

    def _markdown(self, result: Dict[str, object]) -> str:
        nodes = result["nodes"]
        overlaps = result["overlaps"]
        lines = [
            "# 大屏图纸解析报告",
            "",
            f"- Run ID: `{result['runId']}`",
            f"- Image: `{result['imageMeta']['path']}`",
            f"- Nodes: {len(nodes)}",
            f"- Overlap issues: {len(overlaps)}",
            f"- Evidence: `{result['evidenceImage']}`",
            "",
            "## 组件库匹配汇总",
            "",
            "| componentId | title | count | avgScore |",
            "|---|---|---:|---:|",
        ]
        for item in self._component_summary(nodes):
            lines.append(f"| {item['componentId']} | {item['title']} | {item['count']} | {item['avgScore']} |")

        lines.extend([
            "",
            "## 组件节点",
            "",
            "| nodeId | type | parentId | componentId | componentTitle | matchScore | matchMode | confidence | bbox |",
            "|---|---|---|---|---|---:|---|---:|---|",
        ])
        for node in nodes:
            if node["type"] == "Screen":
                continue
            candidate = self._top_candidate(node)
            lines.append(
                f"| {node['node_id']} | {node['type']} | {node.get('parent_id') or ''} | "
                f"{node.get('component_id') or candidate.get('componentId', '')} | "
                f"{candidate.get('title', '')} | {candidate.get('score', '')} | "
                f"{candidate.get('matchMode', '')} | {node['confidence']:.2f} | {node['bbox']} |"
            )

        lines.extend(["", "## 重叠问题", ""])
        if not overlaps:
            lines.append("未检测到同级组件异常重叠。")
        else:
            lines.extend(["| source | target | severity | IoU | overlapRatio |", "|---|---|---|---|---|"])
            for issue in overlaps:
                lines.append(
                    f"| {issue['source']} | {issue['target']} | {issue['severity']} | "
                    f"{issue['iou']} | {issue['overlapRatio']} |"
                )
        return "\n".join(lines) + "\n"

    def _html(self, result: Dict[str, object]) -> str:
        result_json = json.dumps(result, ensure_ascii=False, indent=2)
        evidence_name = Path(str(result["evidenceImage"])).name
        rows = "\n".join(self._html_node_rows(result["nodes"]))
        summary_rows = "\n".join(self._html_summary_rows(result["nodes"]))
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Screen Parser Report</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #d8e3f0; background: #101820; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    img {{ max-width: 100%; border: 1px solid #334155; }}
    table {{ width: 100%; border-collapse: collapse; margin: 12px 0 24px; font-size: 13px; }}
    th, td {{ border: 1px solid #334155; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #172335; color: #e8f2ff; }}
    pre {{ overflow: auto; background: #0b1220; padding: 16px; border: 1px solid #334155; }}
  </style>
</head>
<body>
  <main>
    <h1>大屏图纸解析报告</h1>
    <p>Run ID: <code>{result['runId']}</code></p>
    <p>Nodes: {len(result['nodes'])}, Overlap issues: {len(result['overlaps'])}</p>
    <h2>可视化证据</h2>
    <img src="{evidence_name}" alt="evidence" />
    <h2>组件库匹配汇总</h2>
    <table>
      <thead><tr><th>componentId</th><th>title</th><th>count</th><th>avgScore</th></tr></thead>
      <tbody>{summary_rows}</tbody>
    </table>
    <h2>组件节点</h2>
    <table>
      <thead><tr><th>nodeId</th><th>type</th><th>parentId</th><th>componentId</th><th>componentTitle</th><th>matchScore</th><th>matchMode</th><th>confidence</th><th>bbox</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <h2>结构化 JSON</h2>
    <pre>{result_json}</pre>
  </main>
</body>
</html>
"""

    def _top_candidate(self, node: Dict[str, object]) -> Dict[str, object]:
        candidates = node.get("candidates") or []
        if isinstance(candidates, list) and candidates:
            candidate = candidates[0]
            if isinstance(candidate, dict):
                return candidate
        return {}

    def _component_summary(self, nodes: List[Dict[str, object]]) -> List[Dict[str, object]]:
        groups: Dict[str, Dict[str, object]] = {}
        for node in nodes:
            if node["type"] == "Screen":
                continue
            candidate = self._top_candidate(node)
            component_id = str(node.get("component_id") or candidate.get("componentId") or "")
            if not component_id:
                continue
            group = groups.setdefault(
                component_id,
                {"componentId": component_id, "title": str(candidate.get("title") or ""), "count": 0, "scoreSum": 0.0},
            )
            group["count"] = int(group["count"]) + 1
            group["scoreSum"] = float(group["scoreSum"]) + float(candidate.get("score") or 0.0)

        summary = []
        for group in groups.values():
            count = int(group["count"])
            avg_score = float(group["scoreSum"]) / max(count, 1)
            summary.append({"componentId": group["componentId"], "title": group["title"], "count": count, "avgScore": round(avg_score, 4)})
        summary.sort(key=lambda item: (-int(item["count"]), str(item["componentId"])))
        return summary

    def _html_summary_rows(self, nodes: List[Dict[str, object]]) -> List[str]:
        rows = []
        for item in self._component_summary(nodes):
            rows.append(
                "<tr>"
                f"<td>{escape(str(item['componentId']))}</td>"
                f"<td>{escape(str(item['title']))}</td>"
                f"<td>{item['count']}</td>"
                f"<td>{item['avgScore']}</td>"
                "</tr>"
            )
        return rows

    def _html_node_rows(self, nodes: List[Dict[str, object]]) -> List[str]:
        rows = []
        for node in nodes:
            if node["type"] == "Screen":
                continue
            candidate = self._top_candidate(node)
            component_id = str(node.get("component_id") or candidate.get("componentId") or "")
            rows.append(
                "<tr>"
                f"<td>{escape(str(node['node_id']))}</td>"
                f"<td>{escape(str(node['type']))}</td>"
                f"<td>{escape(str(node.get('parent_id') or ''))}</td>"
                f"<td>{escape(component_id)}</td>"
                f"<td>{escape(str(candidate.get('title', '')))}</td>"
                f"<td>{escape(str(candidate.get('score', '')))}</td>"
                f"<td>{escape(str(candidate.get('matchMode', '')))}</td>"
                f"<td>{float(node['confidence']):.2f}</td>"
                f"<td>{escape(str(node['bbox']))}</td>"
                "</tr>"
            )
        return rows
