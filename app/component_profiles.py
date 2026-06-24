from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .component_library import ComponentLibrary
from .schemas import ComponentRecord


PROFILE_FILE = "component_vlm_profiles.json"


def load_component_profiles(reference_path: Optional[str], library: ComponentLibrary) -> Dict[str, Dict[str, Any]]:
    path = Path(reference_path or "") / PROFILE_FILE if reference_path else None
    if path and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("components") if isinstance(payload, dict) else payload
        if isinstance(items, list):
            profiles = {
                str(item.get("componentId") or ""): item
                for item in items
                if isinstance(item, dict) and item.get("componentId")
            }
            if profiles:
                return profiles
    return {record.key: infer_component_profile(record) for record in library.records}


def infer_component_profile(record: ComponentRecord) -> Dict[str, Any]:
    text = f"{record.key} {record.title} {record.category} {record.category_name} {record.description}".lower()
    content_type = category_content_type(record.category)
    visual_form = content_type

    rules = [
        ("liquidbar", "bar_chart", "liquid_vertical_bar"),
        ("liquid", "bar_chart", "liquid_vertical_bar"),
        ("colorprism", "bar_chart", "3d_prism_bar"),
        ("prismatic", "bar_chart", "3d_prism_bar"),
        ("cylinder", "bar_chart", "cylinder_bar"),
        ("capsule", "bar_chart", "capsule_bar"),
        ("crossrange", "bar_chart", "horizontal_bar"),
        ("barline", "bar_chart", "bar_line_combo"),
        ("vchartbar", "bar_chart", "standard_bar"),
        ("barcommon", "bar_chart", "standard_bar"),
        ("line", "line_chart", "line_chart"),
        ("area", "area_chart", "area_chart"),
        ("piecircle", "pie_chart", "donut_pie"),
        ("piecommon", "pie_chart", "flat_pie"),
        ("pie3dring", "pie_chart", "3d_donut_pie"),
        ("pie3d", "pie_chart", "3d_pie"),
        ("vchartpie", "pie_chart", "flat_pie"),
        ("scatter", "scatter_chart", "scatter_chart"),
        ("table", "table", "table_grid"),
        ("alarmlist", "table", "alarm_list"),
        ("border", "border", "border_frame"),
        ("decorates", "decorate", "decorative_asset"),
        ("title", "title", "title_text"),
        ("text", "title", "text_block"),
        ("map", "map", "map"),
        ("china", "map", "china_map"),
        ("earth", "map", "earth_3d"),
        ("funnel", "funnel_chart", "funnel"),
        ("wordcloud", "wordcloud", "wordcloud"),
        ("radar", "chart", "radar_chart"),
        ("sankey", "chart", "sankey"),
    ]
    compact = re.sub(r"[^a-z0-9]+", "", text)
    for token, matched_type, matched_form in rules:
        if token in compact:
            content_type = matched_type
            visual_form = matched_form
            break

    return {
        "componentId": record.key,
        "title": record.title,
        "category": record.category,
        "contentType": content_type,
        "visualForm": visual_form,
        "layout": infer_layout(text),
        "semanticKeywords": semantic_keywords(record),
        "source": "heuristic",
    }


def category_content_type(category: str) -> str:
    return {
        "Bars": "bar_chart",
        "Lines": "line_chart",
        "Areas": "area_chart",
        "Pies": "pie_chart",
        "Scatters": "scatter_chart",
        "Funnels": "funnel_chart",
        "WordClouds": "wordcloud",
        "Tables": "table",
        "Maps": "map",
        "Borders": "border",
        "Decorates": "decorate",
        "Title": "title",
        "Texts": "title",
        "Inputs": "filter",
    }.get(category, "chart")


def infer_layout(text: str) -> str:
    if any(token in text for token in ["横向", "horizontal", "crossrange"]):
        return "horizontal"
    if any(token in text for token in ["纵向", "vertical", "bar", "柱"]):
        return "vertical"
    if any(token in text for token in ["中心", "圆", "pie", "ring"]):
        return "centered"
    if any(token in text for token in ["表格", "列表", "grid", "table"]):
        return "grid"
    return "balanced"


def semantic_keywords(record: ComponentRecord) -> List[str]:
    text = f"{record.key} {record.title} {record.description}"
    tokens = re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9]{2,24}", text)
    seen: set[str] = set()
    keywords: List[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        keywords.append(token)
        if len(keywords) >= 16:
            break
    return keywords


def profile_match_score(
    profile: Dict[str, Any],
    content_type: str,
    text: str,
    target_visual_form: str = "",
    evidence_text: str = "",
) -> float:
    if not profile:
        return 0.0
    score = 0.0
    profile_type = str(profile.get("contentType") or "")
    visual_form = str(profile.get("visualForm") or "")
    layout = str(profile.get("layout") or "")
    if profile_type == content_type:
        score += 0.62
    elif compatible_content_types(profile_type, content_type):
        score += 0.38

    target_visual_form = normalize_visual_form(target_visual_form)
    profile_visual_form = normalize_visual_form(visual_form)
    evidence_alignment = evidence_profile_alignment(profile, evidence_text)
    if target_visual_form and profile_visual_form:
        if profile_visual_form == target_visual_form:
            score += 0.16 + 0.12 * evidence_alignment
        elif visual_form_compatible(profile_visual_form, target_visual_form):
            score += 0.08 + 0.1 * evidence_alignment
        else:
            score -= max(0.04, 0.14 * (1.0 - evidence_alignment))

    score += 0.52 * evidence_alignment

    lowered = str(text or "").lower()
    # Text is evidence for the broad content family, not for a specific visual
    # component. Exact component selection must come from crop/profile shape
    # agreement so that a new unknown design is not forced into a memorized id.
    if "任务名称" in lowered and profile_type == "table":
        score += 0.12
    if re.search(r"\d{2}-\d{2}", lowered) and profile_type == "line_chart":
        score += 0.1
    if "%" in lowered and profile_type == "pie_chart":
        score += 0.06
    if any(token in lowered for token in ["排行", "top", "指标"]) and profile_type == "bar_chart":
        score += 0.06
    if layout == "horizontal" and any(token in lowered for token in ["排行", "top"]):
        score += 0.04
    return min(1.0, score)


def evidence_profile_alignment(profile: Dict[str, Any], evidence_text: str) -> float:
    evidence = normalize_evidence_terms(evidence_text)
    if not evidence:
        return 0.0

    positive_text = " ".join(
        str(value or "")
        for value in [
            profile.get("title"),
            profile.get("visualForm"),
            profile.get("layout"),
            " ".join(profile.get("semanticKeywords") or []) if isinstance(profile.get("semanticKeywords"), list) else "",
            " ".join(profile.get("distinguishingFeatures") or [])
            if isinstance(profile.get("distinguishingFeatures"), list)
            else "",
        ]
    )
    negative_text = " ".join(profile.get("negativeMatches") or []) if isinstance(profile.get("negativeMatches"), list) else ""
    positive = normalize_evidence_terms(positive_text)
    negative = normalize_evidence_terms(negative_text)

    overlap = len(evidence & positive)
    negative_overlap = len(evidence & negative)
    if overlap == 0 and negative_overlap == 0:
        return 0.0
    return max(0.0, min(1.0, overlap / max(4, len(evidence)) - 0.18 * negative_overlap))


def normalize_evidence_terms(text: str) -> set[str]:
    lowered = str(text or "").lower()
    aliases = {
        "isometric": ["isometric", "等轴测"],
        "prism": ["prism", "prismatic", "棱柱", "棱面", "斜切", "立体柱"],
        "cylinder": ["cylinder", "cylindrical", "圆柱", "柱体"],
        "multicolor": ["distinct colors", "multi color", "multi-color", "多色", "彩色", "blue, purple", "蓝、紫", "橙", "绿", "黄"],
        "base": ["base", "底座", "阴影基座"],
        "top_label": ["top value", "顶部数值", "数值标签", "value label"],
        "bottom_label": ["bottom label", "底部分类", "分类标签", "指标"],
        "gradient": ["gradient", "渐变"],
        "ring": ["ring", "donut", "环形", "圆环"],
        "exploded": ["exploded", "分离", "爆炸"],
        "grid": ["grid", "表格", "栅格"],
        "line": ["line chart", "折线"],
        "axis": ["x-axis", "y-axis", "坐标轴"],
    }
    terms: set[str] = set()
    for term, values in aliases.items():
        if any(value in lowered for value in values):
            terms.add(term)
    for token in re.findall(r"[a-z][a-z0-9_]{2,24}|[\u4e00-\u9fff]{2,8}", lowered):
        if token not in {"the", "and", "with", "chart", "component", "candidate"}:
            terms.add(token)
    return terms


def normalize_visual_form(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def visual_form_compatible(left: str, right: str) -> bool:
    if not left or not right:
        return False
    groups = [
        {"donut_pie", "pie3d_ring", "3d_donut_pie", "ring_pie", "pie_ring", "pie3d_ring_region"},
        {"flat_pie", "pie_chart", "pie_common", "vchart_pie"},
        {"liquid_vertical_bar", "liquid_bar"},
        {"vertical_bar_stacked", "standard_bar", "stacked_vertical_bar", "vertical_bar"},
        {"cylinder_bar", "cylinder_vertical_bar", "gradient_cylinder_bar", "3d_cylinder_bar"},
        {"isometric_prism_bar", "3d_prism_bar", "prismatic_bar", "prismatic_vertical_bar", "color_prism_bar"},
        {"line_chart", "vchart_line"},
        {"line_chart_gradient", "single_line_gradient", "linear_gradient_line"},
        {"line_gradient_area", "single_line_gradient_area"},
        {"double_line_gradient_area", "multi_line_gradient_area"},
        {"percent_area_chart", "percent_area", "vchart_percent_area"},
        {"area_chart", "vchart_area"},
        {"vertical_bar_line_overlay", "bar_line_combo"},
        {"table_grid", "alarm_list", "scroll_table", "table_list"},
    ]
    return any(left in group and right in group for group in groups)


def compatible_content_types(left: str, right: str) -> bool:
    chart_family = {"bar_chart", "line_chart", "area_chart", "pie_chart", "scatter_chart", "funnel_chart", "chart"}
    if left in chart_family and right == "chart":
        return True
    if right in chart_family and left == "chart":
        return True
    return False
