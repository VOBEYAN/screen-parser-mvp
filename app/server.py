from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import default
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .finetune_data import export_qwen_vl_dataset, save_correction_sample
from .pipeline import ScreenParser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent / "web"
REFERENCE_IMAGE_ROOT = PROJECT_ROOT / "data" / "component-reference" / "images"
RUN_ID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


class ParserHandler(BaseHTTPRequestHandler):
    parser: ScreenParser
    artifact_root: Path
    upload_dir: Path

    def do_HEAD(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self._send_file_headers(WEB_ROOT / "index.html", root=WEB_ROOT)
            return
        if path.startswith("/static/"):
            self._send_file_headers(WEB_ROOT / path.removeprefix("/static/"), root=WEB_ROOT)
            return
        if path.startswith("/artifacts/"):
            self._send_file_headers(PROJECT_ROOT / path.lstrip("/"), root=self.artifact_root)
            return
        if path.startswith("/component-reference/images/"):
            self._send_file_headers(REFERENCE_IMAGE_ROOT / path.rsplit("/", 1)[-1], root=REFERENCE_IMAGE_ROOT)
            return
        if path in {"/api/status", "/api/runs", "/api/components"} or path.startswith("/api/runs/"):
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            return
        self.send_error(404, "Not found")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)

        if path == "/":
            self._send_file(WEB_ROOT / "index.html", root=WEB_ROOT)
            return

        if path.startswith("/static/"):
            self._send_file(WEB_ROOT / path.removeprefix("/static/"), root=WEB_ROOT)
            return

        if path == "/api/status":
            self._send_json({"ok": True, "status": self._status_payload()})
            return

        if path == "/api/runs":
            self._send_json({"ok": True, "runs": self._runs_payload()})
            return

        if path == "/api/components":
            self._send_json({"ok": True, "components": self._components_payload()})
            return

        if path.startswith("/api/runs/"):
            run_id = path.rsplit("/", 1)[-1]
            result_path = self._run_result_path(run_id)
            if result_path is None:
                self._send_json({"ok": False, "error": "Run not found"}, status=404)
                return
            self._send_json({"ok": True, "result": self._load_result_payload(result_path)})
            return

        if path.startswith("/artifacts/"):
            file_path = PROJECT_ROOT / path.lstrip("/")
            self._send_file(file_path, root=self.artifact_root)
            return

        if path.startswith("/component-reference/images/"):
            self._send_file(REFERENCE_IMAGE_ROOT / path.rsplit("/", 1)[-1], root=REFERENCE_IMAGE_ROOT)
            return

        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/api/labels":
            self._handle_label_post()
            return
        if path == "/api/finetune/export":
            self._handle_finetune_export()
            return
        if path != "/parse":
            self.send_error(404, "Not found")
            return

        try:
            fields, files = self._parse_multipart_form()
            image_file = files.get("image")
            if image_file is None or not image_file[0]:
                self._send_json({"ok": False, "error": "Missing image file"}, status=400)
                return

            self.upload_dir.mkdir(parents=True, exist_ok=True)
            safe_name = safe_upload_name(image_file[0])
            image_path = self.upload_dir / safe_name
            with image_path.open("wb") as fp:
                fp.write(image_file[1])

            input_type = fields.get("inputType", "design")
            top_k = clamp_int(fields.get("topK"), default=5, minimum=1, maximum=12)
            artifacts = self.parser.parse(
                str(image_path),
                input_type=input_type,
                top_k=top_k,
            )
            result = self._load_result_payload(Path(artifacts["resultJson"]))
            response = {
                "ok": True,
                "artifacts": artifacts,
                "artifactUrls": result["artifactUrls"],
                "summary": result["summary"],
                "result": result,
            }
            self._send_json(response)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=500)

    def _handle_label_post(self) -> None:
        try:
            payload = self._read_json_body()
            sample = save_correction_sample(
                self.artifact_root,
                str(payload.get("runId") or ""),
                str(payload.get("nodeId") or ""),
                str(payload.get("correctComponentId") or ""),
                visual_form=str(payload.get("visualForm") or ""),
                note=str(payload.get("note") or ""),
            )
            self._send_json({"ok": True, "sample": sample})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=500)

    def _handle_finetune_export(self) -> None:
        try:
            payload = self._read_json_body()
            variants = clamp_int(str(payload.get("referenceVariants") or ""), default=8, minimum=1, maximum=120)
            output = Path(payload.get("output") or Path("data") / "finetune" / "qwen_vl_component_recognition.jsonl")
            if not output.is_absolute():
                output = PROJECT_ROOT / output
            manifest = export_qwen_vl_dataset(output, reference_variants=variants)
            self._send_json({"ok": True, "manifest": manifest})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=500)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        if content_length <= 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8") or "{}")

    def _parse_multipart_form(self) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(content_length)
        if not content_type.startswith("multipart/form-data"):
            return {}, {}

        header = (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n\r\n"
        ).encode("utf-8")
        message = BytesParser(policy=default).parsebytes(header + body)
        fields: dict[str, str] = {}
        files: dict[str, tuple[str, bytes]] = {}
        for part in message.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            payload = part.get_payload(decode=True) or b""
            filename = part.get_filename()
            if filename:
                files[name] = (filename, payload)
            else:
                charset = part.get_content_charset() or "utf-8"
                fields[name] = payload.decode(charset, errors="replace")
        return fields, files

    def _send_json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, file_path: Path, root: Path | None = None) -> None:
        if root is not None and not is_child_path(file_path, root):
            self.send_error(403, "Forbidden")
            return
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, "File not found")
            return
        suffix = file_path.suffix.lower()
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".md": "text/markdown; charset=utf-8",
        }.get(suffix, "application/octet-stream")
        payload = file_path.read_bytes()
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_file_headers(self, file_path: Path, root: Path | None = None) -> None:
        if root is not None and not is_child_path(file_path, root):
            self.send_error(403, "Forbidden")
            return
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, "File not found")
            return
        suffix = file_path.suffix.lower()
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".md": "text/markdown; charset=utf-8",
        }.get(suffix, "application/octet-stream")
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.end_headers()

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,HEAD,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _status_payload(self) -> dict[str, Any]:
        detector = self.parser.detector
        visual_library = self.parser.visual_library
        library = self.parser.library
        categories = library.categories()
        model_path = getattr(detector, "model_path", None)
        classifier_config = self.parser.multimodal_classifier.config
        return {
            "detector": detector.__class__.__name__,
            "detectorModel": model_path,
            "detectorConfidence": getattr(detector, "conf_threshold", None),
            "hierarchyMode": self.parser.hierarchyMode,
            "matcherMode": "visual_reference" if visual_library.enabled else "catalog_rules",
            "contentClassifier": self.parser.multimodal_classifier.__class__.__name__,
            "contentClassifierMode": self.parser.multimodal_classifier.mode,
            "llmEnabled": classifier_config.llm_enabled,
            "llmConfigured": bool(classifier_config.model and classifier_config.api_key),
            "llmModel": classifier_config.model,
            "llmBaseUrl": classifier_config.base_url,
            "llmForce": classifier_config.force_llm,
            "llmMaxNodes": classifier_config.max_nodes,
            "llmCandidateK": classifier_config.candidate_k,
            "componentLibraryCount": len(library.records),
            "componentCategoryCount": len(categories),
            "visualReferenceCount": len(visual_library.references),
            "catalogPath": str(Path(self.parser.catalog_path)),
            "artifactRoot": str(self.artifact_root),
        }

    def _components_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "componentId": record.key,
                "title": record.title,
                "category": record.category,
                "categoryName": record.category_name,
            }
            for record in self.parser.library.records
        ]

    def _runs_payload(self, limit: int = 80) -> list[dict[str, Any]]:
        results = []
        if not self.artifact_root.exists():
            return results
        result_paths = sorted(
            self.artifact_root.glob("run_*/result.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for result_path in result_paths[:limit]:
            try:
                result = self._load_result_payload(result_path)
            except Exception:
                continue
            summary = result["summary"]
            image_meta = result.get("imageMeta") or {}
            results.append({
                "runId": result.get("runId") or result_path.parent.name,
                "imageName": Path(str(image_meta.get("path", ""))).name,
                "imageSize": f"{image_meta.get('width', '-')} x {image_meta.get('height', '-')}",
                "summary": summary,
                "artifactUrls": result["artifactUrls"],
                "modifiedAt": result_path.stat().st_mtime,
            })
        return results

    def _run_result_path(self, run_id: str) -> Path | None:
        if not run_id or any(char not in RUN_ID_CHARS for char in run_id):
            return None
        result_path = self.artifact_root / run_id / "result.json"
        if not is_child_path(result_path, self.artifact_root):
            return None
        if not result_path.exists():
            return None
        return result_path

    def _load_result_payload(self, result_path: Path) -> dict[str, Any]:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["artifactUrls"] = build_artifact_urls(result_path, result)
        result["summary"] = summarize_result(result)
        return result


def to_artifact_url(path: str) -> str:
    file_path = Path(path).resolve()
    try:
        relative = file_path.relative_to(PROJECT_ROOT)
    except ValueError:
        return str(file_path)
    return "/" + relative.as_posix()


def build_artifact_urls(result_path: Path, result: dict[str, Any]) -> dict[str, str]:
    run_dir = result_path.parent
    image_meta = result.get("imageMeta") if isinstance(result.get("imageMeta"), dict) else {}
    urls = {
        "resultJson": to_artifact_url(str(result_path)),
        "reportMd": to_artifact_url(str(run_dir / "report.md")),
        "reportHtml": to_artifact_url(str(run_dir / "report.html")),
        "evidenceImage": to_artifact_url(str(result.get("evidenceImage") or run_dir / "evidence.png")),
    }
    source_path = image_meta.get("path") if isinstance(image_meta, dict) else None
    if source_path:
        urls["sourceImage"] = to_artifact_url(str(source_path))
    return urls


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    nodes = result.get("nodes") if isinstance(result.get("nodes"), list) else []
    detections = result.get("detections") if isinstance(result.get("detections"), list) else []
    relations = result.get("relations") if isinstance(result.get("relations"), list) else []
    overlaps = result.get("overlaps") if isinstance(result.get("overlaps"), list) else []
    content_classifier = result.get("contentClassifier") if isinstance(result.get("contentClassifier"), dict) else {}
    ai_schema_components = result.get("aiSchemaComponents") if isinstance(result.get("aiSchemaComponents"), list) else []
    component_nodes = [node for node in nodes if isinstance(node, dict) and node.get("type") != "Screen"]
    type_counts: dict[str, int] = {}
    component_counts: dict[str, dict[str, Any]] = {}
    confidence_sum = 0.0
    confidence_count = 0
    llm_hits = 0

    for node in component_nodes:
        node_type = str(node.get("type") or "Unknown")
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
        confidence = node.get("confidence")
        if isinstance(confidence, (int, float)):
            confidence_sum += float(confidence)
            confidence_count += 1
        classifier = ((node.get("features") or {}).get("contentClassifier") or {}) if isinstance(node.get("features"), dict) else {}
        if isinstance(classifier, dict) and classifier.get("llmComponentId"):
            llm_hits += 1
        candidate = top_candidate(node)
        component_id = str(node.get("component_id") or candidate.get("componentId") or "")
        if component_id:
            item = component_counts.setdefault(
                component_id,
                {
                    "componentId": component_id,
                    "title": str(candidate.get("title") or ""),
                    "count": 0,
                    "scoreSum": 0.0,
                },
            )
            item["count"] += 1
            item["scoreSum"] += float(candidate.get("score") or 0.0)

    components = []
    for item in component_counts.values():
        count = max(1, int(item["count"]))
        components.append({
            "componentId": item["componentId"],
            "title": item["title"],
            "count": item["count"],
            "avgScore": round(float(item["scoreSum"]) / count, 4),
        })
    components.sort(key=lambda item: (-int(item["count"]), str(item["componentId"])))

    return {
        "nodeCount": len(component_nodes),
        "detectionCount": len(detections),
        "relationCount": len(relations),
        "overlapCount": len(overlaps),
        "warningCount": len(result.get("warnings") or []),
        "averageConfidence": round(confidence_sum / max(confidence_count, 1), 4),
        "llmMatchCount": llm_hits,
        "llmEnabled": bool(content_classifier.get("llmEnabled")),
        "llmCallCount": int(content_classifier.get("llmCallCount") or 0),
        "llmModel": content_classifier.get("llmModel"),
        "forceLlm": bool(content_classifier.get("forceLlm")),
        "paddleOcrEnabled": bool(content_classifier.get("paddleOcrEnabled")),
        "paddleOcrTextCount": int(content_classifier.get("paddleOcrTextCount") or 0),
        "paddleOcrError": content_classifier.get("paddleOcrError"),
        "contentClassifierErrors": content_classifier.get("errors") or [],
        "typeCounts": type_counts,
        "components": components[:16],
        "aiSchemaComponentCount": len(ai_schema_components),
        "contentClassifierMode": content_classifier.get("mode"),
    }


def top_candidate(node: dict[str, Any]) -> dict[str, Any]:
    candidates = node.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        return candidates[0]
    return {}


def safe_upload_name(filename: str) -> str:
    path = Path(filename)
    stem = "".join(char if char.isalnum() or char in "-_." else "_" for char in path.stem).strip("._")
    suffix = path.suffix if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} else ".png"
    if not stem:
        stem = "upload"
    return f"{stem}{suffix}"


def clamp_int(value: str | None, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value or default)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def clean_optional(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def is_child_path(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run screen parser MVP HTTP server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--catalog", default=str(PROJECT_ROOT.parent / "ai-schema-view" / "schema-md" / "catalog.md"))
    parser.add_argument("--artifacts", default=str(PROJECT_ROOT / "artifacts"))
    parser.add_argument("--yolo-model", default=None)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--graph-model", default=None)
    parser.add_argument("--reference-library", default=None)
    parser.add_argument("--multimodal-classifier", action="store_true")
    parser.add_argument("--multimodal-model", default=None)
    parser.add_argument("--multimodal-base-url", default=None)
    parser.add_argument("--multimodal-api-key", default=None)
    args = parser.parse_args()

    ParserHandler.parser = ScreenParser(
        args.catalog,
        artifact_root=args.artifacts,
        yolo_model=args.yolo_model,
        yolo_conf=args.yolo_conf,
        graph_model=args.graph_model,
        reference_library=args.reference_library,
        multimodal_classifier=args.multimodal_classifier,
        multimodal_model=args.multimodal_model,
        multimodal_base_url=args.multimodal_base_url,
        multimodal_api_key=args.multimodal_api_key,
    )
    ParserHandler.artifact_root = Path(args.artifacts)
    ParserHandler.upload_dir = Path(args.artifacts) / "uploads"

    server = ThreadingHTTPServer((args.host, args.port), ParserHandler)
    print(f"Screen parser server: http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
