from __future__ import annotations

import argparse
import cgi
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from .pipeline import ScreenParser


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ParserHandler(BaseHTTPRequestHandler):
    parser: ScreenParser
    artifact_root: Path
    upload_dir: Path

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self._send_html(index_html())
            return

        if self.path.startswith("/artifacts/"):
            file_path = PROJECT_ROOT / unquote(self.path.lstrip("/"))
            self._send_file(file_path)
            return

        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        if self.path != "/parse":
            self.send_error(404, "Not found")
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )
        file_item = form["image"] if "image" in form else None
        if file_item is None or not getattr(file_item, "filename", ""):
            self.send_error(400, "Missing image file")
            return

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file_item.filename).name
        image_path = self.upload_dir / safe_name
        with image_path.open("wb") as fp:
            fp.write(file_item.file.read())

        input_type = form.getvalue("inputType", "design")
        top_k = 1
        artifacts = self.parser.parse(str(image_path), input_type=input_type, top_k=top_k)
        response = {
            "ok": True,
            "artifacts": artifacts,
            "reportUrl": to_artifact_url(artifacts["reportHtml"]),
            "resultUrl": to_artifact_url(artifacts["resultJson"]),
        }
        self._send_json(response)

    def _send_html(self, html: str) -> None:
        payload = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, data: dict) -> None:
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, "File not found")
            return
        suffix = file_path.suffix.lower()
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".md": "text/markdown; charset=utf-8",
        }.get(suffix, "application/octet-stream")
        payload = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def to_artifact_url(path: str) -> str:
    file_path = Path(path).resolve()
    try:
        relative = file_path.relative_to(PROJECT_ROOT)
    except ValueError:
        return str(file_path)
    return "/" + relative.as_posix()


def index_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>大屏图纸组件检测 MVP</title>
  <style>
    body { margin: 0; background: #111827; color: #e5e7eb; font-family: Arial, sans-serif; }
    main { max-width: 880px; margin: 0 auto; padding: 32px; }
    form { display: grid; gap: 16px; padding: 20px; border: 1px solid #334155; }
    input, select, button { font-size: 16px; padding: 10px; }
    button { cursor: pointer; background: #38bdf8; border: 0; color: #082f49; font-weight: 700; }
    pre { background: #0b1220; padding: 16px; overflow: auto; }
    a { color: #7dd3fc; }
  </style>
</head>
<body>
  <main>
    <h1>大屏图纸组件检测 MVP</h1>
    <form id="parse-form">
      <label>输入类型
        <select name="inputType">
          <option value="design">高保真设计稿</option>
          <option value="sketch">草图</option>
        </select>
      </label>
      <label>组件匹配数量
        <input name="topK" type="number" value="1" min="1" max="1" readonly />
      </label>
      <label>上传图片
        <input name="image" type="file" accept="image/*" required />
      </label>
      <button type="submit">开始解析</button>
    </form>
    <h2>结果</h2>
    <div id="links"></div>
    <pre id="output">等待上传...</pre>
  </main>
  <script>
    const form = document.getElementById('parse-form');
    const output = document.getElementById('output');
    const links = document.getElementById('links');
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      output.textContent = '解析中...';
      links.innerHTML = '';
      const response = await fetch('/parse', { method: 'POST', body: new FormData(form) });
      const data = await response.json();
      output.textContent = JSON.stringify(data, null, 2);
      if (data.reportUrl) {
        links.innerHTML = `<p><a href="${data.reportUrl}" target="_blank">打开可视化报告</a></p>
          <p><a href="${data.resultUrl}" target="_blank">打开结果 JSON</a></p>`;
      }
    });
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run screen parser MVP HTTP server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--catalog", default=str(PROJECT_ROOT.parent / "ai-schema-view" / "schema-md" / "catalog.md"))
    parser.add_argument("--artifacts", default=str(PROJECT_ROOT / "artifacts"))
    parser.add_argument("--yolo-model", default=None)
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
