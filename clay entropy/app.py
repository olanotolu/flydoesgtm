"""Clay Entropy local MVP server.

Run from the repository root or this directory:

    python3 "clay entropy/app.py"
    python3 app.py 4180
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact import export_package
from clay_client import ClayAPIError, ClayClient, verify_webhook_signature
from entropy_engine import SAMPLES, evaluate, normalize_input
from store import EXPORT_ROOT, attach_package, create, get, list_records, set_status

MAX_BODY = 256 * 1024
JSON = "application/json; charset=utf-8"


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value) -> None:
        self._send(status, json.dumps(value, allow_nan=False).encode(), JSON)

    def _error(self, status: int, exc: Exception) -> None:
        self._json(status, {"error": str(exc)})

    def _body(self) -> dict:
        size = int(self.headers.get("Content-Length", 0))
        if size < 0 or size > MAX_BODY:
            raise ValueError("request body too large")
        raw = self.rfile.read(size) or b"{}"
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _path(self) -> str:
        return unquote(urlsplit(self.path).path).rstrip("/") or "/"

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = self._path()
        try:
            if path == "/":
                body = (ROOT / "index.html").read_bytes()
                self._send(200, body, "text/html; charset=utf-8")
            elif path == "/api/samples":
                self._json(200, {"samples": SAMPLES})
            elif path == "/api/records":
                self._json(200, {"records": list_records()})
            elif path == "/api/clay/status":
                client = ClayClient()
                self._json(200, {"configured": client.configured})
            elif path == "/api/clay/me":
                self._json(200, {"user": ClayClient().authenticated_user()})
            elif path.startswith("/api/records/"): 
                self._json(200, {"record": get(path.rsplit("/", 1)[-1])})
            elif path.startswith("/exports/"):
                rel = Path(path.removeprefix("/exports/"))
                target = (EXPORT_ROOT / rel).resolve()
                if EXPORT_ROOT.resolve() not in target.parents and target != EXPORT_ROOT.resolve():
                    raise ValueError("invalid export path")
                if not target.is_file():
                    raise FileNotFoundError(str(target))
                content_type = "application/octet-stream"
                if target.suffix in {".json", ".md", ".stl"}:
                    content_type = "text/plain; charset=utf-8"
                self._send(200, target.read_bytes(), content_type)
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:
            status = 404 if isinstance(exc, (KeyError, FileNotFoundError)) else 500
            self._error(status, exc)

    def do_POST(self) -> None:
        path = self._path()
        try:
            if path == "/hooks/clay":
                size = int(self.headers.get("Content-Length", 0))
                if size < 0 or size > MAX_BODY:
                    raise ValueError("request body too large")
                raw = self.rfile.read(size)
                if not verify_webhook_signature(
                    raw, self.headers.get("X-Clay-Signature", "")
                ):
                    self._json(401, {"error": "invalid Clay webhook signature"})
                    return
                event = json.loads(raw or b"{}")
                self._json(202, {"accepted": True, "event": event})
            elif path == "/api/clay/search":
                self._json(200, {"result": ClayClient().create_search(self._body().get("query", ""))})
            elif path == "/api/clay/routines/run":
                body = self._body()
                self._json(200, {"result": ClayClient().run_routine(
                    body.get("routine_id", ""), body.get("items", []), body.get("webhook_id")
                )})
            elif path == "/api/evaluate": 
                input_data = normalize_input(self._body())
                self._json(200, {"input": input_data, "decision": evaluate(input_data)})
            elif path == "/api/records":
                input_data = normalize_input(self._body())
                self._json(201, {"record": create(input_data, evaluate(input_data))})
            elif path.startswith("/api/records/") and path.endswith("/review"):
                record_id = path.split("/")[-2]
                action = self._body().get("status")
                self._json(200, {"record": set_status(record_id, action)})
            elif path.startswith("/api/records/") and path.endswith("/export"):
                record_id = path.split("/")[-2]
                record = get(record_id)
                if record["status"] != "approved_relic":
                    raise ValueError("record must be approved_relic before export")
                if record["recommended_action"] != "relic":
                    raise ValueError("only relic recommendations can be exported")
                package = export_package(record, EXPORT_ROOT)
                self._json(200, {"record": attach_package(record_id, package)})
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:
            status = 400 if isinstance(exc, (ValueError, json.JSONDecodeError)) else 500
            if isinstance(exc, KeyError):
                status = 404
            self._error(status, exc)

    def log_message(self, *_):
        pass


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4180
    print(f"clay entropy mvp — http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
