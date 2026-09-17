"""Lightweight Vercel replay function.

The hosted showcase intentionally serves the captured proof path. The full
connectome and live Clay adapter stay on the local/managed server so a public
deployment cannot accidentally receive credentials or expose a send path.
"""
from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).parents[1]
REPLAY = json.loads((ROOT / "demo" / "replay.json").read_text())


def payload_for(path, method, body=b""):
    if method == "GET" and path in ("/api/demo/replay/default", "/replay"):
        return 200, REPLAY
    if method == "GET" and path in ("/api/health", "/health"):
        return 200, {
            "ok": True, "mode": "replay",
            "model_version": REPLAY.get("model_version", "recorded-proof"),
            "safe_to_contact": False,
            "live_clay": "disabled on hosted replay deployment",
        }
    if method == "GET" and path == "/api/artifacts/current":
        return 200, {
            "model_version": REPLAY.get("model_version", "recorded-proof"),
            "mode": "recorded_clay_capture",
            "safe_to_contact": False,
            "source": "versioned replay artifact",
        }
    if method == "GET" and path == "/graph":
        return 200, {"n": 0, "points": [], "groups": {}, "pools": {},
                     "channels": {}, "signal_keys": []}
    if method == "POST" and path == "/api/demo/run":
        try:
            request = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return 400, {"error": "request body is invalid"}
        mode = request.get("mode", "replay")
        if mode == "replay":
            return 200, REPLAY
        if mode == "live_draft":
            return 200, {
                "mode": "live_draft", "rows": [], "distribution": {},
                "query": request.get("query"), "safe_to_contact": False,
                "source": "hosted replay fallback",
                "budget": {"records": 0, "enrichments": 0,
                            "credits_used": 0, "max_records": 10,
                            "max_enrichments": 4, "max_credits": 50},
            }
        return 400, {"error": "mode must be replay or live_draft"}
    return 404, {"error": "route not found"}


class handler(BaseHTTPRequestHandler):
    def _send(self, status, value):
        data = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_page(self):
        data = (ROOT / "demo" / "web" / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlsplit(self.path).path
        if not path.startswith("/api/") and path not in ("/health", "/replay", "/graph"):
            self._send_page()
            return
        status, value = payload_for(path, "GET")
        self._send(status, value)

    def do_POST(self):
        size = int(self.headers.get("Content-Length", 0))
        if size > 64 * 1024:
            self._send(413, {"error": "request body too large"})
            return
        status, value = payload_for(urlsplit(self.path).path, "POST",
                                    self.rfile.read(size))
        self._send(status, value)

    def log_message(self, *_):
        pass
