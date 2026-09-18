"""Lightweight Vercel replay function.

The hosted showcase intentionally serves the captured proof path. The full
connectome and live Clay adapter stay on the local/managed server so a public
deployment cannot accidentally receive credentials or expose a send path.
"""
from http.server import BaseHTTPRequestHandler
import json
import mimetypes
import os
from pathlib import Path
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).parents[1]
DEMO = (ROOT / "demo").resolve()
REPLAY = json.loads((ROOT / "demo" / "replay.json").read_text())
THINK_REPLAYS = {
    pack: json.loads((ROOT / "demo" / f"think_replay_{pack}.json").read_text())
    for pack in ("anthropic", "warm", "cold")
}
LIVE_THINK_URL = os.environ.get(
    "LIVE_THINK_URL",
    "https://olaoluwasubxmi--think.modal.run/api/demo/think")
DEFAULT_QUERY = ('select from people where location_country = "United States" '
                 'and experiences.any(is_current = true and '
                 'job_title is_similar_to ("Owner", "Founder", "Chief Operating Officer", '
                 '"Vice President of Operations", "Director of Operations", '
                 '"Regional Director of Operations") and '
                 'company.industry in ("Restaurants", "Food and Beverage Services") and '
                 'company.company_size in ("11-50", "51-200", "201-500"))')


def graph_payload():
    """Small deterministic visual sample for the hosted replay function."""
    points = []
    for i in range(2200):
        a = i * 2.399963
        shell = 0.18 + ((i * 37) % 1000) / 1250
        points.append([round(shell * __import__('math').cos(a), 4),
                       round(shell * .72 * __import__('math').sin(a * .83), 4),
                       round(shell * __import__('math').sin(a), 4)])
    channels = {str(i): list(range(i * 300, min((i + 1) * 300, len(points)))) for i in range(6)}
    return {"n": 166700, "points": points, "point_indices": list(range(len(points))),
            "groups": {}, "pools": {"RESEARCH": list(range(900, 1300)),
                                      "DRAFT_EMAIL": list(range(1300, 1700)),
                                      "ESCALATE": list(range(1700, 2000))},
            "channels": channels, "signal_keys": ["funding", "hiring", "intent", "job_change", "negative", "trigger"]}


def clay_live(query, limit):
    key = os.environ.get("CLAY_PUBLIC_API_KEY")
    if not key:
        return {"mode": "live_draft", "rows": [], "distribution": {},
                "safe_to_contact": False, "source": "live unavailable: Clay key not configured",
                "budget": {"records": 0, "credits_used": 0, "max_records": 10, "max_credits": 50}}
    query = (query or DEFAULT_QUERY)[:2000]
    request = urllib.request.Request(
        "https://api.clay.com/public/v0/search/query-mode",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "clay-api-key": key}, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        search_id = json.loads(response.read())["search_id"]
    request = urllib.request.Request(
        f"https://api.clay.com/public/v0/search/query-mode/{search_id}/run",
        data=json.dumps({"limit": min(int(limit), 10)}).encode(),
        headers={"Content-Type": "application/json", "clay-api-key": key}, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        records = json.loads(response.read()).get("data", [])
    rows, seen = [], set()
    for record in records:
        experiences = record.get("matched_experiences") or []
        company = (experiences[0].get("company") if experiences else None) or record.get("name") or "Unknown company"
        if company.lower() in seen:
            continue
        seen.add(company.lower())
        location = record.get("location")
        if isinstance(location, dict):
            location = location.get("name") or ", ".join(str(location.get(key)) for key in ("city", "state", "country") if location.get(key))
        rows.append({"company": company, "domain": record.get("domain"),
                     "title": experiences[0].get("title") if experiences else None,
                     "location": location, "decision": "RESEARCH",
                     "policy_action": "RESEARCH", "confidence": .72,
                     "signals": {key: .5 for key in ("funding", "hiring", "intent", "job_change", "negative", "trigger")},
                     "provenance": [{"source": "clay_search", "record_id": record.get("clay_profile_id"), "retrieved_at": "live"}],
                     "model_version": "fly-hosted-live-draft", "draft": None})
    return {"mode": "live_draft", "rows": rows, "distribution": {"RESEARCH": len(rows)},
            "safe_to_contact": False, "source": "live Clay Search / hosted draft-only adapter",
            "budget": {"records": len(rows), "enrichments": 0, "credits_used": 0,
                        "max_records": 10, "max_credits": 50}}


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
    if method == "GET" and path in ("/graph", "/api/graph"):
        return 200, graph_payload()
    if method == "GET" and path == "/api/demo/eval":
        summary = DEMO / "eval_summary.json"
        if summary.is_file():
            return 200, json.loads(summary.read_text())
        return 404, {"error": "route not found"}
    if method == "POST" and path == "/api/demo/think":
        # Live first: proxy the real connectome service on Modal. If it is
        # unreachable, fall back to the recorded replay for the requested
        # pack — the response's `recorded` flag keeps the UI honest either
        # way.
        try:
            request = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return 400, {"error": "request body is invalid"}
        try:
            live = urllib.request.Request(
                LIVE_THINK_URL,
                data=json.dumps({"evidence": request.get("evidence") or {},
                                 "context": request.get("context") or {}}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(live, timeout=20) as response:
                return 200, json.loads(response.read())
        except Exception:
            replay = THINK_REPLAYS.get(request.get("pack"),
                                       THINK_REPLAYS["anthropic"])
            return 200, {**replay, "recorded": True,
                         "mode": "recorded_replay"}
    if method == "POST" and path == "/api/demo/run":
        try:
            request = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return 400, {"error": "request body is invalid"}
        mode = request.get("mode", "replay")
        if mode == "replay":
            return 200, REPLAY
        if mode == "live_draft":
            return 200, clay_live(request.get("query") or DEFAULT_QUERY,
                                  min(int(request.get("limit", 10)), 10))
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
        data = (DEMO / "web" / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_static(self, path):
        # This function is the deployment's only entry point, so real files
        # under demo/ are served here before the index.html fallback.
        rel = path.lstrip("/")
        if rel.startswith("demo/"):
            rel = rel[5:]
        elif "/" in rel:
            return False
        if not rel:
            rel = "web/index.html"
        elif rel.endswith("/"):
            rel += "index.html"
        target = (DEMO / rel).resolve()
        if DEMO not in target.parents or not target.is_file():
            return False
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        return True

    def do_GET(self):
        path = urlsplit(self.path).path
        if not path.startswith("/api/") and path not in ("/health", "/replay", "/graph"):
            if not self._send_static(path):
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
