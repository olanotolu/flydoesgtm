"""Runnable MVP check: engine, persistence, export, and HTTP API."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TMP = Path(tempfile.mkdtemp(prefix="clay-entropy-check-"))
os.environ["CLAY_ENTROPY_DB"] = str(TMP / "test.sqlite3")
os.environ["CLAY_ENTROPY_EXPORTS"] = str(TMP / "exports")
sys.path.insert(0, str(ROOT))

from entropy_engine import SAMPLES, evaluate  # noqa: E402
from app import Handler  # noqa: E402
import store  # noqa: E402


def request(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req) as res:
        return json.loads(res.read())


by_id = {sample["id"]: sample["input"] for sample in SAMPLES}
assert evaluate(by_id["funding"])["recommended_action"] == "relic"
assert evaluate(by_id["launch"])["recommended_action"] == "relic"
assert evaluate(by_id["acquisition"])["recommended_action"] == "relic"
assert evaluate(by_id["expansion"])["recommended_action"] == "relic"
assert evaluate(by_id["weak"])["recommended_action"] == "wait"

no_source = evaluate({**by_id["funding"], "source_urls": []})
assert no_source["recommended_action"] == "wait"
assert "public_and_attributable" in no_source["signal_assessment"]["failed_checks"]

thin = evaluate({**by_id["launch"], "signal_text": "Northbeam launched something.", "known_context": ""})
assert thin["recommended_action"] == "wait"
assert "context_sufficiency" in thin["signal_assessment"]["failed_checks"]

templated = evaluate({
    **by_id["funding"],
    "known_context": by_id["funding"]["known_context"] + " We already sent the same artifact template to twelve founders.",
})
assert templated["recommended_action"] == "digital"
assert "novelty_preservation" in templated["signal_assessment"]["failed_checks"]

record = store.create(by_id["funding"], evaluate(by_id["funding"]))
assert record["status"] == "pending_review"
record = store.set_status(record["id"], "approved_relic")
package = store.attach_package(record["id"], __import__("artifact").export_package(record, store.EXPORT_ROOT))
assert package["status"] == "exported"
for filename in ("artifact.stl", "decision.json", "provenance_card.md", "manifest.json"):
    assert (Path(package["package_dir"]) / filename).exists()
assert (Path(package["package_dir"]) / "artifact.stl").read_text().startswith("solid Threshold_Contour")

server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
base = f"http://127.0.0.1:{server.server_address[1]}"
try:
    samples = request(f"{base}/api/samples")["samples"]
    assert len(samples) == 5
    preview = request(f"{base}/api/evaluate", by_id["funding"])
    assert preview["decision"]["recommended_action"] == "relic"
    saved = request(f"{base}/api/records", by_id["funding"])["record"]
    reviewed = request(f"{base}/api/records/{saved['id']}/review", {"status": "approved_relic"})["record"]
    assert reviewed["status"] == "approved_relic"
    exported = request(f"{base}/api/records/{saved['id']}/export", {})["record"]
    assert exported["status"] == "exported"
    assert (Path(exported["package_dir"]) / "artifact.stl").exists()
finally:
    server.shutdown()
    server.server_close()

print("clay entropy MVP checks passed")
