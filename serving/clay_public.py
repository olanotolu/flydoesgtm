"""Minimal Clay Public API client for optional live account lookup.

Training and the demo do not depend on this module. It is only called
when CLAY_API_KEY is present; errors fall back to public signals.
"""
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://api.clay.com/public/v0"


def _key():
    if os.environ.get("CLAY_PUBLIC_API_KEY"):
        return os.environ["CLAY_PUBLIC_API_KEY"]
    if os.environ.get("CLAY_API_KEY"):
        return os.environ["CLAY_API_KEY"]
    for name in (".env.local", ".env"):
        p = Path(__file__).parents[1] / name
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith(("CLAY_PUBLIC_API_KEY=", "CLAY_API_KEY=")):
                    return line.split("=", 1)[1].strip()
    return None


class ClayError(Exception):
    pass


def _post(path, payload, key):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "clay-api-key": key},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            msg = json.loads(body).get("message", body[:200])
        except ValueError:
            msg = body[:200]
        raise ClayError(f"Clay {e.code}: {msg}") from None


def search_company(domain, limit=1):
    key = _key()
    if not key:
        raise ClayError("CLAY_PUBLIC_API_KEY is not set")
    sid = _post("/search/query-mode",
                 {"query": f'select from companies where domain = "{domain}"'},
                 key)["search_id"]
    return _post(f"/search/query-mode/{sid}/run", {"limit": limit}, key) \
        .get("data", [])


def search_records(query, limit=12):
    """General read-only SQL-grammar search for people or companies."""
    key = _key()
    if not key:
        raise ClayError("CLAY_PUBLIC_API_KEY is not set")
    q = query.strip()
    if not q:
        q = "select from companies"
    elif not q.lower().startswith("select"):
        # ponytail: no '=' means free text — route it to a description
        # match instead of 400ing on Clay's SQL grammar
        q = f"select from companies where {q}" if "=" in q \
            else ('select from companies where description = "'
                  + q.replace('"', "") + '"')
    sid = _post("/search/query-mode", {"query": q}, key)["search_id"]
    return _post(f"/search/query-mode/{sid}/run", {"limit": int(limit)},
                 key).get("data", [])


def search_companies(query, limit=12):
    """Backward-compatible alias for the generic read-only search."""
    return search_records(query, limit)
