"""Legacy scene adapter kept for GIF tooling.

The product UI now loads the provenance-bearing ``demo/replay.json`` rows
directly; this adapter does not invent outreach or outcome claims.
"""
import json
from pathlib import Path

SCENES = [
    {"at": 0.0, "kind": "title", "text": "recorded Clay search capture"},
    {"at": 1.0, "kind": "account", "company": "Feastables", "action": "RESEARCH"},
    {"at": 4.0, "kind": "account", "company": "Big Chicken", "action": "RESEARCH"},
    {"at": 7.0, "kind": "account", "company": "Marcus Samuelsson Group", "action": "RESEARCH"},
    {"at": 10.0, "kind": "account", "company": "Slutty Vegan ATL", "action": "RESEARCH"},
    {"at": 13.0, "kind": "account", "company": "Restaurant HR Group, Inc.", "action": "RESEARCH"},
    {"at": 16.0, "kind": "title", "text": "all outputs remain unsent"},
]


def write(path="demo/replay.json"):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"duration": 18, "scenes": SCENES}, indent=2))
    return p


if __name__ == "__main__":
    print(write())
