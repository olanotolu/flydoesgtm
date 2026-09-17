"""Read-only Clay account replay.

Usage:
  CLAY_API_KEY=... python -m serving.clay_replay acme.com

It searches/enriches one company and emits the fly recommendation. It
never sends email or runs a write-capable Clay action.
"""
import json
import sys

from serving.clay_records import range_midpoint, safe_record
from serving.clay_public import search_company
from serving.normalize import normalize_row
from serving.serve import FlyService


_range_midpoint = range_midpoint


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m serving.clay_replay DOMAIN")
    rows = search_company(sys.argv[1])
    if not rows:
        raise SystemExit("Clay returned no company")
    raw = safe_record(rows[0])
    signals = normalize_row(raw)
    decision = FlyService().decide(signals)
    print(json.dumps({
        "company": raw.get("company"),
        "domain": raw.get("domain"),
        "normalized_signals": signals,
        "fly_decision": decision["decision"],
        "confidence": decision["confidence"],
        "safe_to_contact": False,
        "mode": "read-only replay",
    }, indent=2))


if __name__ == "__main__":
    main()
