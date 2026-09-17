"""Read-only Clay account replay.

Usage:
  CLAY_API_KEY=... python -m serving.clay_replay acme.com

It searches/enriches one company and emits the fly recommendation. It
never sends email or runs a write-capable Clay action.
"""
import json
import re
import sys

from serving.clay_public import search_company
from serving.normalize import normalize_row
from serving.serve import FlyService


def _range_midpoint(value):
    """Clay returns ranges: '201-500', '25M-75M', {'min':..,'max':..}, or null."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        vals = [v for v in (value.get("min"), value.get("max"))
                if v is not None]
        return sum(float(v) for v in vals) / len(vals) if vals else None
    nums = []
    for n, suffix in re.findall(
            r"(\d+(?:\.\d+)?)\s*([kmb])?", str(value).lower()):
        x = float(n)
        x *= {"k": 1e3, "m": 1e6, "b": 1e9}.get(suffix, 1.0)
        nums.append(x)
    return sum(nums) / len(nums) if nums else None


def safe_record(record):
    raw = {"domain": record.get("domain"), "company": record.get("name")}
    funding = _range_midpoint(record.get("total_funding_amount_range_usd"))
    revenue = _range_midpoint(record.get("annual_revenue"))
    # search results often have a null funding range; revenue is the honest
    # scale proxy in that case, labeled as such
    raw["total_funding_usd"] = funding if funding is not None \
        else (revenue or 0)
    employees = _range_midpoint(record.get("size"))
    if employees:
        # ponytail: headcount is a coarse hiring proxy; the real wiring maps
        # an open-roles enrichment instead
        raw["open_roles"] = employees * 0.04
    return raw


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
