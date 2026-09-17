"""Lossless extraction of public Clay search records for Fly inputs."""
from __future__ import annotations

import re


def range_midpoint(value):
    """Parse Clay ranges without inventing a value for null."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        vals = [v for v in (value.get("min"), value.get("max"))
                if v is not None]
        return sum(float(v) for v in vals) / len(vals) if vals else None
    nums = []
    for n, suffix in re.findall(r"(\d+(?:\.\d+)?)\s*([kmb])?",
                                str(value).lower()):
        x = float(n) * {"k": 1e3, "m": 1e6, "b": 1e9}.get(suffix, 1.0)
        nums.append(x)
    return sum(nums) / len(nums) if nums else None


def safe_record(record):
    """Extract only values actually returned by Clay.

    Missing signals stay missing; revenue is never relabeled as funding and
    headcount is never relabeled as hiring.
    """
    experiences = record.get("matched_experiences") or []
    experience = experiences[0] if experiences else {}
    location = record.get("location") or experience.get("location")
    if isinstance(location, dict):
        location = location.get("name") or ", ".join(
            str(location.get(key)) for key in ("city", "state", "country")
            if location.get(key)
        ) or None
    raw = {
        "domain": record.get("domain"),
        "company": experience.get("company") or record.get("name"),
        "contact_name": record.get("name"),
        "first_name": record.get("first_name"),
        "title": experience.get("title"),
        "location": location,
        "industry": record.get("industry"),
    }
    funding = range_midpoint(record.get("total_funding_amount_range_usd"))
    if funding is not None:
        raw["total_funding_usd"] = funding
    if record.get("latest_raise_usd") is not None:
        raw["latest_raise_usd"] = range_midpoint(record["latest_raise_usd"])
    if record.get("open_roles") is not None:
        raw["open_roles"] = record["open_roles"]
    if record.get("headcount_growth_pct") is not None:
        raw["headcount_growth_pct"] = record["headcount_growth_pct"]
    return raw
