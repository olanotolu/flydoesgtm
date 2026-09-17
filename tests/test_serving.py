import json
import urllib.request

from serving.normalize import normalize_row, from_request


def test_normalize_raw_row_bounded():
    out = normalize_row({"total_funding_usd": 40_000_000,
                         "open_roles": 22, "pricing_page_views": 8})
    assert set(out) == {"funding", "hiring", "intent", "job_change",
                         "negative", "trigger"}
    assert all(0 <= x <= 1 for x in out.values())


def test_normalize_accepts_flat_request():
    out = from_request({"open_roles": 10})
    assert 0 < out["hiring"] < 1


def test_replay_is_json():
    from demo.scenarios import SCENES
    assert len(SCENES) == 11
    assert SCENES[7]["action"] == "IGNORE"


def test_clay_range_parsing():
    from serving.clay_replay import _range_midpoint, safe_record
    assert _range_midpoint("201-500") == 350.5
    assert _range_midpoint("25M-75M") == 50e6
    assert _range_midpoint({"min": 10, "max": 30}) == 20
    assert _range_midpoint(None) is None
    raw = safe_record({"domain": "clay.com", "name": "Clay",
                       "size": "201-500", "annual_revenue": "25M-75M",
                       "total_funding_amount_range_usd": None})
    assert raw["total_funding_usd"] == 50e6
    assert raw["open_roles"] > 0
