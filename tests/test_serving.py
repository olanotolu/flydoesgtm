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
    assert len(SCENES) == 7
    assert all(scene.get("action") != "EMAIL" for scene in SCENES)


def test_clay_range_parsing():
    from serving.clay_replay import _range_midpoint, safe_record
    assert _range_midpoint("201-500") == 350.5
    assert _range_midpoint("25M-75M") == 50e6
    assert _range_midpoint({"min": 10, "max": 30}) == 20
    assert _range_midpoint(None) is None
    raw = safe_record({"domain": "clay.com", "name": "Clay",
                       "size": "201-500", "annual_revenue": "25M-75M",
                       "total_funding_amount_range_usd": None})
    assert "total_funding_usd" not in raw
    assert "open_roles" not in raw


def test_missing_signals_are_neutral_not_false_zeroes():
    out = normalize_row({"name": "Clay"})
    assert all(value == 0.5 for value in out.values())


def test_real_fields_are_preserved_without_inference():
    from serving.clay_records import safe_record
    raw = safe_record({
        "name": "Example Restaurants",
        "domain": "example.test",
        "industry": "Restaurants",
        "open_roles": 7,
        "total_funding_amount_range_usd": "1M-5M",
    })
    assert raw["total_funding_usd"] == 3e6
    assert raw["open_roles"] == 7


def test_resolve_static_asset_guards_traversal():
    from serving.serve import resolve_static_asset
    assert resolve_static_asset("/nope.png") is None
    assert resolve_static_asset("/../serve.py") is None
    assert resolve_static_asset("/demo/web/index.html") is None
    assert resolve_static_asset("/api/demo/run") is None


def test_resolve_static_asset_finds_demo_png():
    from serving.serve import resolve_static_asset
    found = resolve_static_asset("/clay-logo.png")
    assert found is not None and found.name == "clay-logo.png"
    found_demo = resolve_static_asset("/demo/clay-logo.png")
    assert found_demo is not None and found_demo.name == "clay-logo.png"
    assert resolve_static_asset("/demo/../serve.py") is None
    assert resolve_static_asset("/demo/sub/x.png") is None


def test_resolve_static_asset_finds_connectome_javascript():
    from serving.serve import resolve_static_asset
    found = resolve_static_asset("/malecns-soma-sample.js")
    assert found is not None and found.name == "malecns-soma-sample.js"


def test_decide_reports_real_neuron_activity(fly_service):
    svc = fly_service
    out = svc.decide({"funding": 0.97, "trigger": 0.92, "intent": 0.88,
                      "hiring": 0.55, "job_change": 0.18, "negative": 0.04})
    na = out["neuron_activity"]
    # Assert the reported window is self-consistent and matches what the
    # service says it served, rather than pinning a literal that goes stale
    # every time the horizon changes.
    served = svc.health()["sim_steps"]
    assert na["steps"] == served
    assert na["dt_ms"] == 20 and na["window_ms"] == 20 * served
    assert na["active"] >= 1 and na["spikes"] >= na["active"]
    assert sum(na["per_step"]) == na["spikes"]
    assert na["hz_max"] > 0
    first = na["top"][0]
    assert isinstance(first["id"], int) and first["label"]
    assert first["spikes"] >= na["top"][-1]["spikes"]
    assert len(out["observation"]) == 16
    assert len(na["frames"]) == na["steps"]
    assert [len(f) for f in na["frames"]] == na["per_step"]
    assert all(isinstance(n, int) for frame in na["frames"] for n in frame)
