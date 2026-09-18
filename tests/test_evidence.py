from serving.evidence import adapt_evidence_to_signals


def episode():
    return {
        "news": {"verified": True, "product_launch": True,
                 "enterprise_controls": True, "pricing_change": True},
        "linkedin": {"verified": True},
        "x": {"verified": True, "url": "https://x.com/claudeai/status/2094848572143407483"},
        "target": {"verified": True, "name": "Dario Amodei"},
    }


def test_adapter_is_conservative_and_auditable():
    out = adapt_evidence_to_signals(episode())
    assert out["signals"] == {
        "funding": 0.5, "hiring": 0.5, "intent": 0.8,
        "job_change": 0.5, "negative": 0.5, "trigger": 1.0,
    }
    assert out["excluded"] == []
    assert len(out["provenance"]) == 6


def test_missing_evidence_stays_neutral():
    out = adapt_evidence_to_signals({})
    assert all(value == 0.5 for value in out["signals"].values())


def test_unverified_x_is_excluded_without_becoming_negative():
    data = episode()
    data["x"] = {"verified": False}
    out = adapt_evidence_to_signals(data)
    assert out["excluded"] == [{"source": "x", "reason": "not publicly verified"}]
    assert out["signals"]["negative"] == 0.5


def test_verified_distress_signals_stack_on_negative():
    data = {"negative": {"verified": True}}
    assert adapt_evidence_to_signals(data)["signals"]["negative"] == 0.7
    data["layoffs"] = {"verified": True}
    assert adapt_evidence_to_signals(data)["signals"]["negative"] == 0.82
    data["board"] = {"verified": True}
    assert adapt_evidence_to_signals(data)["signals"]["negative"] == 0.9


def test_unverified_distress_signals_do_not_stack():
    data = {"negative": {"verified": True},
            "layoffs": {"verified": False},
            "board": {"event": "director_exodus"}}
    out = adapt_evidence_to_signals(data)
    assert out["signals"]["negative"] == 0.7
    rule = next(p for p in out["provenance"] if p["channel"] == "negative")
    assert rule["inputs"] == {"negative": True, "layoffs": False, "board": False}


def test_distress_without_negative_still_derives():
    out = adapt_evidence_to_signals({"layoffs": {"verified": True},
                                     "board": {"verified": True}})
    assert out["signals"]["negative"] == 0.7


def test_chapter11_drags_funding_and_layoffs_drags_hiring():
    out = adapt_evidence_to_signals(
        {"negative": {"verified": True, "event": "chapter_11"},
         "layoffs": {"verified": True}})
    sig = out["signals"]
    assert sig["funding"] == 0.3 and sig["hiring"] == 0.3
    assert sig["negative"] == 0.82
    rules = {p["channel"]: p["rule"] for p in out["provenance"]}
    assert "capital access impaired" in rules["funding"]
    assert "workforce contracting" in rules["hiring"]


def test_non_bankruptcy_negative_leaves_market_channels_neutral():
    out = adapt_evidence_to_signals({"negative": {"verified": True,
                                                "event": "lawsuit"}})
    assert out["signals"]["funding"] == 0.5
    assert out["signals"]["hiring"] == 0.5
