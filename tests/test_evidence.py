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
