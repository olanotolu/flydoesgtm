import json

from serving.think import (build_reasons, build_think_response,
                           crowdedness_of, recommend)


def test_crowdedness_counts_hot_channels():
    energies = {"funding": 0.9, "hiring": 0.8, "intent": 0.85,
                "job_change": 0.5, "negative": 0.1, "trigger": 0.9}
    assert crowdedness_of(energies) == 4


def test_wait_triggers_on_engage_and_crowded():
    assert recommend("RESEARCH", 4) == ("WAIT", 14)
    assert recommend("DRAFT_EMAIL", 5) == ("WAIT", 21)


def test_ignore_never_waits():
    assert recommend("IGNORE", 6) == ("NO", 0)
    assert recommend("IGNORE", 0) == ("NO", 0)


def test_yes_when_engage_and_calm():
    assert recommend("RESEARCH", 3) == ("YES", 0)


def test_reasons_name_top_channels():
    decide_out = {"channel_activity": {7: 2.5, 14: 5.1},
                  "confidence": 0.92, "decision": "RESEARCH"}
    energies = {"funding": 0.9, "hiring": 0.55, "intent": 0.8,
                "job_change": 0.5, "negative": 0.1, "trigger": 0.75}
    lines = build_reasons(energies, decide_out, 3, 0)
    assert lines[0] == "Detected Funding signal at 90%"
    assert lines[1] == "Detected Intent signal at 80%"
    assert any("Deep channels responding: 14 · 7" in line for line in lines)
    assert any("Information value: HIGH" in line for line in lines)


def test_think_response_is_json_serializable():
    import numpy as np
    decide_out = {"probabilities": {"RESEARCH": np.float64(0.92)},
                  "confidence": np.float64(0.92), "decision": "RESEARCH",
                  "channel_activity": {7: np.float64(2.5)},
                  "sim_steps": 4}
    energies = {"funding": 0.9, "hiring": 0.55, "intent": 0.8,
                "job_change": 0.5, "negative": 0.1, "trigger": 0.75}
    body = json.dumps(build_think_response(energies, decide_out))
    assert '"recommendation": "YES"' in body


def test_think_integration_real_brain():
    from serving.serve import FlyService
    svc = FlyService()
    out = svc.decide({"funding": 0.9, "hiring": 0.55, "intent": 0.8,
                      "job_change": 0.5, "negative": 0.1, "trigger": 0.75})
    body = build_think_response(
        {k: 0.5 for k in ("funding", "hiring", "intent", "job_change",
                          "negative", "trigger")}, out)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["recommendation"] in ("YES", "NO", "WAIT")
    assert isinstance(body["reasons"], list) and body["reasons"]
