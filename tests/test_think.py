import json

from serving.think import (build_reasons, build_think_response,
                           crowdedness_of, recommend)


def test_crowdedness_counts_hot_channels():
    energies = {"funding": 0.9, "hiring": 0.8, "intent": 0.85,
                "job_change": 0.5, "negative": 0.1, "trigger": 0.9}
    assert crowdedness_of(energies) == 4


def test_every_fly_action_maps_to_a_verdict():
    """The fly owns the verdict. A missing action here means the demo
    silently reports one of the fly's decisions as something else."""
    from environment.clay_world import ACTIONS

    engage = {"RESEARCH", "ENRICH", "EMAIL", "ESCALATE"}
    abstain = {"WAIT", "OBSERVE"}
    for action in ACTIONS:
        verdict, _ = recommend(action, 0)
        if action in engage:
            assert verdict == "YES", action
        elif action in abstain:
            assert verdict == "WAIT", action
        else:
            assert verdict == "NO", action


def test_engage_actions_are_yes():
    for action in ("RESEARCH", "ENRICH", "EMAIL", "DRAFT_EMAIL", "ESCALATE"):
        assert recommend(action, 0) == ("YES", 0)


def test_abstentions_are_wait_not_no():
    """WAIT/OBSERVE used to fall through to NO, reporting an abstention
    as a rejection."""
    assert recommend("WAIT", 0) == ("WAIT", 7)
    assert recommend("OBSERVE", 0) == ("WAIT", 7)
    assert recommend("WAIT", 6)[0] != "NO"


def test_ignore_never_waits():
    assert recommend("IGNORE", 6) == ("NO", 0)
    assert recommend("IGNORE", 0) == ("NO", 0)


def test_crowdedness_no_longer_changes_the_verdict():
    """The evidence adapter pins four channels at 0.5, so crowdedness
    cannot reach the old `>= 4` threshold — that branch was dead code."""
    for action in ("RESEARCH", "WAIT", "IGNORE"):
        assert recommend(action, 0) == recommend(action, 6)


def test_yes_when_engage_and_calm():
    assert recommend("RESEARCH", 3) == ("YES", 0)


def test_reasons_name_top_channels():
    decide_out = {"confidence": 0.92, "decision": "RESEARCH",
                  "probabilities": {"RESEARCH": 0.5, "WAIT": 0.3,
                                    "OBSERVE": 0.2},
                  "neuron_activity": {"steps": 12, "dt_ms": 20,
                                      "spikes": 23000, "hz_max": 50}}
    energies = {"funding": 0.9, "hiring": 0.55, "intent": 0.8,
                "job_change": 0.5, "negative": 0.1, "trigger": 0.75}
    lines = build_reasons(energies, decide_out, 3, 0)
    assert lines[0].startswith("4 of 6 evidence channels moved")
    assert "funding 0.90" in lines[0] and "intent 0.80" in lines[0]
    assert any("12 × 20ms — 23,000 spikes, peak 50 Hz" in line
               for line in lines)
    assert any("RESEARCH 50% · WAIT 30%" in line for line in lines)
    assert any("RESEARCH pursues the account → YES" in line
               for line in lines)
    assert any("softmax rank, not a win probability" in line
               for line in lines)


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


def test_think_integration_real_brain(fly_service):
    svc = fly_service
    out = svc.decide({"funding": 0.9, "hiring": 0.55, "intent": 0.8,
                      "job_change": 0.5, "negative": 0.1, "trigger": 0.75})
    body = build_think_response(
        {k: 0.5 for k in ("funding", "hiring", "intent", "job_change",
                          "negative", "trigger")}, out)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["recommendation"] in ("YES", "NO", "WAIT")
    assert isinstance(body["reasons"], list) and body["reasons"]


def test_think_passes_through_neuron_activity():
    decide_out = {
        "channel_activity": {}, "confidence": 0.9, "decision": "RESEARCH",
        "probabilities": {}, "sim_steps": 4,
        "observation": [0.9, 0.5, 0.8, 0.5, 0.1, 0.75] + [1.0, 0.5, 0.0, 0.0,
                                                          1.0, 1.0, 1.0, 1.0,
                                                          1.0, 0.5],
        "neuron_activity": {
            "steps": 4, "dt_ms": 20, "window_ms": 80, "active": 3, "spikes": 6,
            "per_step": [1, 2, 1, 2], "hz_max": 50.0,
            "top": [{"id": 10013, "label": "MBON01", "side": "R",
                     "region": "cb_intrinsic", "spikes": 3}],
        },
    }
    signals = {"funding": 0.9, "hiring": 0.5, "intent": 0.8,
               "job_change": 0.5, "negative": 0.1, "trigger": 0.75}
    body = build_think_response(signals, decide_out)
    assert body["neuron_activity"]["active"] == 3
    assert body["neuron_activity"]["top"][0]["label"] == "MBON01"
    assert body["sees"]["total"] == 16
    assert body["sees"]["lit"] == sum(1 for v in decide_out["observation"] if v > 0.5)


def test_think_sees_defaults_when_observation_missing():
    body = build_think_response(
        {"funding": 0.9}, {"decision": "RESEARCH", "confidence": 0.9})
    assert body["sees"]["total"] == 16
    assert "neuron_activity" in body
