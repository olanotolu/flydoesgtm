"""Pure think-layer: crowdedness, recommendation, reasons. No I/O, no brain."""
from __future__ import annotations

from typing import Any

from environment.signals import CHANNEL_NAMES

SIGNAL_ORDER = ("funding", "hiring", "intent", "job_change",
                "negative", "trigger")
LABELS = {"funding": "Funding", "hiring": "Hiring", "intent": "Intent",
          "job_change": "Job change", "negative": "Negative",
          "trigger": "Trigger"}
ENGAGE = ("RESEARCH", "ENRICH", "EMAIL", "DRAFT_EMAIL", "ESCALATE")
ABSTAIN = ("WAIT", "OBSERVE")


def clamp_energies(raw: dict) -> dict[str, float]:
    return {key: min(1.0, max(0.0, float(raw.get(key, 0.5))))
            for key in SIGNAL_ORDER}


def crowdedness_of(energies: dict) -> int:
    """How many channels the evidence pushed above neutral.

    Reported for context only. This used to gate the verdict, but
    `serving/evidence.py` pins funding, hiring, job_change and negative at
    a neutral 0.5, so crowdedness cannot exceed 2 and the old `>= 4`
    branch was unreachable. See `recommend`.
    """
    return sum(1 for key in SIGNAL_ORDER if energies.get(key, 0.0) > 0.6)


def recommend(decision: str, crowdedness: int) -> tuple[str, int]:
    """Map the fly's action onto the demo's three verdicts.

    The fly owns the verdict. Two bugs lived here:

    1. `ENGAGE` omitted ENRICH and EMAIL, so a fly that decided to email
       the account was reported to the user as NO.
    2. The fly's own WAIT and OBSERVE also fell through to NO, so an
       abstention was reported as a rejection — while the only path to
       WAIT was a crowdedness threshold the evidence adapter cannot reach.

    Every action the policy can emit now maps somewhere deliberate:

        RESEARCH ENRICH EMAIL DRAFT_EMAIL ESCALATE -> YES  (pursue)
        WAIT OBSERVE                               -> WAIT (not yet)
        IGNORE                                     -> NO   (write off)

    `crowdedness` is kept for signature compatibility and no longer
    changes the verdict.
    """
    if decision in ENGAGE:
        return "YES", 0
    if decision in ABSTAIN:
        return "WAIT", 7
    return "NO", 0


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    convert = getattr(value, "tolist", None)
    if callable(convert):
        return convert()
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _top_inputs(energies: dict) -> list[tuple[str, float]]:
    return sorted(((key, energies.get(key, 0.0)) for key in SIGNAL_ORDER),
                  key=lambda item: item[1], reverse=True)


def _deep_channels(decide_out: dict) -> list[int]:
    activity = decide_out.get("channel_activity") or {}
    pairs = [(int(key), float(value)) for key, value in activity.items()]
    return [key for key, _ in sorted(pairs, key=lambda item: item[1],
                                     reverse=True) if _ > 0][:4]


VERDICT_WHY = {"YES": "pursues the account",
               "WAIT": "holds the account without acting",
               "NO": "writes the account off"}


def build_reasons(energies: dict, decide_out: dict, crowdedness: int,
                  wait_days: int) -> list[str]:
    """Narrate what actually happened, in the order it happened:

    evidence values -> connectome activity -> readout scores -> verdict.
    No internal jargon: channel names are the adapter's own words, spike
    counts come from the recorded simulation, and the last two lines say
    exactly how an action index becomes a YES/WAIT/NO and what the
    confidence number does and does not mean.
    """
    moved = [(LABELS[key].lower(), value)
             for key, value in _top_inputs(energies)
             if abs(value - 0.5) > 0.1]
    if moved:
        lines = [f"{len(moved)} of {len(SIGNAL_ORDER)} evidence channels "
                 "moved off neutral — " +
                 " · ".join(f"{name} {value:.2f}" for name, value in moved)]
        neutral = len(SIGNAL_ORDER) - len(moved)
        if neutral:
            lines.append(f"{neutral} stayed neutral — no verified evidence "
                         "either way")
    else:
        lines = ["evidence flat — every channel reads neutral"]

    na = decide_out.get("neuron_activity") or {}
    steps = int(na.get("steps") or decide_out.get("sim_steps") or 0)
    if steps and na.get("spikes") is not None:
        lines.append(f"connectome ran {steps} × {na.get('dt_ms', 20):g}ms — "
                     f"{int(na['spikes']):,} spikes, "
                     f"peak {float(na.get('hz_max', 0)):g} Hz")

    probs = sorted(((action, float(p)) for action, p in
                    (decide_out.get("probabilities") or {}).items()),
                   key=lambda item: item[1], reverse=True)
    if probs:
        lines.append("readout scored " + " · ".join(
            f"{action} {round(p * 100)}%" for action, p in probs[:3]))

    decision = str(decide_out.get("decision", "—"))
    recommendation, _ = recommend(decision, crowdedness)
    lines.append(f"{decision} {VERDICT_WHY[recommendation]} "
                 f"→ {recommendation}")
    lines.append("confidence is a softmax rank, not a win probability")
    return lines


def build_think_response(signals: dict, decide_out: dict) -> dict:
    energies = clamp_energies(signals)
    crowdedness = crowdedness_of(energies)
    decision = str(decide_out.get("decision", "RESEARCH"))
    recommendation, wait_days = recommend(decision, crowdedness)
    observation = decide_out.get("observation") or []
    slots = [bool(float(v) > 0.5) for v in observation][:16]
    slots += [False] * (16 - len(slots))
    sees = {"lit": sum(slots), "total": 16, "slots": slots}
    return {
        "energies": energies,
        "probabilities": _jsonable(decide_out.get("probabilities") or {}),
        "confidence": float(decide_out.get("confidence", 0.5)),
        "decision": decision,
        "deep_channels": _deep_channels(decide_out),
        "crowdedness": crowdedness,
        "recommendation": recommendation,
        "wait_days": wait_days,
        "reasons": build_reasons(energies, decide_out, crowdedness,
                                 wait_days),
        "sim_steps": int(decide_out.get("sim_steps", 0)),
        "neuron_activity": decide_out.get("neuron_activity") or {},
        "sees": sees,
        "provenance": _jsonable(decide_out.get("provenance") or []),
        "sources": _jsonable(decide_out.get("sources") or []),
        "excluded": _jsonable(decide_out.get("excluded") or []),
        "connectome": decide_out.get("connectome"),
        "model_version": decide_out.get("model_version"),
    }
