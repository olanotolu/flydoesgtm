"""Pure think-layer: crowdedness, recommendation, reasons. No I/O, no brain."""
from __future__ import annotations

from typing import Any

SIGNAL_ORDER = ("funding", "hiring", "intent", "job_change",
                "negative", "trigger")
LABELS = {"funding": "Funding", "hiring": "Hiring", "intent": "Intent",
          "job_change": "Job change", "negative": "Negative",
          "trigger": "Trigger"}
ENGAGE = ("RESEARCH", "DRAFT_EMAIL", "ESCALATE")


def clamp_energies(raw: dict) -> dict[str, float]:
    return {key: min(1.0, max(0.0, float(raw.get(key, 0.5))))
            for key in SIGNAL_ORDER}


def crowdedness_of(energies: dict) -> int:
    return sum(1 for key in SIGNAL_ORDER if energies.get(key, 0.0) > 0.6)


def recommend(decision: str, crowdedness: int) -> tuple[str, int]:
    if decision in ENGAGE and crowdedness >= 4:
        return "WAIT", 7 * (crowdedness - 2)
    if decision in ENGAGE:
        return "YES", 0
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


def build_reasons(energies: dict, decide_out: dict, crowdedness: int,
                  wait_days: int) -> list[str]:
    top = _top_inputs(energies)[:2]
    lines = [f"Detected {LABELS[key]} signal at {round(value * 100)}%"
             for key, value in top]
    deep = _deep_channels(decide_out)
    if deep:
        lines.append("Deep channels responding: " +
                     " · ".join(str(key) for key in deep))
    confidence = float(decide_out.get("confidence", 0.5))
    value = ("HIGH" if confidence >= 0.75 else
             "MEDIUM" if confidence >= 0.5 else "LOW")
    lines.append(f"Information value: {value}")
    if crowdedness >= 4:
        lines.append(f"{crowdedness} coincident hot signals — "
                     "attention contested")
    lines.append(f"Recommended action: {decide_out.get('decision', '—')}")
    if wait_days:
        lines.append(f"Suggested wait: {wait_days} days")
    return lines


def build_think_response(signals: dict, decide_out: dict) -> dict:
    energies = clamp_energies(signals)
    crowdedness = crowdedness_of(energies)
    decision = str(decide_out.get("decision", "RESEARCH"))
    recommendation, wait_days = recommend(decision, crowdedness)
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
    }
