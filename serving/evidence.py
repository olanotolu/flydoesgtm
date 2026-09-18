"""Evidence records -> the six GTM channels understood by the fly policy.

This module is deliberately conservative: missing evidence is neutral (0.5),
not zero, and unavailable social channels never become negative evidence.
Every derived value carries the inputs and rule that produced it.
"""
from __future__ import annotations

from typing import Any

SIGNAL_KEYS = ("funding", "hiring", "intent", "job_change", "negative", "trigger")


def _verified(evidence: dict[str, Any], key: str) -> bool:
    item = evidence.get(key) or {}
    return bool(item.get("verified"))


def adapt_evidence_to_signals(evidence: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be an object")

    launch = _verified(evidence, "news") and bool(
        (evidence.get("news") or {}).get("product_launch"))
    linkedin = _verified(evidence, "linkedin")
    x_post = _verified(evidence, "x")
    enterprise = _verified(evidence, "news") and bool(
        (evidence.get("news") or {}).get("enterprise_controls"))
    pricing = _verified(evidence, "news") and bool(
        (evidence.get("news") or {}).get("pricing_change"))
    target = _verified(evidence, "target")
    funding = _verified(evidence, "funding")
    hiring = _verified(evidence, "hiring")
    job_change = _verified(evidence, "job_change")
    negative = _verified(evidence, "negative")

    # Unknown account dimensions stay neutral. Each channel moves only on
    # verified evidence of its own kind — a launch is not funding proof.
    signals = {
        "funding": round(min(1.0, 0.5 + 0.25 * funding), 4),
        "hiring": round(min(1.0, 0.5 + 0.20 * hiring), 4),
        "intent": round(min(1.0, 0.5 + 0.12 * launch + 0.08 * enterprise
                            + 0.05 * pricing + 0.03 * linkedin + 0.02 * x_post), 4),
        "job_change": round(min(1.0, 0.5 + 0.15 * job_change), 4),
        "negative": round(min(1.0, 0.5 + 0.20 * negative), 4),
        "trigger": round(min(1.0, 0.5 + 0.27 * launch + 0.06 * linkedin
                             + 0.07 * x_post + 0.05 * enterprise
                             + 0.05 * target), 4),
    }

    provenance = [
        {"channel": "funding", "value": signals["funding"],
         "status": "derived" if funding else "unknown_neutral",
         "rule": "verified funding round" if funding
         else "no verified funding evidence"},
        {"channel": "hiring", "value": signals["hiring"],
         "status": "derived" if hiring else "unknown_neutral",
         "rule": "verified hiring signal" if hiring
         else "no verified hiring evidence"},
        {"channel": "intent", "value": signals["intent"],
         "status": "derived", "rule": "launch + enterprise controls + pricing + LinkedIn + X",
         "inputs": {"launch": launch, "enterprise_controls": enterprise,
                    "pricing_change": pricing, "linkedin": linkedin,
                    "x": x_post}},
        {"channel": "job_change", "value": signals["job_change"],
         "status": "derived" if job_change else "unknown_neutral",
         "rule": "verified champion job change" if job_change
         else "no verified champion job change"},
        {"channel": "negative", "value": signals["negative"],
         "status": "derived" if negative else "unknown_neutral",
         "rule": "verified negative event" if negative
         else "no verified negative event"},
        {"channel": "trigger", "value": signals["trigger"],
         "status": "derived", "rule": "verified launch + social confirmation + target",
         "inputs": {"launch": launch, "linkedin": linkedin, "x": x_post,
                    "enterprise_controls": enterprise, "target": target}},
    ]

    x_item = evidence.get("x") or {}
    sources = [dict(item, key=key) for key, item in evidence.items()
               if isinstance(item, dict)]
    return {
        "signals": signals,
        "provenance": provenance,
        "sources": sources,
        "excluded": [{"source": "x", "reason": "not publicly verified"}]
        if not x_item.get("verified") else [],
    }
