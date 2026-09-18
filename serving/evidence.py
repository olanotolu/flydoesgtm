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
    layoffs = _verified(evidence, "layoffs")
    board = _verified(evidence, "board")
    # Verified distress is evidence *against* the matching market channel,
    # not just a flag on the negative one: a Chapter 11 filer cannot take a
    # priced round, and a company in a mass layoff is not hiring.
    chapter11 = negative and bool(
        (evidence.get("negative") or {}).get("event") == "chapter_11")

    # Unknown account dimensions stay neutral. Each channel moves only on
    # verified evidence of its own kind — a launch is not funding proof.
    signals = {
        "funding": round(min(1.0, 0.5 + 0.25 * funding
                             - 0.20 * chapter11), 4),
        "hiring": round(min(1.0, 0.5 + 0.20 * hiring
                            - 0.20 * layoffs), 4),
        "intent": round(min(1.0, 0.5 + 0.12 * launch + 0.08 * enterprise
                            + 0.05 * pricing + 0.03 * linkedin + 0.02 * x_post), 4),
        "job_change": round(min(1.0, 0.5 + 0.15 * job_change), 4),
        # Corroborating verified distress keys stack onto the negative
        # channel — a Chapter 11 with confirmed layoffs and a board exodus
        # is stronger evidence than a lone negative headline.
        "negative": round(min(1.0, 0.5 + 0.20 * negative
                              + 0.12 * layoffs + 0.08 * board), 4),
        "trigger": round(min(1.0, 0.5 + 0.27 * launch + 0.06 * linkedin
                             + 0.07 * x_post + 0.05 * enterprise
                             + 0.05 * target), 4),
    }

    provenance = [
        {"channel": "funding", "value": signals["funding"],
         "status": "derived" if (funding or chapter11)
         else "unknown_neutral",
         "rule": ("chapter 11 — capital access impaired" if chapter11
                  else "verified funding round" if funding
                  else "no verified funding evidence")},
        {"channel": "hiring", "value": signals["hiring"],
         "status": "derived" if (hiring or layoffs)
         else "unknown_neutral",
         "rule": ("mass layoff — workforce contracting" if layoffs
                  else "verified hiring signal" if hiring
                  else "no verified hiring evidence")},
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
         "status": "derived" if (negative or layoffs or board)
         else "unknown_neutral",
         "rule": "verified negative event + corroborating distress"
         if (negative or layoffs or board)
         else "no verified negative event",
         "inputs": {"negative": negative, "layoffs": layoffs,
                    "board": board}},
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
