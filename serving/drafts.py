"""Unsent GTM artifacts produced from sourced Clay signals only."""
from __future__ import annotations

from typing import Any


def draft_artifact(record: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    company = record.get("company") or "your group"
    first_name = record.get("first_name") or "there"
    signal = record.get("verified_signal") or "a recent operating signal"
    return {
        "status": "UNSENT DRAFT",
        "recipient": {"name": first_name, "email": None},
        "company": company,
        "sequence": [
            {
                "step": 1,
                "delay_days": 0,
                "subject": f"A question about {company}'s operating load",
                "body": (
                    f"Hi {first_name} — I noticed {signal} at {company}. "
                    "Concya acts as an autonomous general manager across guest "
                    "calls, coverage, vendors, and recovery while keeping "
                    "approvals and an audit trail. Would a 15-minute walkthrough "
                    "of one operating workflow be useful?\n\n"
                    "Best,\n{{sender_name}}"
                ),
            },
            {
                "step": 2,
                "delay_days": 4,
                "subject": None,
                "body": (
                    "Worth closing the loop? I can map one workflow—missed guest "
                    "calls, shift coverage, or vendor follow-up—to your current "
                    "stack and show exactly what stays behind approval.\n\n"
                    "Best,\n{{sender_name}}"
                ),
            },
        ],
        "policy_action": decision.get("policy_action", decision.get("decision")),
        "provenance": decision.get("provenance", []),
    }
