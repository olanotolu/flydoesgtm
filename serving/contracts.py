"""Stable, provenance-first contracts for the safe Clay demonstration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

SIGNAL_KEYS = ("funding", "hiring", "intent", "job_change", "negative", "trigger")
DEMO_ACTIONS = ("IGNORE", "RESEARCH", "DRAFT_EMAIL", "ESCALATE")
POLICY_TO_DEMO = {"EMAIL": "DRAFT_EMAIL"}
MAX_RECORDS = 10
MAX_ENRICHMENTS = 4
MAX_CREDITS = 50.0
MIN_REMAINING_CREDITS = 300.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Signal:
    name: str
    value: Any
    missing: bool
    source: str
    retrieved_at: str
    confidence: float | None = None
    credit_cost: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Decision:
    action: str
    policy_action: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str
    mode: str
    budget: dict[str, float]
    diagnostic: dict[str, Any] = field(default_factory=dict)
    provenance: list[dict[str, Any]] = field(default_factory=list)
    generated_at: str = field(default_factory=utc_now)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def demo_action(policy_action: str) -> str:
    """Map policy semantics to the only allowed real-data demo semantics."""
    return POLICY_TO_DEMO.get(policy_action, policy_action)


def as_signal_map(raw: dict[str, Any], *, source: str,
                  retrieved_at: str | None = None,
                  costs: dict[str, float] | None = None) -> dict[str, Signal]:
    stamp = retrieved_at or utc_now()
    costs = costs or {}
    return {
        key: Signal(
            name=key,
            value=raw.get(key),
            missing=raw.get(key) is None,
            source=source,
            retrieved_at=stamp,
            confidence=None if raw.get(key) is None else 1.0,
            credit_cost=float(costs.get(key, 0.0)),
        )
        for key in SIGNAL_KEYS
    }
