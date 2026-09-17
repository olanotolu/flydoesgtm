"""Safe Clay routine adapter for the live-draft demonstration.

This module intentionally has no send, enrollment, reply, webhook, CRM
mutation, or campaign activation operation. Clay routines are invoked only
for read/enrichment work and every call is bounded by a local budget.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

from serving.contracts import (MAX_CREDITS, MAX_ENRICHMENTS,
                               MIN_REMAINING_CREDITS, MAX_RECORDS)

SAFE_RESEARCH_ROUTINES = {
    "Enrich Company",
    "Company Job Openings",
    "Company News",
    "Website Technology Stack",
    "Website Traffic",
}
FORBIDDEN_ACTION_WORDS = (
    "send", "reply", "enroll", "campaign", "webhook", "crm", "update",
    "delete", "remove", "pause",
)


@dataclass
class DemoBudget:
    records: int = 0
    enrichments: int = 0
    credits: float = 0.0
    max_records: int = MAX_RECORDS
    max_enrichments: int = MAX_ENRICHMENTS
    max_credits: float = MAX_CREDITS
    min_remaining_credits: float = MIN_REMAINING_CREDITS
    events: list[dict[str, Any]] = field(default_factory=list)
    enrichments_by_record: dict[str, int] = field(default_factory=dict)

    def admit_record(self):
        if self.records >= self.max_records:
            raise RuntimeError("live draft limit reached: 10 records")
        self.records += 1

    def admit_enrichment(self, routine_name: str, record_id: str = "default"):
        if routine_name not in SAFE_RESEARCH_ROUTINES:
            raise RuntimeError(f"routine is not approved for draft demo: {routine_name}")
        key = str(record_id)
        used = self.enrichments_by_record.get(key, 0)
        if used >= self.max_enrichments:
            raise RuntimeError("live draft limit reached: 4 enrichments per record")
        self.enrichments += 1
        self.enrichments_by_record[key] = used + 1

    def record(self, *, source: str, routine: str | None = None,
               cost: float = 0.0, status: str = "complete"):
        projected = self.credits + float(cost)
        if projected > self.max_credits:
            raise RuntimeError("live draft credit budget exceeded")
        self.credits = projected
        self.events.append({
            "source": source,
            "routine": routine,
            "cost": float(cost),
            "status": status,
        })

    def as_dict(self, remaining: float | None = None):
        return {
            "records": self.records,
            "enrichments": self.enrichments,
            "credits_used": round(self.credits, 4),
            "credits_remaining": None if remaining is None else float(remaining),
            "max_records": self.max_records,
            "max_enrichments": self.max_enrichments,
            "max_credits": self.max_credits,
            "min_remaining_credits": self.min_remaining_credits,
            "enrichments_by_record": dict(self.enrichments_by_record),
            "events": self.events,
        }


def _run_cli(args: list[str], *, timeout: int) -> dict[str, Any]:
    """Run the installed Clay CLI without a shell or interpolated commands."""
    proc = subprocess.run(
        args, capture_output=True, text=True, timeout=timeout, check=False,
        env={**os.environ, "CLAYFLY_LIVE": "1"},
    )
    if proc.returncode:
        detail = proc.stderr.strip()[-500:] or "Clay CLI call failed"
        raise RuntimeError(detail)
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("Clay CLI returned invalid JSON") from exc


def _routine_id():
    routine = os.environ.get("CLAYFLY_ROUTINE_ID", "").strip()
    if not routine:
        raise RuntimeError("set CLAYFLY_ROUTINE_ID to the NO SEND workflow routine")
    lower = routine.lower()
    if any(word in lower for word in FORBIDDEN_ACTION_WORDS):
        raise RuntimeError("configured routine is not permitted for draft mode")
    return routine


def credit_balance(timeout=20) -> dict[str, Any]:
    """Read-only workspace preflight used before any managed enrichment."""
    return _run_cli(["clay", "credits", "balance"], timeout=timeout)


def run_routine(action: str, row: dict[str, Any], *, live=False,
                budget: DemoBudget | None = None, timeout=60):
    """Run one safe enrichment routine or return a deterministic dry-run."""
    if action not in ("RESEARCH", "ENRICH"):
        raise RuntimeError("draft mode only runs read/enrichment routines")
    budget = budget or DemoBudget()
    budget.admit_enrichment(
        str(row.get("routine_name", "Enrich Company")),
        str(row.get("record_id") or row.get("domain") or row.get("company") or "record"),
    )
    if not live:
        budget.record(source="demo", routine="dry-run", status="simulated")
        return {"mode": "replay", "status": "simulated", "result": row}
    routine = _routine_id()
    payload = {"items": [{"id": str(row.get("domain") or row.get("company") or "record"),
                          "inputs": {k: v for k, v in row.items()
                                     if k != "routine_name"}}]}
    started = _run_cli(
        ["clay", "routines", "runs", "start", routine,
         "--input", json.dumps(payload)], timeout=timeout)
    run_id = started.get("routineRunId")
    if not run_id:
        raise RuntimeError("Clay did not return a routine run id")
    result = _run_cli(
        ["clay", "routines", "runs", "get", run_id, "--wait", str(timeout)],
        timeout=timeout + 10,
    )
    if result.get("status") != "complete":
        raise RuntimeError(f"Clay routine did not complete: {result.get('status')}")
    budget.record(source="clay", routine=routine,
                  cost=float(result.get("creditsConsumed", 0.0)))
    return {"mode": "live_draft", "status": "complete", "run_id": run_id,
            "result": result}
