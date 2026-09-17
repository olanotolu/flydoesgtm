"""SQLite persistence for Clay Entropy decision records."""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("CLAY_ENTROPY_DB", ROOT / "clay_entropy.sqlite3"))
EXPORT_ROOT = Path(os.environ.get("CLAY_ENTROPY_EXPORTS", ROOT / "exports"))

REVIEW_STATES = {
    "pending_review",
    "approved_relic",
    "digital_selected",
    "wait_selected",
    "rejected",
    "exported",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL,
            input_json TEXT NOT NULL,
            decision_json TEXT NOT NULL,
            package_dir TEXT
        )
        """
    )
    return conn


def create(input_data: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    record_id = uuid.uuid4().hex[:12]
    now = utcnow()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO records
            (id, created_at, updated_at, status, input_json, decision_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                now,
                now,
                "pending_review",
                json.dumps(input_data, sort_keys=True),
                json.dumps(decision, sort_keys=True),
            ),
        )
    return get(record_id)


def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    decision = json.loads(row["decision_json"])
    input_data = json.loads(row["input_json"])
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "status": row["status"],
        "package_dir": row["package_dir"],
        "input": input_data,
        "decision": decision,
        "account_name": input_data.get("account_name", ""),
        "signal_type": input_data.get("signal_type", ""),
        "recommended_action": decision.get("recommended_action", ""),
        "artifact_name": decision.get("artifact_concept", {}).get("name", ""),
    }


def get(record_id: str) -> dict[str, Any]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
    if not row:
        raise KeyError(f"record not found: {record_id}")
    return _row_to_record(row)


def list_records() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM records
            ORDER BY created_at DESC, id DESC
            """
        ).fetchall()
    return [_row_to_record(row) for row in rows]


def set_status(record_id: str, status: str) -> dict[str, Any]:
    if status not in REVIEW_STATES:
        raise ValueError(f"unknown status: {status}")
    get(record_id)
    with connect() as conn:
        conn.execute(
            "UPDATE records SET status = ?, updated_at = ? WHERE id = ?",
            (status, utcnow(), record_id),
        )
    return get(record_id)


def attach_package(record_id: str, package_dir: Path) -> dict[str, Any]:
    get(record_id)
    with connect() as conn:
        conn.execute(
            """
            UPDATE records
            SET status = 'exported', package_dir = ?, updated_at = ?
            WHERE id = ?
            """,
            (str(package_dir), utcnow(), record_id),
        )
    return get(record_id)
