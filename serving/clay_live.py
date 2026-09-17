"""Flag-gated live Clay routine adapter.

The CLI/API surface is alpha and changes; keep it outside the simulation
critical path. Callers must explicitly pass live=True and provide a
routine command. No emails are sent by this adapter.
"""
import json
import os
import subprocess


def run_routine(action, row, *, live=False, timeout=30):
    if not live:
        return {"mode": "simulated", "action": action, "row": row}
    if os.environ.get("CLAYFLY_LIVE") != "1":
        raise RuntimeError("live mode requires CLAYFLY_LIVE=1")
    # The command is configured by the user because Clay workflow/routine
    # names are workspace-specific. Never construct a shell command from
    # row data; argv avoids injection at the trust boundary.
    routine = os.environ.get("CLAYFLY_ROUTINE")
    if not routine:
        raise RuntimeError("set CLAYFLY_ROUTINE to a Clay routine name")
    proc = subprocess.run(
        ["clay", "routines", "run", routine, "--input", json.dumps(row)],
        capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-500:] or "Clay routine failed")
    return {"mode": "live", "action": action,
            "result": json.loads(proc.stdout or "null")}
