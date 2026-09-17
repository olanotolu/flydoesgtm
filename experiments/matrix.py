"""Immutable B200 experiment matrix and run-manifest utilities.

This module schedules evidence; it does not claim that a run has completed.
Each run receives a unique directory and records the paired world seeds and
the exact arm definition before training starts.
"""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

ARMS = (
    "fly_hybrid", "connectome_only", "matched_mlp", "matched_gru",
    "random_reservoir", "rewired_connectome", "random_sensory_map",
    "zero_propagation", "teacher_only", "ppo_only", "bc_plus_ppo",
)


def create_manifests(root, seeds, *, worlds=("train", "validation", "test")):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    created = []
    for arm in ARMS:
        for seed in seeds:
            run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{arm}-s{seed}-{uuid.uuid4().hex[:8]}"
            path = root / run_id
            path.mkdir()
            manifest = {
                "run_id": run_id, "arm": arm, "seed": int(seed),
                "worlds": list(worlds), "paired_world_seed": int(seed) * 1009,
                "status": "planned", "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            created.append(manifest)
    return created


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments/runs")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    args = parser.parse_args()
    print(json.dumps(create_manifests(args.root, args.seeds), indent=2))


if __name__ == "__main__":
    main()
