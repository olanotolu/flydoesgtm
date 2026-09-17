"""Common-seed reward-profile calibration for the deterministic baselines."""
import json

from experiments.benchmark import run_policy
from environment.clay_world import REWARD_PROFILES


def sweep(n=20, days=15, seeds=range(3)):
    rows = []
    for profile in REWARD_PROFILES:
        for seed in seeds:
            row = run_policy("rules", n, days, 700 + seed, profile)
            row["profile"] = profile
            row["seed"] = seed
            rows.append(row)
    return rows


if __name__ == "__main__":
    print(json.dumps(sweep(), indent=2))
