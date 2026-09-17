"""Small, reproducible benchmark harness.

This deliberately reports the fly alongside simple policies; it does not
hide a weak result behind a screenshot. Run:
  python -m experiments.benchmark --n 20 --days 15
"""
import argparse
import json
import numpy as np

from environment.clay_world import (World, WAIT, OBSERVE, RESEARCH, ENRICH,
                                     EMAIL, ESCALATE, IGNORE)


def run_policy(kind, n, days, seed, reward_profile="balanced"):
    w = World(n_accounts=n, days=days, seed=seed,
              reward_profile=reward_profile)
    slot = [0]
    for day in range(days):
        w.start_day(day)
        obs, idxs = w.observe(day)
        actions = []
        for row in obs:
            funding, hiring, intent, job_change, negative, trigger = row[:6]
            if kind == "random":
                a = int(w.rng.integers(7))
            elif kind == "wait":
                a = WAIT
            elif kind == "rules":
                # Buy information only for ambiguous accounts; qualify bad
                # rows; contact hot rows after one research pass.
                hot = intent > 0.68 and hiring > 0.55
                cold = intent < 0.35 or negative > 0.75
                a = IGNORE if cold else (EMAIL if hot else RESEARCH)
            else:
                raise ValueError(kind)
            actions.append(a)
        w.apply(day, idxs, np.asarray(actions),
                slot_of=lambda k: slot.__setitem__(0, slot[0] + 1)
                or slot[0] - 1)
    w.end_episode()
    return {"policy": kind, "return": float(sum(w.econ.values())),
            **{k: (float(v) if isinstance(v, float) else int(v))
               for k, v in w.stats.items()}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--days", type=int, default=15)
    p.add_argument("--reward-profile", default="balanced")
    args = p.parse_args()
    rows = [run_policy(k, args.n, args.days, 700 + i,
                       args.reward_profile)
            for i, k in enumerate(("wait", "rules", "random"))]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
