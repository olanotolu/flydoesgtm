"""Paired multi-seed eval of the trained arms on identical held-out worlds.

Every arm is scored on the same world seeds, so arm differences are
paired — feed the per-seed returns to report.py's bootstrap/Holm.
Checkpoints are the Modal B200 pulls; each brings its own trained
encoder gains, which are part of the policy, so they are applied.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from brain.loader import get_brain, make_shuffled_connectome
from brain.populations import build_channel_map, tracked_set
from learning.encoder import Encoder
from learning.policy import FlyPolicy, MLPPolicy
from learning.rollout import evaluate
from learning.trace import BatchTrace


def _load_policy(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if Path(path).name.startswith("mlp"):
        hidden = int(ck["state"]["net.0.weight"].shape[0])
        policy = MLPPolicy(16, hidden)
    else:
        policy = FlyPolicy(int(ck["n_feat"]),
                           use_raw_head=bool(ck.get("use_raw_head", True)))
    policy.load_state_dict(ck["state"], strict=True)
    policy.eval()
    return policy, ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[777, 779, 787, 797, 809, 811])
    p.add_argument("--n", type=int, default=160)
    p.add_argument("--days", type=int, default=45)
    p.add_argument("--out", default="results/eval_matrix.json")
    p.add_argument("--rent", type=float, default=0.0)
    p.add_argument("--kill-bonus", type=float, default=0.0)
    p.add_argument("--kill-penalty", type=float, default=0.0)
    p.add_argument("--expire-all", action="store_true")
    args = p.parse_args()

    device = os.environ.get("FLY_DEVICE", "cpu")
    world_kwargs = {"rent": args.rent, "kill_bonus": args.kill_bonus,
                    "kill_penalty": args.kill_penalty,
                    "expire_all": args.expire_all}
    shuffled_dir = make_shuffled_connectome(
        Path.home() / "fly-data-shuffled")
    arms = {
        "fly": ("results/fly_policy.pt", None),
        "fly_noraw": ("remote_results_noraw/fly_policy.pt", None),
        "fly_shuffled": ("remote_results_shuffled/fly_policy.pt",
                         str(shuffled_dir)),
        "mlp": ("remote_results_mlp/mlp_policy.pt", None),
    }
    brains = {}
    rows = []
    for arm, (ckpt, data) in arms.items():
        policy, ck = _load_policy(ckpt)
        key = data or "real"
        if key not in brains:
            brain = get_brain(batch=args.n, device=device, data=data)
            ch_map = build_channel_map(brain)
            tracked = tracked_set(brain, ch_map)
            brains[key] = (brain, ch_map,
                           BatchTrace(brain, tracked, tau=(0.05, 0.15, 0.5)))
        brain, ch_map, trace = brains[key]
        enc = Encoder(ch_map,
                      gains=np.asarray(ck["encoder_gains"], np.float32)
                      if ck.get("encoder_gains") is not None else None)
        for seed in args.seeds:
            val, stats = evaluate(brain, trace, enc, policy, device,
                                  seed=seed, n=args.n, days=args.days,
                                  world_kwargs=world_kwargs)
            rows.append({"arm": arm, "seed": seed, "return": float(val),
                         **{k: (float(v) if isinstance(v, np.floating)
                                else int(v) if isinstance(v, np.integer)
                                else v) for k, v in stats.items()}})
            print(f"{arm} s{seed}: {val:9.1f}  "
                  f"meet {stats.get('meetings')}  closed {stats.get('closed')}  "
                  f"spam {stats.get('spam')}", flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    summarize(rows, Path(args.out).with_name("eval_summary.json"))


def summarize(rows, out):
    from experiments.report import holm_bonferroni, paired_bootstrap_delta
    arms = sorted({r["arm"] for r in rows})
    summary = {}
    for arm in arms:
        vals = np.array([r["return"] for r in rows if r["arm"] == arm])
        summary[arm] = {"mean": float(vals.mean()), "std": float(vals.std()),
                        "min": float(vals.min()), "max": float(vals.max()),
                        "n": int(vals.size)}
    # Paired deltas vs the real fly on identical world seeds.
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["arm"]] = r["return"]
    seeds = sorted(by_seed)
    comparisons = [a for a in arms if a != "fly"]
    deltas, pvals = {}, []
    for arm in comparisons:
        a = np.array([by_seed[s][arm] for s in seeds])
        b = np.array([by_seed[s]["fly"] for s in seeds])
        mean_delta, (lo, hi) = paired_bootstrap_delta(a, b)
        rng = np.random.default_rng(0)
        d = a - b
        boot = d[rng.integers(0, len(d), size=(10_000, len(d)))].mean(1)
        p = 2 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
        deltas[arm] = {"mean_delta": mean_delta, "ci95": [lo, hi],
                       "p_boot": min(1.0, p)}
        pvals.append(min(1.0, p))
    for arm, adj in zip(comparisons, holm_bonferroni(pvals)):
        deltas[arm]["p_holm"] = adj
    payload = {"arms": summary, "paired_vs_fly": deltas,
               "seeds": seeds, "baseline": "fly"}
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    import sys
    if "--summarize-only" in sys.argv:
        rows = json.loads(Path("results/eval_matrix.json").read_text())
        summarize(rows, Path("results/eval_summary.json"))
    else:
        main()
