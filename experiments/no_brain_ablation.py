"""Decisive control: does the connectome pathway contribute at all?

Three arms on identical held-out worlds per checkpoint:
  full           — the checkpoint as-is
  no_brain       — brain trace features zeroed (feat * 0). With
                   use_raw_head=True the 16-dim raw obs head stays live,
                   so this isolates the connectome's contribution; on a
                   use_raw_head=False checkpoint feat=0 means NO input
                   and is the sanity floor.
  permuted_brain — features randomly permuted (fixed rng): same feature
                   magnitudes, destroyed connectome->readout alignment.

Eval protocol matches modal_eval.py / eval_remote.json: n=160, 45 days,
triage economy (rent + kill_bonus + expire_all), greedy policy.

  python experiments/no_brain_ablation.py \
      --checkpoint remote_results_v2/fly_policy.pt --tag v2
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from brain.loader import get_brain
from brain.populations import build_channel_map, tracked_set
from learning.encoder import Encoder
from learning.policy import FlyPolicy
from learning.rollout import evaluate
from learning.trace import BatchTrace
from experiments.fly_ablation import FeatureAblation

V2 = {"rent": 0.05, "kill_bonus": 3.0, "expire_all": True}


def run(checkpoint, tag, n=160, days=45, seed=619, out=None):
    device = "cpu"
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base = FlyPolicy(int(ck["n_feat"]),
                     use_raw_head=bool(ck.get("use_raw_head", True)))
    base.load_state_dict(ck["state"], strict=True)
    base.eval()
    f = base.readout.in_features
    rng = np.random.default_rng(19)
    arms = {
        "full": base,
        "no_brain": FeatureAblation(base, mask=torch.zeros(f)),
        "permuted_brain": FeatureAblation(
            base, permutation=torch.tensor(rng.permutation(f),
                                           dtype=torch.long)),
    }
    brain = get_brain(batch=n, device=device)
    channels = build_channel_map(brain)
    tracked = tracked_set(brain, channels)
    trace = BatchTrace(brain, tracked, tau=(0.05, 0.15, 0.5))
    enc = Encoder(channels,
                  gains=np.asarray(ck["encoder_gains"], np.float32)
                  if ck.get("encoder_gains") is not None else None)
    rows = []
    for arm, policy in arms.items():
        t0 = time.time()
        val, stats = evaluate(brain, trace, enc, policy, device,
                              seed=seed, n=n, days=days,
                              world_kwargs=V2)
        rows.append({"checkpoint": tag, "arm": arm, "seed": seed,
                     "n": n, "days": days, "econ": "v2",
                     "return": float(val),
                     "wall_s": round(time.time() - t0, 1),
                     **{k: (float(v) if isinstance(v, np.floating)
                            else int(v) if isinstance(v, np.integer) else v)
                        for k, v in stats.items()}})
        print(f"{tag}/{arm} s{seed}: {val:9.1f}  "
              f"({rows[-1]['wall_s']}s) {stats}", flush=True)
    if out:
        Path(out).write_text(json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--n", type=int, default=160)
    p.add_argument("--days", type=int, default=45)
    p.add_argument("--seed", type=int, default=619)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    tag = args.tag or Path(args.checkpoint).parent.name
    out = args.out or f"results/no_brain_ablation_{tag}.json"
    print(json.dumps(run(args.checkpoint, tag, n=args.n, days=args.days,
                         seed=args.seed, out=out), indent=2))
