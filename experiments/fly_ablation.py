"""Small real-fly readout ablations on identical held-out worlds.

These ablate the engineered readout features, not the biology. A true
connectome rewiring arm stays separate because it changes the scientific
question.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from brain.loader import get_brain
from brain.populations import build_channel_map, tracked_set
from environment.clay_world import World
from learning.encoder import Encoder
from learning.policy import FlyPolicy
from learning.rollout import evaluate
from learning.trace import BatchTrace


class FeatureAblation(torch.nn.Module):
    def __init__(self, base, mask=None, permutation=None):
        super().__init__()
        self.base = base
        self.register_buffer("mask", mask if mask is not None
                             else torch.ones(base.readout.in_features))
        self.register_buffer("permutation", permutation if permutation is not None
                             else torch.arange(base.readout.in_features))

    def forward(self, feat, obs):
        return self.base(feat[:, self.permutation] * self.mask, obs)


def run(checkpoint="results/fly_policy.pt", n=20, days=15):
    device = "cpu"
    brain = get_brain(batch=n, device=device)
    channels = build_channel_map(brain)
    tracked = tracked_set(brain, channels)
    trace = BatchTrace(brain, tracked, tau=(0.05, 0.15, 0.5))
    enc = Encoder(channels)
    state = torch.load(checkpoint, map_location=device)
    base = FlyPolicy(int(state["n_feat"]))
    base.load_state_dict(state["state"], strict=False)
    base.eval()
    f = base.readout.in_features
    rng = np.random.default_rng(19)
    masks = {
        "fly": torch.ones(f),
        "half_features_lesioned": torch.tensor(
            rng.random(f) > 0.5, dtype=torch.float32),
    }
    perm = torch.tensor(rng.permutation(f), dtype=torch.long)
    policies = {name: FeatureAblation(base, mask=mask)
                for name, mask in masks.items()}
    policies["shuffled_features"] = FeatureAblation(
        base, permutation=perm)
    rows = []
    for name, policy in policies.items():
        world = World(n_accounts=n, days=days, seed=777)
        value, stats = evaluate(brain, trace, enc, policy, device,
                                seed=777, n=n, days=days)
        rows.append({"arm": name, "return": float(value),
                     **{k: (float(v) if isinstance(v, np.floating)
                             else int(v) if isinstance(v, np.integer) else v)
                        for k, v in stats.items()}})
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="results/fly_policy.pt")
    args = p.parse_args()
    print(json.dumps(run(args.checkpoint), indent=2))
