"""Is the frozen fly actually sensitive to the GTM channels?

A readout can only be as good as the features it reads. This measures,
without training anything, whether a linear probe on the fly's spike
traces can tell account quality apart — and sweeps the two knobs that
control it: encoder injection gain and sim steps.

If probe accuracy sits at chance, the readout cannot possibly
discriminate and any confident-looking policy output is a constant. Run
this before trusting a checkpoint, and before a demo run.

    python -m experiments.sensitivity
    python -m experiments.sensitivity --quick
"""
import argparse
import json
from pathlib import Path

import numpy as np

from brain.loader import get_brain
from brain.populations import build_channel_map, tracked_set
from environment.clay_world import ACTION_COST
from learning.encoder import Encoder
from learning.trace import BatchTrace

SIGNAL_KEYS = ("funding", "hiring", "intent", "job_change", "negative",
               "trigger")
# Four accounts no human would confuse: dead, weak, strong, maxed.
DEFAULT_LEVELS = (0.05, 0.35, 0.65, 0.95)
BASE_COSTS = np.array([2.0, 1.0, 5.0, 15.0], np.float32)


def build_obs(level, rng, sigma):
    """One account at `level` on all six channels, plus observation noise."""
    obs = np.zeros((1, 16), np.float32)
    obs[0, :6] = np.clip(np.full(6, level) + rng.normal(0, sigma, 6), 0, 1)
    obs[0, 6] = 1.0                      # budget_frac at day start
    obs[0, 7] = 0.5                      # day_frac
    obs[0, 10:14] = ACTION_COST[[2, 3, 4, 5]] / BASE_COSTS
    obs[0, 14] = 1.0
    return obs


def collect(brain, trace, channels, steps, gain, levels, n_per_level, sigma,
            seed):
    """Run the frozen brain once per sample; return (features, labels)."""
    enc = Encoder(channels, gains=np.full(16, gain, np.float32))
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    for label, level in enumerate(levels):
        for _ in range(n_per_level):
            obs = build_obs(level, rng, sigma)
            brain.reset()
            trace.reset()
            for _ in range(steps):
                brain.step(inject=enc.inject(obs, np.array([0]), 1))
                trace.observe(brain)
            row = np.asarray(trace.features([0])).reshape(-1)
            feats.append(row.astype(np.float64))
            labels.append(label)
    return np.stack(feats), np.array(labels)


def probe_accuracy(feats, labels, n_classes, seed=0, split=0.6, ridge=1e-2):
    """Ridge linear readout on one-hot targets, trained on a held-out split.

    This is the same function class the real policy uses (a linear map from
    trace features), so its accuracy is an upper bound on what the shipped
    readout could express from these features.

    Feature dimension (thousands) dwarfs sample count (tens), so the ridge
    problem is solved in the dual — an n x n system instead of d x d. Same
    objective, and it makes averaging over many splits cheap.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    cut = int(len(labels) * split)
    train, test = order[:cut], order[cut:]
    mu, sd = feats[train].mean(0), feats[train].std(0) + 1e-8
    xtr, xte = (feats[train] - mu) / sd, (feats[test] - mu) / sd
    targets = np.eye(n_classes)[labels[train]]
    kernel = xtr @ xtr.T + ridge * np.eye(len(train))
    alpha = np.linalg.solve(kernel, targets)
    predicted = ((xte @ xtr.T) @ alpha).argmax(1)
    return float((predicted == labels[test]).mean())


def mean_probe_accuracy(feats, labels, n_classes, n_splits=100):
    """Average over random splits. One split has too few test points to
    trust alone — a 13-point test set moves in 7.7% steps."""
    scores = [probe_accuracy(feats, labels, n_classes, seed=s)
              for s in range(n_splits)]
    return float(np.mean(scores)), float(np.std(scores))


def occupancy(feats):
    """Fraction of feature columns that ever fire — a dead trace reads nothing."""
    return float((feats != 0).any(axis=0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[4, 8, 12, 25])
    ap.add_argument("--gains", type=float, nargs="+", default=[0.6, 1.0, 1.5])
    ap.add_argument("--n-per-level", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--splits", type=int, default=100,
                    help="random train/test splits averaged per config")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--min-probe", type=float, default=0.60,
                    help="below this, the readout cannot discriminate")
    ap.add_argument("--quick", action="store_true",
                    help="single configuration, for CI")
    ap.add_argument("--out", default="results/sensitivity.json")
    args = ap.parse_args()

    if args.quick:
        args.steps, args.gains = [4], [0.6]
        args.n_per_level, args.splits = 4, 8

    levels = DEFAULT_LEVELS
    brain = get_brain(batch=1, device="cpu")
    channels = build_channel_map(brain)
    tracked = tracked_set(brain, channels)
    trace = BatchTrace(brain, tracked, tau=(0.05, 0.15, 0.5))
    chance = 1.0 / len(levels)

    print(f"dt={float(brain.dt) * 1000:.0f}ms  tracked={len(tracked)}  "
          f"levels={levels}  chance={chance:.0%}  "
          f"samples/config={len(levels) * args.n_per_level}")
    print(f"{'gain':>5} {'steps':>6} {'ms':>6} {'occupancy':>10} "
          f"{'probe acc':>17}")
    rows, flagged = [], []
    for gain in args.gains:
        for steps in args.steps:
            feats, labels = collect(brain, trace, channels, steps, gain,
                                    levels, args.n_per_level, args.sigma,
                                    args.seed)
            acc, spread = mean_probe_accuracy(feats, labels, len(levels),
                                              args.splits)
            occ = occupancy(feats)
            ms = steps * float(brain.dt) * 1000
            rows.append({"gain": gain, "steps": steps, "ms": ms,
                         "probe_accuracy": acc, "probe_std": spread,
                         "occupancy": occ})
            mark = ""
            if acc < args.min_probe:
                mark = "  <- below threshold"
                flagged.append((gain, steps, acc))
            print(f"{gain:>5.2f} {steps:>6d} {ms:>5.0f}ms "
                  f"{occ:>9.1%} {acc:>9.1%} +/- {spread:>4.1%}{mark}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "chance": chance, "min_probe": args.min_probe,
        "n_per_level": args.n_per_level, "sigma": args.sigma,
        "splits": args.splits, "seed": args.seed, "results": rows,
    }, indent=1))
    print(f"\nwrote {out}")

    if flagged:
        worst = min(flagged, key=lambda r: r[2])
        print(f"\n{len(flagged)} configuration(s) below "
              f"{args.min_probe:.0%} probe accuracy. Worst: gain="
              f"{worst[0]} steps={worst[1]} at {worst[2]:.1%}. The frozen "
              f"features do not carry account quality there, so the readout "
              f"cannot discriminate and any confident output is a constant.")
        return 1
    print(f"\nall configurations reach {args.min_probe:.0%} probe accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
