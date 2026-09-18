"""THE falsification gate: does the connectome's specific wiring matter?

The architecture rests on a claim — that the curated readout population
(descending + ascending + Kenyon cells + dopaminergic + motor, 4,004
neurons) and the real topology are doing work that a generic reservoir
would not. This measures that claim instead of asserting it.

Two arms, identical in everything but topology:

  REAL      the MaleCNS v1.0 connectome.
  SHUFFLED  `make_shuffled_connectome` — the same cells, the same channel
            map, the same readout populations, and every neuron keeps its
            out-degree and its outgoing weight multiset. Only the
            postsynaptic targets are globally re-permuted. So it is the
            correct null: it isolates topology and nothing else.

Within each arm, five readouts are probed on the same frozen features,
same world, same seed, same split procedure:

  raw_obs     16 channels straight from the world. Reference point: how
              much brain does this task need at all?
  injected    the 1,024 neurons the encoder writes into (16 x 64). The
              input port. `populations.py` excludes these from the
              tracked set "so the readout cannot win by copying its raw
              input back to itself" — this tests whether that is
              load-bearing or decorative.
  curated     `tracked_set`, the architecture's chosen population.
  random4k    an arbitrary 4,004-neuron sample — same size as curated.
  random20k   an arbitrary 20,000-neuron sample.

Decision rule, stated before the numbers are read:

  If REAL and SHUFFLED score the same, the wiring is not doing the work.
  If random samples match or beat the curated set, the curation is not
  doing the work either. Either result invalidates further training spend
  on the "use the connectome's structure" thesis.

    python -m experiments.wiring_gate
    python -m experiments.wiring_gate --quick
"""
import argparse

import numpy as np

from brain.loader import get_brain, make_shuffled_connectome, weights_checksum
from brain.populations import build_channel_map, tracked_set
from environment.clay_world import World
from environment.teacher import teacher_actions
from learning.encoder import Encoder
from learning.rollout import SIM_STEPS
from learning.trace import BatchTrace

SHUFFLED_CACHE = "/tmp/fly-data-shuffled"


def _cols(pos, ids, stride):
    """Feature columns for `ids`, for every trace time constant."""
    p = np.array([pos[int(i)] for i in ids if int(i) in pos])
    return np.concatenate([p + s * stride for s in range(3)])


def _probe(Ztr, ytr, Zte, yte, k, ridge=1e-2):
    """Ridge solved in the dual — d >> n, so the n x n solve is instant."""
    Y = np.zeros((len(ytr), k))
    Y[np.arange(len(ytr)), ytr] = 1.0
    K = Ztr @ Ztr.T + ridge * np.eye(len(Ztr))
    A = np.linalg.solve(K, Y)
    return float((((Zte @ Ztr.T) @ A).argmax(1) == yte).mean())


def _run(Z, y, k, splits=20):
    """Mean +- std over random splits; one split quantises too coarsely."""
    acc = []
    for s in range(splits):
        r = np.random.default_rng(s)
        idx = r.permutation(len(Z))
        cut = int(0.6 * len(Z))
        tr, te = idx[:cut], idx[cut:]
        acc.append(_probe(Z[tr], y[tr], Z[te], y[te], k))
    return np.mean(acc) * 100, np.std(acc) * 100


def measure(label, data_dir, n, days, seed, splits):
    brain = get_brain(batch=n, device="cpu", data=data_dir)
    chmap = build_channel_map(brain)
    injected = np.unique(np.concatenate(
        [np.asarray(v, np.int64) for v in chmap.values()]))
    curated = tracked_set(brain, chmap)
    rng = np.random.default_rng(5)
    rand4k = rng.choice(np.arange(brain.n), len(curated), replace=False)
    rand20k = rng.choice(np.arange(brain.n), 20000, replace=False)

    readouts = [("curated", curated), ("injected", injected),
                ("random4k", rand4k), ("random20k", rand20k)]
    union = np.unique(np.concatenate([ids for _, ids in readouts]))
    pos = {int(x): i for i, x in enumerate(union)}
    stride = len(union)

    trace = BatchTrace(brain, union, tau=(0.05, 0.15, 0.5))
    enc = Encoder(chmap)
    brain.reset()
    trace.reset()

    blocks, obs_rows, A, Q = [], [], [], []
    w = World(n_accounts=n, days=days, seed=seed - 1)
    for day in range(w.days):
        w.start_day(day)
        rows, idxs = w.observe(day)
        if len(idxs) == 0:
            continue
        inj = enc.inject(rows, idxs, brain.batch)
        for _ in range(SIM_STEPS):
            brain.step(inject=inj)
            trace.observe(brain)
        blocks.append(np.asarray(trace.features(idxs), dtype=np.float32))
        obs_rows.append(np.asarray(rows, dtype=np.float32))
        A.append(teacher_actions(rows))
        Q.append(np.digitize(w.intent[np.asarray(idxs)], [0.25, 0.5, 0.75]))
        w.apply(day, idxs, A[-1], slot_of=lambda k: k)
    w.end_episode()

    FULL = np.concatenate(blocks)
    OBS = np.concatenate(obs_rows)
    A = np.concatenate(A)
    Q = np.concatenate(Q)

    print(f"\n{label}:  sha {weights_checksum(brain=brain)[:16]}...  "
          f"union {len(union):,}  block {FULL.shape}  "
          f"decisions {len(A):,}", flush=True)

    out = {}
    for nm, ids in [("raw_obs", None)] + readouts:
        Z = OBS if ids is None else FULL[:, _cols(pos, ids, stride)]
        a, asd = _run(Z, A, 7, splits)
        q, qsd = _run(Z, Q, 4, splits)
        out[nm] = (a, asd, q, qsd, Z.shape[1])
        print(f"   {nm:<10} {Z.shape[1]:>6}d   ACTIONS {a:6.2f}% +-{asd:4.1f}"
              f"   QUALITY {q:6.2f}% +-{qsd:4.1f}", flush=True)
    return out, (np.bincount(A).argmax(), np.bincount(A).max() / len(A) * 100,
                 np.bincount(Q).argmax(), np.bincount(Q).max() / len(Q) * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accounts", type=int, default=100)
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--shuffled-dir", default=SHUFFLED_CACHE)
    ap.add_argument("--quick", action="store_true",
                    help="small world; sanity-check the harness only")
    args = ap.parse_args()

    n, days = (60, 6) if args.quick else (args.accounts, args.days)
    shuffled = make_shuffled_connectome(args.shuffled_dir)

    real, base = measure("REAL connectome", None, n, days, args.seed,
                         args.splits)
    shuf, _ = measure("SHUFFLED wiring", str(shuffled), n, days, args.seed,
                      args.splits)

    print(f"\nmajority baseline:  ACTIONS {base[1]:.2f}%   "
          f"QUALITY {base[3]:.2f}%")
    print("\n" + "=" * 68)
    print(f"{'readout':<11}{'dims':>7}  {'target':<9}"
          f"{'REAL':>11}{'SHUFFLED':>11}{'delta':>9}")
    print("-" * 68)
    for nm in ("raw_obs", "injected", "curated", "random4k", "random20k"):
        for ti, tn in ((0, "ACTIONS"), (2, "QUALITY")):
            r, s = real[nm][ti], shuf[nm][ti]
            print(f"{nm:<11}{real[nm][4]:>7}  {tn:<9}"
                  f"{r:>10.2f}%{s:>10.2f}%{r - s:>+9.2f}")
    print("=" * 68)
    print("delta ~ 0  -> the wiring is not doing the work.")
    print("random >= curated -> the curation is not doing the work.")
    print("Either one invalidates further spend on the structure thesis.")


if __name__ == "__main__":
    main()
