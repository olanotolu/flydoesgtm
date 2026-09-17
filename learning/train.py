"""PPO curriculum over the frozen fly brain.

    python -m learning.train            # local smoke (CPU, small worlds)
    python -m learning.train --full     # full curriculum (GPU/Modal)
    python -m learning.train --mlp      # parameter-matched dense baseline

The brain batch dimension == number of accounts: fly i only ever sees
account i, so the whole day is stepped in lockstep.
"""
import os
import sys
import time

import numpy as np
import torch

from brain.loader import get_brain, weights_checksum
from brain.populations import build_channel_map, tracked_set
from environment.clay_world import World
from learning.encoder import Encoder
from learning.policy import FlyPolicy, MLPPolicy
from learning.ppo import ppo_update
from learning.pretrain import warm_start
from learning.rollout import run_episode, evaluate
from learning.trace import BatchTrace

FULL = [
    # Qualification is available from the first stage; otherwise a bad
    # account remains active forever and the policy is punished for not
    # repeatedly touching it. Smaller lanes keep this reproducible on one
    # B200; each lane is still a real full-connectome fly.
    dict(name="stage1 100acc/30d/5act", n=100, days=30,
         allowed={0, 1, 2, 4, 6}, lam=0.25, episodes=20),
    dict(name="stage2 128acc/35d/7act", n=128, days=35,
         allowed=set(range(7)), lam=0.25, episodes=20),
    dict(name="stage3 160acc/45d/7act", n=160, days=45,
         allowed=set(range(7)), lam=0.25, episodes=20),
]
SMOKE = [
    dict(name="smoke 24acc/15d/4act", n=24, days=15,
         allowed={0, 1, 2, 4}, lam=0.25, episodes=6),
    dict(name="smoke 48acc/20d/7act", n=48, days=20,
         allowed=set(range(7)), lam=0.25, episodes=4),
]
CONSERVE = 0.03


def run(curriculum, brain_device="cpu", torch_device=None,
        out_dir="results", train_mlp=False, seed0=10_000,
        val_every=5, verbose=True):
    """Shared trainer for local runs and Modal remote calls.

    Returns (best_val, save_path). Writes checkpoints + metrics.json
    into out_dir."""
    import json
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch_device or (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"torch device: {device}  brain device: {brain_device}",
          flush=True)

    w_sha = weights_checksum()
    print(f"connectome sha256: {w_sha[:16]}...  (frozen)", flush=True)

    ep = 0
    best_val, best_state = -float("inf"), None
    history = []
    t0 = time.time()
    policy = None
    for si, stage in enumerate(curriculum):
        print(f"\n== {stage['name']} ==", flush=True)
        brain = get_brain(batch=stage["n"], device=brain_device)
        ch_map = build_channel_map(brain)
        tracked = tracked_set(brain, ch_map)
        enc = Encoder(ch_map)
        trace = BatchTrace(brain, tracked, tau=(0.05, 0.15, 0.5))
        if ep == 0:
            print(f"  tracked features: {trace.F} neurons, "
                  f"{len(ch_map)} channels", flush=True)
        if train_mlp:
            policy = MLPPolicy(16, 128).to(device)
            save_to = out / "mlp_policy.pt"
        else:
            policy = FlyPolicy(trace.F * trace.n_scales).to(device)
            if best_state is not None:          # carry weights between stages
                policy.load_state_dict(best_state)
            save_to = out / "fly_policy.pt"
        opt = torch.optim.Adam(policy.parameters(), lr=0.01)
        if ep == 0 and not train_mlp \
                and os.environ.get("CLAYFLY_WARM_START", "1") == "1":
            teacher_world = World(n_accounts=stage["n"], days=min(10, stage["days"]),
                                  seed=seed0 - 1)
            teacher_acc = warm_start(policy, brain, trace, enc,
                                     teacher_world, device)
            print(f"  teacher warm-start accuracy: {teacher_acc:.3f}",
                  flush=True)
        print(f"  trainable params: "
              f"{sum(p.numel() for p in policy.parameters())}",
              flush=True)

        stage_val, stage_state = -float("inf"), None
        for e in range(stage["episodes"]):
            world = World(n_accounts=stage["n"], days=stage["days"],
                          seed=seed0 + ep, lambda_cost=stage["lam"],
                          conserve_coef=CONSERVE, vary_costs=True)
            traj = run_episode(brain, trace, enc, policy, world,
                               stage["allowed"], device)
            ppo_update(policy, opt, traj)
            enc.adapt(traj["obs"].detach().cpu().numpy(),
                      traj["adv"].detach().cpu().numpy())
            ep += 1
            if ep % val_every == 0 or e == stage["episodes"] - 1:
                val, st = evaluate(brain, trace, enc, policy, device,
                                   n=min(stage["n"], 200),
                                   days=min(stage["days"], 45))
                flag = ""
                if val > stage_val:
                    stage_val = val
                    stage_state = {k: v.detach().clone() for k, v in
                                   policy.state_dict().items()}
                if val > best_val:
                    best_val, best_state = val, stage_state
                    flag = "  <- best"
                rec = {"ep": ep, "stage": si,
                       "reward": float(traj["reward"].sum()),
                       "spent": float(world.stats["spent"]), "val": float(val),
                       **{k: world.stats[k] for k in
                          ("meetings", "closed", "spam", "replies")}}
                history.append(rec)
                if verbose:
                    print(f"  ep {ep:>4}  reward {rec['reward']:>9.1f}  "
                          f"spent ${rec['spent']:>7.2f}  "
                          f"meet {rec['meetings']}  "
                          f"closed {rec['closed']}  "
                          f"spam {rec['spam']}  "
                          f"val {val:>9.1f}  "
                          f"[{time.time()-t0:6.0f}s]{flag}", flush=True)

    if best_state is not None:
        policy.load_state_dict(best_state)
    torch.save({"state": policy.state_dict(),
                "n_feat": trace.F * trace.n_scales,
                "w_sha256": w_sha}, save_to)
    (out / "metrics.json").write_text(
        json.dumps(history, indent=1, default=lambda x: x.item()
                   if hasattr(x, "item") else float(x)))
    print(f"\nsaved -> {save_to}  (best val {best_val:.1f})", flush=True)
    assert weights_checksum() == w_sha, "brain weights changed!"
    return best_val, str(save_to)


def main():
    curriculum = FULL if "--full" in sys.argv else SMOKE
    run(curriculum, train_mlp="--mlp" in sys.argv)


if __name__ == "__main__":
    main()
