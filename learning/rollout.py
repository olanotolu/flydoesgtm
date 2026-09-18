"""Episode rollout: world <-> brain <-> policy, with cached features.

The frozen brain turns each observation into a trace feature vector
once; PPO then reuses those cached features across epochs — the
expensive rollout happens once per episode, the update is cheap.

Batch lane discipline: fly i lives in lane i and only ever sees
account i. Persistent voltages + persistent trace = per-account memory.
"""
import numpy as np
import torch
from torch.distributions import Categorical

from environment.clay_world import World, ACTIONS
from environment.teacher import teacher_actions
from learning.ppo import generalized_advantage


def as_torch(value, device):
    """Move NumPy/CuPy trace features without forcing CUDA through CPU."""
    if hasattr(value, "__dlpack__"):
        return torch.utils.dlpack.from_dlpack(value).to(device)
    return torch.as_tensor(value, device=device)

SIM_STEPS = int(__import__("os").environ.get("FLY_SIM_STEPS", "12"))
# 12 steps (12 x 20ms = 240ms) is the shortest horizon at which injected
# current reliably reaches the descending/ascending/Kenyon populations the
# readout watches. At 4 steps only ~3.4% of trace features ever fire and a
# linear probe cannot separate account qualities (52.8%, chance 25%); at 12
# steps 76% of features fire and the probe reaches 85.7%. Train and serve
# at the same value — the server reads this horizon back out of the
# checkpoint, so changing it here alone will not silently desync the demo.
N_ACTIONS = 7


def run_episode(brain, trace, enc, policy, world, allowed, device,
              train=True, sim_steps=SIM_STEPS, greedy=False):
    """One episode. Returns a trajectory dict with CACHED features —
    the policy's forward operates on (feat, obs), never re-runs the
    brain during the update."""
    allowed_logits = torch.tensor(
        [j in allowed for j in range(N_ACTIONS)], device=device)

    feats, obss, action_idx, logprobs, values = [], [], [], [], []
    positions = {}
    row_slots = []
    n_slots = 0

    brain.reset()
    trace.reset()

    for day in range(world.days):
        world.start_day(day)
        obs, idxs = world.observe(day)
        if len(idxs) == 0:
            continue

        inject = enc.inject(obs, idxs, brain.batch)
        for _ in range(sim_steps):
            brain.step(inject=inject)
            trace.observe(brain)
        feat = as_torch(trace.features(idxs), device)
        x = torch.as_tensor(obs, device=device)

        with torch.set_grad_enabled(train):
            logits, value = policy(feat, x)
        logits = logits.masked_fill(~allowed_logits, -1e9)
        dist = Categorical(logits=logits)
        requested = logits.argmax(1) if greedy else dist.sample()

        base = n_slots
        slots = list(range(base, base + len(idxs)))
        n_slots += len(idxs)
        executed = world.apply(
            day, idxs, requested.cpu().numpy().astype(np.int64),
            slot_of=lambda k: slots[k],
        )
        a_world = torch.as_tensor(executed, device=device,
                                  dtype=torch.long)

        row_start = sum(x.shape[0] for x in feats)
        for row, account in enumerate(idxs):
            positions.setdefault(int(account), []).append(row_start + row)
            row_slots.append(slots[row])
        feats.append(feat.detach())
        obss.append(x)
        action_idx.append(a_world)
        logprobs.append(dist.log_prob(a_world).detach())
        values.append(value.detach())

    world.end_episode()
    feat_all = torch.cat(feats)
    obs_all = torch.cat(obss)
    val_all = torch.cat(values)
    rewards = torch.zeros(len(feat_all), device=device)
    dones = torch.zeros(len(feat_all), dtype=torch.bool, device=device)
    # Delayed account economics lands on the decision slot that caused it.
    # Each account's rows are gathered before GAE, so day-major batching
    # cannot leak one account's rewards into another account's history.
    for account_positions in positions.values():
        account_rows = torch.as_tensor(account_positions, device=device)
        for row, slot in zip(account_positions,
                             [row_slots[i] for i in account_positions]):
            rewards[row] = float(world.credit.get(slot, 0.0))
        dones[account_rows[-1]] = True

    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    for account_positions in positions.values():
        account_rows = torch.as_tensor(account_positions, device=device)
        account_adv, account_ret = generalized_advantage(
            rewards[account_rows], val_all[account_rows],
            dones[account_rows])
        advantages[account_rows] = account_adv
        returns[account_rows] = account_ret

    return {
        "feat": feat_all, "obs": obs_all,
        "act": torch.cat(action_idx), "logp": torch.cat(logprobs),
        "val": val_all, "ret": returns.detach(),
        "adv": advantages.detach(), "reward": rewards, "done": dones,
        "action_mask": allowed_logits.expand(len(feat_all), -1).clone(),
        "teacher": torch.as_tensor(teacher_actions(obs_all.detach().cpu().numpy()),
                                   device=device),
    }


def evaluate(brain, trace, enc, policy, device, seed=777, n=500,
             days=60, world_kwargs=None):
    """Greedy episode on a held-out world scored on the pure-economics
    ledger — checkpoint selection, same discipline as the worm."""
    world = World(n_accounts=n, days=days, daily_budget=100.0, seed=seed,
                  **(world_kwargs or {}))
    run_episode(brain, trace, enc, policy, world, set(range(7)),
                device, train=False, greedy=True)
    return sum(world.econ.values()), dict(world.stats)
