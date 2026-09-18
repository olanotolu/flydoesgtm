"""Short behavior-cloning warm start for the artificial readout.

The teacher is transparent rules, not an oracle hidden in the fly. It
only initializes the engineered readout; PPO remains responsible for
learning economics and delayed outcomes afterward.
"""
import torch
import torch.nn.functional as F

from environment.teacher import teacher_actions
from learning.rollout import SIM_STEPS, as_torch


def warm_start(policy, brain, trace, enc, world, device, epochs=20):
    brain.reset(); trace.reset()
    feats, obs, labels = [], [], []
    for day in range(world.days):
        world.start_day(day)
        rows, idxs = world.observe(day)
        if len(idxs) == 0:
            continue
        inject = enc.inject(rows, idxs, brain.batch)
        for _ in range(SIM_STEPS):
            brain.step(inject=inject)
            trace.observe(brain)
        feats.append(as_torch(trace.features(idxs), device))
        obs.append(torch.as_tensor(rows, device=device))
        labels.append(torch.as_tensor(teacher_actions(rows), device=device))
        actions = labels[-1].detach().cpu().numpy()
        world.apply(day, idxs, actions, slot_of=lambda k: k)
    world.end_episode()
    if not feats:
        return 0.0
    x = torch.cat(feats); raw = torch.cat(obs); y = torch.cat(labels)
    # `temperature` is deliberately left out: it is the knob that sets the
    # logit scale, and letting an unregularised cross-entropy fit raise it
    # is how the readout ended up with |logit| ~ 880. The warm start only
    # needs to point the readout in the right direction; PPO can raise the
    # temperature later if confidence actually helps.
    opt = torch.optim.Adam(
        list(policy.readout.parameters()) +
        list(policy.sensory_head.parameters()) +
        [policy.sensory_scale, policy.action_bias, policy.action_gain],
        lr=0.01)
    for _ in range(epochs):
        logits, _ = policy(x, raw)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        accuracy = (policy(x, raw)[0].argmax(1) == y).float().mean()
    return float(accuracy.cpu())
