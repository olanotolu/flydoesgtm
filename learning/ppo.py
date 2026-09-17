import torch
from torch.distributions import Categorical

LR = 0.01
PPO_EPOCHS = 3
MINIBATCH = 4096
CLIP = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5


def generalized_advantage(rewards, values, dones, gamma=0.99, lam=0.95):
    """GAE with an explicit reset at account/episode boundaries."""
    adv = torch.zeros_like(rewards)
    carry = torch.zeros((), device=rewards.device)
    next_value = torch.zeros((), device=rewards.device)
    for t in range(len(rewards) - 1, -1, -1):
        not_done = (~dones[t]).to(rewards.dtype)
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        carry = delta + gamma * lam * not_done * carry
        adv[t] = carry
        next_value = values[t]
    return adv, adv + values


def ppo_update(policy, opt, traj):
    feat, obs, act = traj["feat"], traj["obs"], traj["act"]
    logp_old, ret, val_old = traj["logp"], traj["ret"], traj["val"]
    action_mask = traj.get("action_mask")

    adv = traj.get("adv")
    if adv is None:
        adv = ret - val_old
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = len(feat)
    for _ in range(PPO_EPOCHS):
        perm = torch.randperm(n, device=feat.device)
        for i in range(0, n, MINIBATCH):
            b = perm[i:i + MINIBATCH]
            logits, value = policy(feat[b], obs[b])
            if action_mask is not None:
                logits = logits.masked_fill(~action_mask[b], -1e9)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act[b])

            ratio = torch.exp(logp - logp_old[b])
            s1 = ratio * adv[b]
            s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv[b]
            policy_loss = -torch.min(s1, s2).mean()
            value_loss = (value - ret[b]).pow(2).mean()
            entropy = dist.entropy().mean()
            imitation = 0.0
            if "teacher" in traj:
                imitation = torch.nn.functional.cross_entropy(
                    logits, traj["teacher"][b])

            loss = policy_loss + VALUE_COEF * value_loss \
                - ENTROPY_COEF * entropy + 0.2 * imitation
            opt.zero_grad()
            loss.backward()
            opt.step()
