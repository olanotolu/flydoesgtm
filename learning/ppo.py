import torch
from torch.distributions import Categorical

LR = 0.01
PPO_EPOCHS = 3
MINIBATCH = 4096
CLIP = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
# The readout used to be a plain Linear over ~12,000 non-negative log1p
# spike traces, so its output grew with the feature count: measured logits
# reached ~1e3 against returns of ~3. PPO's importance ratio is
# exp(logp - logp_old), which overflows float32 once logits move ~88 apart,
# so the second update turned the whole policy to NaN.
#
# That is now fixed at the source, in learning.policy: a cosine readout plus
# a tanh squash bounds every logit to +-LOGIT_MAX by construction. The
# clamp that used to sit in `ppo_update` has been deleted rather than left
# in place, because it was masking the real bug and, once the logits were
# genuinely bounded, could never fire again. Gradient clipping is kept as
# ordinary PPO hygiene, not as the fix.
GRAD_CLIP = 1.0
# Weight on the teacher cross-entropy anchor inside the PPO loss. The
# teacher (environment/teacher.py) emits IGNORE only on corroborated
# dead accounts, so the anchor now teaches the write-off boundary
# instead of pinning the IGNORE logit at ~0. `learning.train` can decay
# it per-episode; `ppo_update` keeps it constant by default.
IMITATION_COEF = 0.2


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


def ppo_update(policy, opt, traj, imitation_coef=IMITATION_COEF):
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
            # Already bounded to +-LOGIT_MAX inside the policy; no clamp here.
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
                - ENTROPY_COEF * entropy + imitation_coef * imitation
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
            opt.step()
