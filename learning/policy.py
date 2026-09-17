"""The engineered decision layer — the ONLY thing that learns.

features (log-squashed spike traces over DN/AN/KC/DAN populations)
    -> linear readout -> 6 action scores
WAIT is decoded from insufficient commitment (threshold on the max
score), value head on the same features for PPO, plus the two tricks
ported from C. Claygans: per-action bias/gain (compensates hub-hot
populations) and price margins (logits discounted by margin x today's
dollar price read off the observation channels).
"""
import numpy as np
import torch
import torch.nn as nn

ACTION_ORDER = ["OBSERVE", "RESEARCH", "ENRICH", "EMAIL",
                "ESCALATE", "IGNORE"]
# world order: WAIT=0 then ACTION_ORDER = 1..6

MARGIN_IDX = [2, 3, 4, 5]                 # RESEARCH ENRICH EMAIL ESCALATE
BASE_COSTS = torch.tensor([2.0, 1.0, 5.0, 15.0])


class FlyPolicy(nn.Module):
    def __init__(self, n_feat, wait_threshold=0.05):
        super().__init__()
        self.readout = nn.Linear(n_feat, 6)
        self.sensory_head = nn.Linear(16, 6, bias=False)
        nn.init.zeros_(self.sensory_head.weight)
        self.sensory_scale = nn.Parameter(torch.tensor(0.25))
        self.value_head = nn.Linear(n_feat, 1)
        self.action_bias = nn.Parameter(torch.zeros(7))
        self.action_gain = nn.Parameter(torch.ones(7))
        # cost-scaled margins for the four expensive actions, applied to
        # the CURRENT price read straight from obs channels 10-13.
        self.margins = nn.Parameter(torch.zeros(4))
        self.wait_threshold = wait_threshold

    def forward(self, feat, obs):
        """feat (m, F) trace features, obs (m, 16) raw channels
        (the margins need today's prices — economics is an input too)."""
        scores = self.readout(feat) + self.sensory_scale * self.sensory_head(obs)
        wait = self.wait_threshold - scores.max(dim=1, keepdim=True).values
        logits = torch.cat([wait, scores], dim=1) \
            * self.action_gain + self.action_bias
        cost_now = obs[:, 10:14] * BASE_COSTS.to(feat.device)
        logits[:, MARGIN_IDX] -= self.margins * cost_now
        value = self.value_head(feat).squeeze(-1)
        return logits, value


class MLPPolicy(nn.Module):
    """Parameter-matched dense baseline — no brain anywhere."""

    def __init__(self, n_in=16, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, 7)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, feat, obs):
        h = self.net(obs)            # baseline sees the raw observation
        return self.head(h), self.value_head(h).squeeze(-1)
