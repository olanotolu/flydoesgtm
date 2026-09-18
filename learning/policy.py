"""The engineered decision layer — the ONLY thing that learns.

features (log-squashed spike traces over DN/AN/KC/DAN populations)
    -> cosine readout -> 6 action scores
WAIT is decoded from insufficient commitment (threshold on the max
score), value head on the same features for PPO, plus the two tricks
ported from C. Claygans: per-action bias/gain (compensates hub-hot
populations) and price margins (logits discounted by margin x today's
dollar price read off the observation channels).

Why the readout is a **cosine** map and not a plain Linear
---------------------------------------------------------
A plain Linear over ~12,000 non-negative log1p spike traces has no
reason to keep its output in a usable range, and it does not: measured
|logit| reaches ~880. That makes softmax one-hot and makes PPO's
exp(logp - logp_old) overflow float32 (which needs only ~88). Normalising
the weight rows and the feature vector bounds the cosine term to [-1, 1],
so the score is `temperature * cos` and the logit scale is a parameter we
control rather than an accident of feature count.

Three details are load-bearing for that bound to actually hold:

1. The readout has **no bias**. A bias is added *after* the cosine and is
   unbounded, so `temperature * (cos + bias)` would leak the whole thing
   back to infinity. The per-action offset this would have provided is
   already carried by `action_bias` below.
2. `temperature` is sigmoid-bounded by TEMP_MAX, so the optimiser cannot
   simply raise it to reintroduce the overflow.
3. The final logits are squashed by `LOGIT_MAX * tanh(logits / LOGIT_MAX)`.
   This is the unconditional guarantee: it survives action_gain, margins
   and the raw sensory head, none of which the cosine bound covers. It is
   near-identity in the working range and caps every logit at LOGIT_MAX,
   so a pathological parameter vector yields finite saturated logits
   instead of inf -> NaN.

   Be precise about what that buys. Measured derivative of the squash:
   |x| <= 30 -> >= 0.42 (healthy), 60-150 -> thin, >= 300 -> exactly 0.
   So the squash is *not* a licence to let the pre-squash value run away;
   it is a backstop. The reason it works here is that the cosine readout
   already caps the score at +-TEMP_MAX = +-20, so the squash only ever
   sees values in its healthy band. The auxiliary terms it is backstopping
   (action_gain, action_bias, margins, sensory head) all start at 0 or 1
   and move slowly. An earlier version of this file claimed the squash
   keeps gradient "at the boundary"; that was wrong, and the measurement
   above is what the claim should have been.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_ORDER = ["OBSERVE", "RESEARCH", "ENRICH", "EMAIL",
                "ESCALATE", "IGNORE"]
# world order: WAIT=0 then ACTION_ORDER = 1..6

MARGIN_IDX = [2, 3, 4, 5]                 # RESEARCH ENRICH EMAIL ESCALATE
BASE_COSTS = torch.tensor([2.0, 1.0, 5.0, 15.0])
EPS = 1e-6
TEMP_MAX = 20.0        # ceiling on the learnable logit temperature
LOGIT_MAX = 30.0       # hard bound on every emitted logit

# Stamped into every checkpoint. The readout's semantics have changed twice
# (linear -> cosine -> bounded cosine) and the weights are not portable
# between them: a linear readout encoded its logit scale in |W|, which a
# cosine readout discards. Loading across versions would "work" and produce
# arbitrary decisions, so the loader compares this string instead of
# guessing from which keys happen to be present.
READOUT_ARCH = "cosine-v2"


def _temp_raw(effective):
    """Inverse of TEMP_MAX * sigmoid(.) so `temperature=5.0` means 5.0."""
    frac = min(max(float(effective) / TEMP_MAX, 1e-4), 1 - 1e-4)
    return float(np.log(frac / (1.0 - frac)))


class FlyPolicy(nn.Module):
    def __init__(self, n_feat, wait_threshold=0.05, use_raw_head=True,
                 temperature=5.0):
        super().__init__()
        self.use_raw_head = bool(use_raw_head)
        # bias=False is deliberate — see the module docstring. A bias would
        # be added after the cosine and would be unbounded.
        self.readout = nn.Linear(n_feat, 6, bias=False)
        self.sensory_head = nn.Linear(16, 6, bias=False)
        nn.init.zeros_(self.sensory_head.weight)
        self.sensory_scale = nn.Parameter(torch.tensor(0.25))
        self.value_head = nn.Linear(n_feat, 1)
        self.action_bias = nn.Parameter(torch.zeros(7))
        self.action_gain = nn.Parameter(torch.ones(7))
        # Raw, pre-sigmoid. `forward` maps it into (0, TEMP_MAX) so the
        # policy can calibrate its own confidence but cannot run away.
        self.temperature = nn.Parameter(torch.tensor(_temp_raw(temperature)))
        # cost-scaled margins for the four expensive actions, applied to
        # the CURRENT price read straight from obs channels 10-13.
        self.margins = nn.Parameter(torch.zeros(4))
        self.wait_threshold = wait_threshold

    def logit_temperature(self):
        """The effective temperature, bounded by TEMP_MAX."""
        return TEMP_MAX * torch.sigmoid(self.temperature)

    def forward(self, feat, obs):
        """feat (m, F) trace features, obs (m, 16) raw channels
        (the margins need today's prices — economics is an input too)."""
        weight = self.readout.weight
        unit_weight = weight / (weight.norm(dim=1, keepdim=True) + EPS)
        unit_feat = feat / (feat.norm(dim=1, keepdim=True) + EPS)
        scores = self.logit_temperature() * F.linear(unit_feat, unit_weight)
        if self.use_raw_head:
            scores = scores + self.sensory_scale * self.sensory_head(obs)
        wait = self.wait_threshold - scores.max(dim=1, keepdim=True).values
        logits = torch.cat([wait, scores], dim=1) \
            * self.action_gain + self.action_bias
        cost_now = obs[:, 10:14] * BASE_COSTS.to(feat.device)
        logits[:, MARGIN_IDX] -= self.margins * cost_now
        # Smooth, unconditional bound. See the module docstring.
        logits = LOGIT_MAX * torch.tanh(logits / LOGIT_MAX)
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


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def matched_mlp_hidden(target_params, n_in=16, tolerance=0.01):
    """Choose a dense control width whose parameter count is within 1%."""
    candidates = []
    for hidden in range(8, 1025):
        count = parameter_count(MLPPolicy(n_in=n_in, hidden=hidden))
        candidates.append((abs(count - target_params), hidden, count))
    _, hidden, count = min(candidates)
    if abs(count - target_params) / max(1, target_params) > tolerance:
        raise ValueError("could not parameter-match MLP within 1%")
    return hidden, count
