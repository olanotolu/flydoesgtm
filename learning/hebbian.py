"""Three-factor Hebbian on the READOUT synapses — the dopamine ablation.

e_ij accumulates feature_i x action_j at decision time and decays;
when reward lands, W += lr * r * e. Plasticity lives exclusively on
our engineered interface — the fly's own wiring is untouched (see
ARCHITECTURE.md). Experimental arm; PPO is the primary learner.
"""
import numpy as np


class Hebbian:
    def __init__(self, n_feat, n_actions=7, lr=0.002, decay=0.92):
        self.e = np.zeros((n_feat, n_actions), np.float32)
        self.lr = lr
        self.decay = decay
        self.log = []

    def mark(self, feat, action):
        """Decision time: deposit eligibility, decay the rest."""
        self.e *= self.decay
        self.e[:, action] += feat

    def reward(self, r):
        """Reward time: strengthen/weaken whatever was recently active."""
        self.e *= self.lr * r
        self.log.append(float(np.abs(self.e).max()))
        return self.e.copy()
