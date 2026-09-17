"""Observation -> current injection into sensory populations.

obs row k describes account lanes[k] — lane index == account index ==
batch fly index. Inactive lanes get zero current (their fly idles on
noise). Amounts are volts added per step; the LIF threshold is 1.0 and
resting tonic is 0.14, so ~0.3-0.8 is a strong but non-saturating
drive.
"""
import numpy as np

DEFAULT_GAIN = 0.6


class Encoder:
    def __init__(self, channel_map, gains=None):
        self.ch = channel_map
        self.gains = (np.full(16, DEFAULT_GAIN, np.float32)
                      if gains is None
                      else np.asarray(gains, np.float32))

    def inject(self, obs, lanes, batch):
        """obs (m, 16) -> inject list for FlyBrain.step()."""
        inject = []
        for c, pop in self.ch.items():
            amt = np.zeros(batch, np.float32)
            amt[lanes] = obs[:, c] * self.gains[c]
            if np.any(amt):
                inject.append((pop, amt))
        return inject

    def adapt(self, obs, advantage, lr=0.01):
        """Reward-weighted sensory-gain calibration.

        Brain dynamics remain frozen. This is a small black-box interface
        update: channels correlated with positive advantage get more drive;
        channels correlated with negative advantage get less. It is the
        minimal learnable sensory interface that does not backprop through
        the biological simulator.
        """
        x = np.asarray(obs, np.float32)
        a = np.asarray(advantage, np.float32).reshape(-1)
        if len(x) != len(a) or not len(x):
            return
        centered = x - x.mean(axis=0, keepdims=True)
        delta = np.mean(centered * a[:, None], axis=0)
        scale = np.mean(np.abs(delta)) + 1e-6
        self.gains = np.clip(self.gains + lr * delta / scale, 0.1, 2.0)
