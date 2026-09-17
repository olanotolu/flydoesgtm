"""Decaying spike traces, kept on the brain's device.

A scalar tau preserves the original API. A tuple exposes multiple
biological time constants. CUDA uses CuPy scatter-add in place; no
per-step `.get()` or CPU round-trip during training.
"""
import numpy as np


class BatchTrace:
    def __init__(self, brain, tracked, tau=0.15):
        self.device = brain.device
        self.xp = brain.xp
        self.slot = self.xp.full(brain.n, -1, dtype=np.int64)
        idx = np.asarray(tracked)
        self.slot[self.xp.asarray(idx)] = self.xp.arange(len(idx))
        self.F = len(idx)
        self.B = brain.batch
        taus = (tau,) if np.isscalar(tau) else tuple(tau)
        self.taus = taus
        self.n_scales = len(taus)
        shape = (self.F, self.B) if self.n_scales == 1 \
            else (self.n_scales, self.F, self.B)
        self.trace = self.xp.zeros(shape, dtype=self.xp.float32)
        self.decay = self.xp.asarray(
            np.exp(-brain.dt / np.asarray(taus)), dtype=self.xp.float32)

    def observe(self, brain):
        flat = brain.fired
        if self.n_scales == 1:
            self.trace *= self.decay[0]
            if len(flat):
                r, c = self.xp.divmod(flat, self.B)
                s = self.slot[r]
                ok = s >= 0
                self.xp.add.at(self.trace, (s[ok], c[ok]), 1.0)
            return self.trace

        self.trace *= self.decay[:, None, None]
        if len(flat):
            r, c = self.xp.divmod(flat, self.B)
            s = self.slot[r]
            ok = s >= 0
            for scale in range(self.n_scales):
                self.xp.add.at(self.trace[scale], (s[ok], c[ok]), 1.0)
        return self.trace

    def features(self, lanes=None):
        lanes = slice(None) if lanes is None else lanes
        if self.n_scales == 1:
            t = self.trace[:, lanes]
            return self.xp.log1p(t.T * 8.0)
        t = self.trace[:, :, lanes]
        # (scales, features, lanes) -> (lanes, scales * features)
        return self.xp.log1p(t * 8.0).transpose(2, 0, 1).reshape(
            -1, self.n_scales * self.F)

    def reset(self):
        self.trace[...] = 0
