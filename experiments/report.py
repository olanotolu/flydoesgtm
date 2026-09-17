"""Paired bootstrap and Holm correction for the B200 evidence report."""
from __future__ import annotations

import numpy as np


def paired_bootstrap_delta(a, b, *, n=10_000, seed=0):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("paired samples must be non-empty one-dimensional arrays")
    delta = a - b
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), size=(n, len(delta)))].mean(1)
    return float(delta.mean()), tuple(np.quantile(samples, [0.025, 0.975]))


def holm_bonferroni(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, values[index] * (len(values) - rank)))
        adjusted[index] = running
    return adjusted.tolist()
