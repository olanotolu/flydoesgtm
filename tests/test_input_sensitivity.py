"""The fly must be able to tell accounts apart.

A policy can only be as good as the features it reads. This is the
regression test for the failure that shipped a constant "YES": at gain 0.6
and 4 sim steps only 3.4% of trace features ever fired, a linear probe on
those features could not separate four account qualities (52.8%, chance
25%), and the readout duly emitted the same action for every input.

These tests assert the configured training/serving point sits above that
dead zone, and pin the measurement that justifies the defaults in
`learning/encoder.py` and `learning/rollout.py`. If someone reverts the
gain or the horizon, `test_dead_zone_stays_dead` fails and says why.
"""
import numpy as np
import pytest

from brain.loader import get_brain
from brain.populations import build_channel_map, tracked_set
from environment.clay_world import ACTION_COST
from learning.encoder import DEFAULT_GAIN, Encoder
from learning.rollout import SIM_STEPS
from learning.trace import BatchTrace

LEVELS = (0.05, 0.35, 0.65, 0.95)      # dead, weak, strong, maxed
# 16 per level (64 samples) keeps the probe stable enough to gate on while
# staying inside a few seconds of brain time. Below ~8 per level the split
# noise swamps the signal — a 24-sample run reported 57.8% for a setting
# that measures 85.7% at 80 samples. Use experiments/sensitivity.py for the
# real numbers; this test only guards the dead zone.
N_PER_LEVEL = 16
SIGMA = 0.08
CHANCE = 1.0 / len(LEVELS)
DEAD_GAIN, DEAD_STEPS = 0.6, 4         # the configuration that shipped
MIN_PROBE = 0.55                        # comfortably above chance (25%)
BASE_COSTS = np.array([2.0, 1.0, 5.0, 15.0], np.float32)


@pytest.fixture(scope="module")
def rig():
    brain = get_brain(batch=1, device="cpu")
    channels = build_channel_map(brain)
    trace = BatchTrace(brain, tracked_set(brain, channels),
                       tau=(0.05, 0.15, 0.5))
    return brain, trace, channels


@pytest.fixture(scope="module")
def samples(rig):
    """Collect each (steps, gain) configuration once, not once per test."""
    cache = {}

    def get(steps, gain):
        key = (steps, gain)
        if key not in cache:
            cache[key] = _collect(rig, steps, gain)
        return cache[key]

    return get


def _collect(rig, steps, gain, seed=1234):
    brain, trace, channels = rig
    enc = Encoder(channels, gains=np.full(16, gain, np.float32))
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    for label, level in enumerate(LEVELS):
        for _ in range(N_PER_LEVEL):
            obs = np.zeros((1, 16), np.float32)
            obs[0, :6] = np.clip(np.full(6, level)
                                 + rng.normal(0, SIGMA, 6), 0, 1)
            obs[0, 6] = 1.0
            obs[0, 7] = 0.5
            obs[0, 10:14] = ACTION_COST[[2, 3, 4, 5]] / BASE_COSTS
            obs[0, 14] = 1.0
            brain.reset()
            trace.reset()
            for _ in range(steps):
                brain.step(inject=enc.inject(obs, np.array([0]), 1))
                trace.observe(brain)
            feats.append(np.asarray(trace.features([0])).reshape(-1)
                         .astype(np.float64))
            labels.append(label)
    return np.stack(feats), np.array(labels)


def _probe(feats, labels, n_splits=100):
    """Mean ridge-probe accuracy over random splits.

    Solved in the dual: the feature dimension is thousands, the sample
    count is tens, so an n x n system is both correct and cheap.
    """
    accs = []
    for s in range(n_splits):
        rng = np.random.default_rng(s)
        order = rng.permutation(len(labels))
        cut = int(len(labels) * 0.6)
        train, test = order[:cut], order[cut:]
        mu, sd = feats[train].mean(0), feats[train].std(0) + 1e-8
        xtr = (feats[train] - mu) / sd
        xte = (feats[test] - mu) / sd
        targets = np.eye(len(LEVELS))[labels[train]]
        alpha = np.linalg.solve(xtr @ xtr.T + 1e-2 * np.eye(len(train)),
                                targets)
        accs.append(float((((xte @ xtr.T) @ alpha).argmax(1)
                           == labels[test]).mean()))
    return float(np.mean(accs))


def test_configured_horizon_populates_the_trace(samples):
    """At the configured point most tracked features must actually fire."""
    feats, _ = samples(SIM_STEPS, DEFAULT_GAIN)
    occupancy = float((feats != 0).any(axis=0).mean())
    assert occupancy > 0.30, (
        f"only {occupancy:.1%} of trace features ever fired at "
        f"{SIM_STEPS} steps / gain {DEFAULT_GAIN}. The readout is reading "
        f"an idle brain; see experiments/sensitivity.py.")


def test_configured_point_separates_account_quality(samples):
    """A linear readout must be able to tell account levels apart."""
    feats, labels = samples(SIM_STEPS, DEFAULT_GAIN)
    accuracy = _probe(feats, labels)
    assert accuracy > MIN_PROBE, (
        f"linear probe reached only {accuracy:.1%} at {SIM_STEPS} steps / "
        f"gain {DEFAULT_GAIN} (chance {CHANCE:.0%}). The frozen features do "
        f"not carry account quality there, so no readout can discriminate "
        f"and any confident output is a constant.")


def test_dead_zone_stays_dead(samples):
    """Pin the measurement that justifies the defaults.

    gain 0.6 / 4 steps is the configuration that shipped a constant answer.
    If this ever stops being the worse of the two, the defaults need
    re-measuring rather than assuming.
    """
    dead_feats, dead_labels = samples(DEAD_STEPS, DEAD_GAIN)
    live_feats, live_labels = samples(SIM_STEPS, DEFAULT_GAIN)
    dead = _probe(dead_feats, dead_labels)
    live = _probe(live_feats, live_labels)
    assert dead < live, (
        f"the dead zone (gain {DEAD_GAIN} / {DEAD_STEPS} steps, {dead:.1%}) "
        f"is no longer worse than the configured point "
        f"(gain {DEFAULT_GAIN} / {SIM_STEPS} steps, {live:.1%}). "
        f"Re-run experiments/sensitivity.py before changing defaults.")


def test_checkpoint_records_its_own_serving_config():
    """A checkpoint must carry the horizon and gains it was trained at.

    `serving/serve.py` reads both back out of the checkpoint, which is what
    stops a train/serve horizon mismatch from silently producing a constant.
    """
    from pathlib import Path

    import torch

    path = Path(__file__).parents[1] / "results" / "fly_policy.pt"
    if not path.exists():
        pytest.skip("no trained checkpoint present")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert "sim_steps" in checkpoint, "checkpoint omits its horizon"
    assert "encoder_gains" in checkpoint, "checkpoint omits its gains"
    assert len(checkpoint["encoder_gains"]) == 16
