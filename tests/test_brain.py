"""Brain interface invariants — real populations, frozen weights."""
import numpy as np
import pytest
import torch

from brain.loader import get_brain, weights_checksum
from brain.populations import (build_channel_map, tracked_set,
                               dan_populations, action_pools)
from learning.encoder import Encoder
from learning.trace import BatchTrace


@pytest.fixture(scope="session")
def brain():
    return get_brain(batch=4)


@pytest.fixture(scope="session")
def ch_map(brain):
    return build_channel_map(brain)


def test_channel_map_covers_16_nonempty(brain, ch_map):
    assert len(ch_map) == 16
    for c, idx in ch_map.items():
        assert len(idx) > 0, f"channel {c} resolved to zero neurons"
        assert idx.max() < brain.n


def test_channels_are_disjoint(ch_map):
    allidx = np.concatenate(list(ch_map.values()))
    assert len(np.unique(allidx)) == len(allidx)


def test_tracked_set_contents(brain, ch_map):
    tr = tracked_set(brain, ch_map)
    dn = brain.cells(["descending_neuron"])
    assert np.isin(dn, tr).all()
    assert len(tr) > 3000


def test_dan_and_pools_exist(brain):
    dans = dan_populations(brain)
    assert len(dans["punish"]) > 0 and len(dans["reward"]) > 0
    pools = action_pools(brain)
    assert all(len(v) > 0 for v in pools.values())


def test_injection_actually_fires(brain, ch_map):
    """A hot observation must produce more tracked spikes than a cold
    one — otherwise the senses are dead."""
    enc = Encoder(ch_map)
    tr = BatchTrace(brain, tracked_set(brain, ch_map))
    hot = np.ones((4, 16), np.float32)
    cold = np.zeros((4, 16), np.float32)
    lanes = np.arange(4)

    brain.reset(); tr.reset()
    for _ in range(10):
        brain.step(inject=enc.inject(hot, lanes, 4))
        tr.observe(brain)
    hot_total = tr.trace.sum()

    brain.reset(); tr.reset()
    for _ in range(10):
        brain.step(inject=enc.inject(cold, lanes, 4))
        tr.observe(brain)
    cold_total = tr.trace.sum()
    assert hot_total > cold_total * 1.2


def test_multiscale_trace_exposes_three_time_constants(brain, ch_map):
    trace = BatchTrace(brain, tracked_set(brain, ch_map),
                       tau=(0.05, 0.15, 0.5))
    assert trace.n_scales == 3
    brain.reset(); trace.reset()
    brain.step(inject=Encoder(ch_map).inject(
        np.ones((4, 16), np.float32), np.arange(4), 4))
    trace.observe(brain)
    assert trace.features(np.arange(4)).shape == (4, trace.F * 3)


def test_weights_stay_frozen(brain):
    before = weights_checksum()
    for _ in range(5):
        brain.step()
    assert weights_checksum() == before
