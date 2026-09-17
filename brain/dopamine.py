"""Reward events -> real dopaminergic populations.

PAM = appetitive DAN cluster, PPL1 = aversive (the cluster the
fly-plays-Doom project pulsed on damage). A reward event injects a
pulse into the matching cluster — real spikes in real DANs, visible in
the visualization. Plasticity lives on our readout synapses
(learning/hebbian.py), never on the fly's own wiring — see
ARCHITECTURE.md for the honesty contract.
"""
import numpy as np

PULSE = 0.8          # volts; above resting tonic, below a hard drive
PULSE_STEPS = 3


def rpe_pulse(brain, dans, rpe):
    """Return inject pairs for a dopamine pulse scaled by |RPE|.

    rpe > 0 -> PAM (appetitive), rpe < 0 -> PPL1 (aversive).
    Returns [] for tiny RPE so the viz doesn't flicker on noise.
    """
    if abs(rpe) < 0.5:
        return []
    pop = dans["reward"] if rpe > 0 else dans["punish"]
    if len(pop) == 0:
        return []
    amt = min(1.5, PULSE * (0.5 + abs(rpe) / 50.0))
    return [(pop, amt)]
