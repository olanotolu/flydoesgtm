"""Connectome/readout ablation definitions.

The expensive full benchmark is intentionally opt-in. This module makes
all comparison arms explicit so a result cannot silently change what
"fly policy" means.
"""
import argparse
import json

import numpy as np


ARMS = {
    "fly": "frozen MaleCNS + learned readout",
    "random": "uniform random action",
    "rules": "handwritten signal threshold",
    "mlp": "parameter-matched dense MLP on raw obs",
    "shuffled": "same fly weights, shuffled neuron indices",
    "lesion_kc": "fly with KC feature population disabled",
    "lesion_dan": "fly with PPL1/PAM traces disabled",
    "pooled": "named DN motor pools only",
}


def shuffle_features(features, seed=7):
    """Destroy feature topology while preserving the observed values."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(features.shape[-1])
    return features[..., order], order


def lesion_features(features, indices):
    """Zero selected readout features; does not mutate the input."""
    out = np.array(features, copy=True)
    out[..., np.asarray(indices)] = 0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.list:
        print(json.dumps(ARMS, indent=2))
    else:
        print("Run each arm on identical seeds/worlds; report pure-econ "
              "return, spend, meetings, closed, spam, and return/$.")


if __name__ == "__main__":
    main()
