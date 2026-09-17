"""FlyBrain loading + the one brain process-wide callers share.

sensory_input=False is load-bearing: it deletes synapses onto sensory
neurons so injected channel current is the fly's actual sensory world
(with it, olfactory receptors run away on recurrent excitation and
drown everything we inject).
"""
import hashlib
import os
from pathlib import Path

import numpy as np
from scipy import sparse


def get_brain(batch=1, device=None, seed=64):
    from flybrain import FlyBrain
    return FlyBrain(device=device or os.environ.get("FLY_DEVICE", "cpu"),
                    batch=batch, seed=seed, sensory_input=False)


def weights_checksum(data=None):
    """sha256 of the loaded weight matrix — the frozen-biology invariant."""
    from flybrain.data import DATA, ensure_data
    d = ensure_data(data or DATA)
    W = sparse.load_npz(Path(d) / "weights.npz")
    h = hashlib.sha256()
    for a in (W.indptr, W.indices, W.data):
        h.update(a.tobytes())
    return h.hexdigest()
