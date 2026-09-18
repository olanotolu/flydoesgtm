"""FlyBrain loading + the one brain process-wide callers share.

sensory_input=False is load-bearing: it deletes synapses onto sensory
neurons so injected channel current is the fly's actual sensory world
(with it, olfactory receptors run away on recurrent excitation and
drown everything we inject).
"""
import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
from scipy import sparse


def get_brain(batch=1, device=None, seed=64, data=None):
    from flybrain import FlyBrain
    return FlyBrain(data=data,
                    device=device or os.environ.get("FLY_DEVICE", "cpu"),
                    batch=batch, seed=seed, sensory_input=False)


def make_shuffled_connectome(out_dir, data=None, seed=7):
    """Random-reservoir control: same cells, same channels, same
    out-degree and outgoing weight multiset per neuron — only the
    postsynaptic targets are globally re-permuted. Idempotent."""
    out = Path(out_dir)
    if (out / "weights.npz").exists():
        return out
    from flybrain.data import DATA, ensure_data
    d = Path(ensure_data(data or DATA))
    W = sparse.load_npz(d / "weights.npz").tocsc()   # cols = presynaptic
    rng = np.random.default_rng(seed)
    W.indices = rng.permutation(W.indices)           # retarget all synapses
    W.sum_duplicates()
    W.sort_indices()
    out.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out / "weights.npz", W.tocsr())
    shutil.copyfile(d / "brain.npz", out / "brain.npz")
    return out


def weights_checksum(data=None, brain=None):
    """sha256 of the frozen weights, preferring the in-memory brain array."""
    if brain is not None and hasattr(brain, "weights"):
        h = hashlib.sha256()
        h.update(np.asarray(brain.weights).tobytes())
        return h.hexdigest()
    from flybrain.data import DATA, ensure_data
    d = ensure_data(data or DATA)
    W = sparse.load_npz(Path(d) / "weights.npz")
    h = hashlib.sha256()
    for a in (W.indptr, W.indices, W.data):
        h.update(a.tobytes())
    return h.hexdigest()
