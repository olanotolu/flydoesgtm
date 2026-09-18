"""Deterministic seed and artifact helpers shared by every experiment."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_commit(root: str | Path) -> str | None:
    """Local HEAD, falling back to FLY_GIT_COMMIT.

    Modal's `add_local_python_source` copies the package without `.git`, so
    the remote container cannot derive the commit itself. The entrypoint
    reads it from the local checkout and passes it through the environment,
    which keeps checkpoint and manifest provenance intact.
    """
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("FLY_GIT_COMMIT") or None


def state_checksum(state: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if torch.is_tensor(value):
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(repr((key, value)).encode())
    return h.hexdigest()


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True,
                                      default=str) + "\n")
