"""Deterministic teacher used only to warm-start the engineered readout."""
import numpy as np

from environment.clay_world import (EMAIL, IGNORE, OBSERVE, RESEARCH,
                                     WAIT)


def rules_action(obs):
    """A conservative, inspectable qualification rule for one 16ch row."""
    funding, hiring, intent, job_change, negative, trigger = obs[:6]
    if negative > 0.78 or (intent < 0.28 and hiring < 0.35):
        return IGNORE
    if intent > 0.72 and hiring > 0.55 and negative < 0.45:
        return EMAIL
    if (0.35 <= intent <= 0.72 or 0.35 <= hiring <= 0.65) \
            and obs[8] < 0.5:
        return RESEARCH
    if trigger > 0.75 or job_change > 0.75:
        return OBSERVE
    return WAIT


def teacher_actions(obs):
    """Vectorized labels for an (n, 16) observation matrix."""
    return np.asarray([rules_action(row) for row in np.asarray(obs)],
                      dtype=np.int64)
