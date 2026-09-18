"""Deterministic teacher used only to warm-start the engineered readout.

The teacher previously opened with a disqualification rule:

    if negative > 0.78 or (intent < 0.28 and hiring < 0.35):
        return IGNORE

That branch has been removed. `IGNORE` sets `world.active[i] = False`
(environment/clay_world.py), so it does not merely decline *this* day's
decision -- it forecloses every future decision on that account, and it
does so at zero cost, which makes it look free to the optimiser. The
teacher fired it on 26.3% of decisions, and the accounts it wrote off
still converted. Measured over 5 seeds (128 accounts x 45 days), the
teacher returns 1372.7 mean economics, and every variant that keeps the
account alive returns about twice that --

    teacher as-is             1372.7   1.00x
    IGNORE -> WAIT            2934.5   2.14x
    IGNORE -> OBSERVE         2846.0   2.07x
    IGNORE -> RESEARCH        2901.2   2.11x
    IGNORE branch deleted     2995.0   2.18x

The replacement barely matters, which is the tell: the cost is not in
which action was chosen instead, it is in the deactivation. So the rule
is deleted rather than retuned, and the remaining rules decide.

This matters beyond the teacher's own score. The teacher is the warm-start
target and the imitation term during PPO, so a teacher 2.2x off the
achievable return both anchors the policy low and actively fights it
learning better. It is a scaffold, not an oracle: a real run should beat
it, and a `--no-raw-head` run should beat it using the fly alone.
"""
import numpy as np

from environment.clay_world import EMAIL, OBSERVE, RESEARCH, WAIT


def rules_action(obs):
    """A conservative, inspectable qualification rule for one 16ch row."""
    funding, hiring, intent, job_change, negative, trigger = obs[:6]
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
