"""Deterministic teacher used only to warm-start the engineered readout.

The teacher previously opened with a disqualification rule:

    if negative > 0.78 or (intent < 0.28 and hiring < 0.35):
        return IGNORE

That branch was removed because in the v1 economy `IGNORE` only cost:
it sets `world.active[i] = False` (environment/clay_world.py), so it
does not merely decline *this* day's decision -- it forecloses every
future decision on that account, at zero immediate payoff. Fired on
26.3% of decisions, mostly on a single noisy channel read, it wrote off
accounts that still converted and halved the teacher's return.

The v2 economy changed the ledger (environment/clay_world.py:334-346):
IGNORE now pays kill_bonus*(1-intent) - kill_penalty*intent, expire_all
charges -3 to any unpursued account that lives past its buying window,
and rent accrues every day an account stays active. A confirmed-dead
account is therefore worth more dead (+~2.4 at intent~0.2) than parked
(-0.05/day bleed, then -3 at expiry). So the write-off branch is
restored -- calibrated, this time:

    cold     = intent < 0.25 and negative > 0.4
               and funding < 0.45 and hiring < 0.45
    qualified = cold and (weak-fit confirmed or deep into the episode)

Every market channel must agree the account is dead before the teacher
touches it -- the old branch's failures were single-channel misreads at
sigma=0.45, and a corroborating quorum is what separates "dead" from
"read cold once". The qualifier then demands evidence beyond the read:
either RESEARCH ran and revealed no hidden upside (pain < 0.25 -- the
account's best case was inspected and rejected), or the episode is past
60% (day_frac), where an account still reading dead on every channel is
overwhelmingly likely past its buying window and marching toward the
-3 expiry.

Measured over 15 seeds (128 accounts x 45 days, v2 economy), driving
the world with teacher actions:

    teacher, no IGNORE          2786.5   0 ign/ep
    IGNORE branch restored      2890.5  30.7 ign/ep
                                       killed intent mean 0.196
                                       8.7% of kills intent > 0.5
                                       pays +2.41/kill, fires 0.82% of rows

The kill rate is ~30x lower than the deleted branch's, the accounts it
writes off average true intent 0.20, and closed-won count is unchanged
(25.9 vs 25.9). Accounts it cannot see clearly -- hot, warm, or merely
ambiguous -- still fall through to EMAIL/RESEARCH/OBSERVE/WAIT.

This matters beyond the teacher's own score. The teacher is the warm-start
target and the imitation term during PPO, so a teacher that never emits
IGNORE pins the policy's IGNORE logit at ~0 no matter what the ledger
pays. The restored branch makes the write-off reachable; PPO decides how
far past this conservative boundary the economics push it. The teacher is
a scaffold, not an oracle: a real run should beat it, and a
`--no-raw-head` run should beat it using the fly alone.
"""
import numpy as np

from environment.clay_world import EMAIL, IGNORE, OBSERVE, RESEARCH, WAIT


def rules_action(obs):
    """A conservative, inspectable qualification rule for one 16ch row."""
    funding, hiring, intent, job_change, negative, trigger = obs[:6]
    # Calibrated write-off (see module docstring). All four market
    # channels must agree the account is dead -- a lone noisy read never
    # kills -- and either RESEARCH confirmed weak fit (pain low) or the
    # episode is deep enough that a still-cold account is almost surely
    # past its buying window and paying rent toward a -3 expiry.
    if intent < 0.25 and negative > 0.4 and funding < 0.45 \
            and hiring < 0.45 \
            and ((obs[8] > 0.5 and obs[15] < 0.25) or obs[7] > 0.6):
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
