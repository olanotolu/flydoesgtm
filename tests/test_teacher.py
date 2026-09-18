"""The restored IGNORE branch: calibrated write-offs only.

The teacher kills an account only when every market channel reads dead
AND the write-off is qualified -- research confirmed weak fit, or the
episode is deep enough that a still-cold account is past its window and
paying rent toward a -3 expiry. Hidden state is pinned and sigma zeroed
so the observable row is exact.
"""
import numpy as np

from environment.clay_world import World, EMAIL, IGNORE
from environment.teacher import rules_action, teacher_actions


def _world(days=10, **kw):
    kw.setdefault("n_accounts", 4)
    kw.setdefault("rent", 0.05)
    kw.setdefault("kill_bonus", 3.0)
    kw.setdefault("expire_all", True)
    w = World(days=days, **kw)
    w.sigma[:] = 0.0          # exact reads, no observation noise
    w.inflate[:] = 0.0
    w.win_end[:] = days - 1   # keep accounts alive; expiry isn't under test
    return w


def _pin(w, i, intent, urgency, icp, champion=0.5, competitor=0.5):
    """Pin hidden state AFTER start_day (market events mutate it there)."""
    w.intent[i] = intent
    w.urgency[i] = urgency
    w.icp[i] = icp
    w.champion[i] = champion
    w.competitor[i] = competitor


def test_dead_researched_weak_fit_is_written_off():
    """Researched, every channel cold, pain absent -> IGNORE pays
    kill_bonus*(1-intent) ~ +2.7 instead of rent bleed toward -3."""
    w = _world(seed=1)
    i = 0
    w.researched[i] = True
    w.pain[i] = 0.1
    w.start_day(0)
    _pin(w, i, 0.1, 0.1, 0.1, 0.5, 0.1)
    obs, idxs = w.observe(0)
    row = obs[int(np.where(idxs == i)[0][0])]
    assert row[8] == 1.0 and row[15] < 0.25     # researched, weak fit
    assert rules_action(row) == IGNORE


def test_dead_unresearched_is_ignored_only_late():
    """Unresearched cold reads may be noise -- the teacher waits until
    day_frac > 0.6 before writing off on sight alone."""
    w = _world(days=45, seed=2)
    i = 1
    w.start_day(0)
    _pin(w, i, 0.1, 0.1, 0.1, 0.5, 0.1)
    obs, idxs = w.observe(0)
    early = obs[int(np.where(idxs == i)[0][0])]
    assert early[7] < 0.6
    assert rules_action(early) != IGNORE        # too early to be sure

    w.start_day(28)                             # day_frac 28/44 = 0.64
    _pin(w, i, 0.1, 0.1, 0.1, 0.5, 0.1)
    obs, idxs = w.observe(28)
    late = obs[int(np.where(idxs == i)[0][0])]
    assert late[7] > 0.6 and late[8] == 0.0
    assert rules_action(late) == IGNORE


def test_cold_look_without_quorum_survives():
    """A single cold channel is not a dead account: corroboration is
    required, so one noisy read can never kill."""
    w = _world(days=45, seed=3)
    i = 2
    # low intent channel but funding looks fine (liar-style inflation)
    w.start_day(30)
    _pin(w, i, 0.05, 0.95, 0.9, 0.5, 0.9)
    w.inflate[i] = 0.5
    obs, idxs = w.observe(30)
    row = obs[int(np.where(idxs == i)[0][0])]
    # intent channel = 0.75*0.05 + 0.25*0.95 = 0.275 -> not < 0.25 either
    assert rules_action(row) != IGNORE


def test_hot_account_is_never_written_off():
    """Hot accounts stay on the pursue track even late and researched."""
    w = _world(days=45, seed=4)
    i = 3
    w.researched[i] = True
    w.pain[i] = 0.8                              # strong pain hypothesis
    w.start_day(30)
    _pin(w, i, 0.9, 0.9, 0.9, 0.5, 0.2)
    obs, idxs = w.observe(30)
    row = obs[int(np.where(idxs == i)[0][0])]
    assert rules_action(row) != IGNORE


def test_teacher_actions_emits_some_ignore_not_all():
    """The BC label mix: a dead world produces IGNORE labels while a
    hot world produces none."""
    dead = _world(days=45, seed=5)
    dead.researched[:] = True
    dead.pain[:] = 0.1
    dead.start_day(30)
    for i in range(dead.n):
        _pin(dead, i, 0.05, 0.05, 0.05, 0.5, 0.05)
    obs, _ = dead.observe(30)
    labels = teacher_actions(obs)
    assert (labels == IGNORE).all()

    hot = _world(days=45, seed=6)
    hot.start_day(30)
    for i in range(hot.n):
        _pin(hot, i, 0.9, 0.9, 0.9, 0.5, 0.2)
    obs, _ = hot.observe(30)
    labels = teacher_actions(obs)
    assert not (labels == IGNORE).any()
    assert (labels == EMAIL).any()
