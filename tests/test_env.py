"""Environment invariants — the economy has to be honest before the
fly's decisions mean anything."""
import numpy as np

from environment.clay_world import (
    World, ACTIONS, ACTION_COST, WAIT, OBSERVE, RESEARCH, ENRICH, EMAIL,
    ESCALATE, IGNORE, OUTCOME_REWARD)


def make(**kw):
    kw.setdefault("n_accounts", 40)
    kw.setdefault("days", 20)
    return World(**kw)


def test_budget_cap_enforced():
    """Daily spend can never exceed the cap, even if the policy tries."""
    w = make(daily_budget=10.0)
    w.start_day(0)
    obs, idxs = w.observe(0)
    # try to ESCALATE ($15) every account with a $10 cap
    slot = [0]
    w.apply(0, idxs, np.full(len(idxs), ESCALATE),
            slot_of=lambda k: slot.__setitem__(0, slot[0] + 1) or slot[0])
    assert w.spend_today <= 10.0
    assert w.stats["escalations"] == 0   # everything fell back to WAIT


def test_research_reveals_pain_channel():
    w = make(seed=3)
    w.start_day(0)
    i = 0
    w.apply(0, np.array([i]), np.array([RESEARCH]), slot_of=lambda k: k)
    assert w.researched[i]
    assert w.sigma[i] < 0.1
    obs, _ = w.observe(0)
    assert not np.isclose(obs[0, 15], 0.5)  # pain revealed, not unknown


def test_observe_never_certain():
    """OBSERVE asymptotes at sigma floor — certainty is never free."""
    w = make(seed=1)
    for d in range(15):
        w.start_day(d)
        w.apply(d, np.array([0]), np.array([OBSERVE]), slot_of=lambda k: k)
    assert w.sigma[0] >= 0.35 - 1e-6


def test_ignore_drops_account():
    w = make(seed=2)
    w.start_day(0)
    w.apply(0, np.array([0]), np.array([IGNORE]), slot_of=lambda k: k)
    assert not w.active[0]


def test_delayed_rewards_resolve():
    """An EMAIL that lands a reply must credit the deciding slot later."""
    w = make(n_accounts=5, days=30, seed=11)
    # force a hot account
    w.intent[:] = 0.95
    w.icp[:] = 0.95
    w.urgency[:] = 0.9
    for d in range(30):
        w.start_day(d)
        obs, idxs = w.observe(d)
        act = np.where(w.researched[idxs], EMAIL, RESEARCH)
        w.apply(d, idxs, act, slot_of=lambda k: int(idxs[k]) * 100 + d)
    w.end_episode()
    assert sum(w.credit.values()) > 0
    assert w.stats["replies"] + w.stats["unsubscribes"] \
        + w.stats["spam"] > 0


def test_price_regimes_vary():
    w = make(seed=5, vary_costs=True)
    scheds = set()
    for d in range(w.days):
        scheds.add(tuple(np.round(w.cost_sched[d], 3)))
    assert len(scheds) > 1


def test_reward_profiles_change_only_outcome_values():
    gentle = make(reward_profile="gentle_spam")
    assert gentle.outcome_reward["spam"] == -15.0
    assert gentle.outcome_reward["closed"] == OUTCOME_REWARD["closed"]


def test_teacher_qualifies_cold_and_hot_rows():
    from environment.teacher import rules_action
    cold = np.zeros(16, np.float32); cold[2] = 0.1
    hot = np.zeros(16, np.float32); hot[1:3] = 0.9
    assert rules_action(cold) == IGNORE
    assert rules_action(hot) == EMAIL


def test_liars_exist_and_inflate_funding():
    """The claimed-funding channel must be able to lie."""
    w = make(n_accounts=200, seed=9, liar_frac=0.3)
    liars = np.where(w.inflate > 0)[0]
    assert len(liars) > 0
    assert (w.intent[liars] < 0.5).all()
    w.start_day(0)
    i = int(liars[0])
    obs = w._visible_signals(i)
    # funding channel inflated beyond what intent alone could produce
    assert obs[0] > 0.3
