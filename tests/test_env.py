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
    """The teacher must not disqualify. IGNORE deactivates the account
    permanently, and the teacher's old disqualification branch cost it
    ~2.2x of achievable economics (see environment/teacher.py). A cold
    row now falls through to WAIT, which keeps the option alive."""
    from environment.teacher import rules_action
    cold = np.zeros(16, np.float32); cold[2] = 0.1
    hot = np.zeros(16, np.float32); hot[1:3] = 0.9
    assert rules_action(cold) == WAIT
    assert rules_action(hot) == EMAIL
    assert rules_action(cold) != IGNORE


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


def test_rent_charges_only_live_accounts():
    """Attention rent accrues each day an account is held; IGNORE stops
    the bleed and pays the calibrated kill bonus."""
    w = make(n_accounts=5, days=10, seed=3, rent=0.5, kill_bonus=2.0)
    w.intent[:] = 0.1  # all dead accounts -> kill bonus ~1.8 each
    w.start_day(0)
    idxs = np.arange(5)
    w.apply(0, idxs, np.array([WAIT, WAIT, WAIT, WAIT, IGNORE]),
            slot_of=lambda k: k)
    # day 0: 5 accounts each paid 0.5 rent; ignored account also got bonus
    econ0 = sum(w.econ.values())
    assert abs(econ0 - (5 * -0.5 + 2.0 * (1 - 0.1))) < 0.05
    assert w.stats["ignored"] == 1
    assert w.stats["rent_paid"] == 2.5
    # day 1: only the 4 live accounts pay rent
    w.start_day(1)
    live = np.where(w.active)[0]
    assert len(live) == 4
    w.apply(1, live, np.full(4, WAIT), slot_of=lambda k: 10 + k)
    assert w.stats["rent_paid"] == 2.5 + 4 * 0.5


def test_expire_all_kills_cold_accounts():
    """With expire_all, a cold unpursued account past its window pays the
    expiry penalty instead of parking forever."""
    w = make(n_accounts=10, days=10, seed=4, expire_all=True)
    w.intent[:] = 0.1   # cold — old rule would never expire these
    w.win_end[:] = 2
    w.start_day(0)
    idxs = np.where(w.active)[0]
    w.apply(0, idxs, np.full(len(idxs), WAIT), slot_of=lambda k: k)
    for d in range(1, 6):
        w.start_day(d)
        live = np.where(w.active)[0]
        if len(live):
            w.apply(d, live, np.full(len(live), WAIT),
                    slot_of=lambda k: 100 * d + k)
    assert not w.active.any()
    assert sum(w.econ.values()) < 0


def test_kill_penalty_turns_ignore_against_hot_accounts():
    """IGNORE nets kill_bonus*(1-intent) - kill_penalty*intent, written
    to both ledgers: a hot account's kill goes negative once the
    penalty bites."""
    w = make(n_accounts=2, days=10, seed=7, kill_bonus=3.0,
             kill_penalty=4.0)
    w.start_day(0)
    w.intent[0] = 0.95   # hot: 3*0.05 - 4*0.95 = -3.65
    w.intent[1] = 0.05   # cold: 3*0.95 - 4*0.05 = +2.65
    w.apply(0, np.array([0, 1]), np.array([IGNORE, IGNORE]),
            slot_of=lambda k: k)
    assert abs(w.credit[0] + 3.65) < 1e-5
    assert abs(w.econ[0] + 3.65) < 1e-5
    assert w.credit[1] > 0 and w.econ[1] > 0
    # penalty alone (no bonus) still writes the negative kill entry
    w2 = make(n_accounts=1, days=10, seed=9, kill_penalty=2.0)
    w2.start_day(0)
    w2.intent[0] = 0.9
    w2.apply(0, np.array([0]), np.array([IGNORE]), slot_of=lambda k: 0)
    assert abs(w2.credit[0] + 1.8) < 1e-5


def test_kill_penalty_symmetric_bonus_penalty():
    """kill_bonus == kill_penalty makes the write-off (1-2*intent)*bonus:
    killing a dead account still pays, killing a hot one costs."""
    w = make(n_accounts=2, days=10, seed=8, kill_bonus=3.0,
             kill_penalty=3.0)
    w.start_day(0)
    w.intent[0] = 1.0
    w.intent[1] = 0.0
    w.apply(0, np.array([0, 1]), np.array([IGNORE, IGNORE]),
            slot_of=lambda k: k)
    assert w.credit[0] < 0 and w.econ[0] < 0
    assert w.credit[1] > 0 and w.econ[1] > 0


def test_ep_mult_scales_stage_episodes_without_mutating_full(
        monkeypatch, tmp_path):
    """run() must scale each stage's episode count by ep_mult on a
    copy — the shared FULL curriculum stays canonical."""
    import pytest
    import torch
    from unittest.mock import MagicMock
    from learning import train

    calls = []

    class Done(Exception):
        pass

    def fake_episode(*a, **k):
        calls.append(1)
        if len(calls) >= 40:   # FULL[0] has 20 episodes; ep_mult=2 -> 40
            raise Done
        z = torch.zeros(1)
        return {"obs": z, "adv": z, "reward": z}

    policy = MagicMock()
    policy.parameters.return_value = [torch.nn.Parameter(torch.zeros(2))]
    policy.to.return_value = policy
    monkeypatch.setattr(train, "get_brain", lambda **k: MagicMock())
    monkeypatch.setattr(train, "build_channel_map", lambda b: {})
    monkeypatch.setattr(train, "tracked_set", lambda b, m: [])
    monkeypatch.setattr(train, "Encoder", lambda *a, **k: MagicMock())
    monkeypatch.setattr(train, "BatchTrace", lambda *a, **k: MagicMock())
    monkeypatch.setattr(train, "FlyPolicy", lambda *a, **k: policy)
    monkeypatch.setattr(train, "run_episode", fake_episode)
    monkeypatch.setattr(train, "ppo_update", lambda *a: None)
    monkeypatch.setattr(train, "evaluate", lambda *a, **k: (0.0, {}))
    monkeypatch.setenv("CLAYFLY_WARM_START", "0")
    stage = dict(train.FULL[0])
    with pytest.raises(Done):
        train.run([stage], out_dir=str(tmp_path), ep_mult=2.0)
    assert len(calls) == 40
    assert stage["episodes"] == 20
    assert train.FULL[0]["episodes"] == 20
