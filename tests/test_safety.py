import numpy as np

from environment.clay_world import EMAIL, IGNORE, RESEARCH, World
from serving.clay_live import DemoBudget
from serving.contracts import demo_action


def test_world_ledger_conserves_every_event():
    world = World(n_accounts=8, days=12, seed=17)
    for day in range(world.days):
        world.start_day(day)
        _, idxs = world.observe(day)
        world.apply(day, idxs, np.full(len(idxs), RESEARCH),
                    slot_of=lambda k, d=day: d * 100 + k)
    world.end_episode()
    checks = world.ledger_invariants()
    assert checks["credit_conserved"]
    assert checks["econ_conserved"]
    assert checks["credit_events"] >= world.n


def test_budget_fallback_returns_actual_wait_action():
    world = World(n_accounts=2, days=1, daily_budget=1.0, seed=2)
    world.start_day(0)
    _, idxs = world.observe(0)
    executed = world.apply(0, idxs, np.full(len(idxs), EMAIL),
                           slot_of=lambda k: k)
    assert (executed == 0).all()


def test_policy_email_is_unsent_demo_draft():
    assert demo_action("EMAIL") == "DRAFT_EMAIL"
    assert demo_action("RESEARCH") == "RESEARCH"


def test_demo_budget_caps_records_and_enrichments():
    budget = DemoBudget(max_records=1, max_enrichments=1)
    budget.admit_record()
    budget.admit_enrichment("Enrich Company")
    budget.record(source="test", routine="demo", cost=1)
    try:
        budget.admit_record()
    except RuntimeError:
        pass
    else:
        raise AssertionError("record cap was not enforced")


def test_demo_budget_enrichment_cap_is_per_record():
    budget = DemoBudget(max_enrichments=2)
    budget.admit_enrichment("Enrich Company", "a")
    budget.admit_enrichment("Company News", "a")
    budget.admit_enrichment("Enrich Company", "b")
    try:
        budget.admit_enrichment("Company News", "a")
    except RuntimeError:
        pass
    else:
        raise AssertionError("per-record enrichment cap was not enforced")


def test_dense_control_is_parameter_matched():
    from learning.policy import FlyPolicy, MLPPolicy, matched_mlp_hidden, parameter_count
    target = parameter_count(FlyPolicy(12012))
    hidden, actual = matched_mlp_hidden(target)
    assert hidden >= 8
    assert abs(actual - target) / target <= 0.01
    assert parameter_count(MLPPolicy(hidden=hidden)) == actual
