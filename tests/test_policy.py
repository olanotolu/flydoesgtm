"""Policy mechanics — WAIT decode, margins, shapes."""
import numpy as np
import torch

from learning.policy import FlyPolicy, MLPPolicy, ACTION_ORDER


def test_wait_wins_when_no_commitment():
    p = FlyPolicy(100)
    with torch.no_grad():
        p.readout.weight.zero_()
        p.action_bias.zero_()
    feat = torch.zeros(3, 100)
    obs = torch.zeros(3, 16)
    logits, v = p(feat, obs)
    assert logits.shape == (3, 7)
    # all scores 0 -> wait logit = +threshold -> WAIT wins
    assert logits.argmax(1).eq(0).all()


def test_margins_discount_expensive_actions():
    p = FlyPolicy(50)
    with torch.no_grad():
        p.margins.fill_(1.0)     # strong price sensitivity
        p.readout.weight.zero_()
        p.action_bias.zero_()
    feat = torch.ones(2, 50)
    cheap = torch.zeros(2, 16)
    dear = torch.zeros(2, 16)
    dear[:, 10:14] = 2.0          # prices at 2x base
    l_cheap, _ = p(feat, cheap)
    l_dear, _ = p(feat, dear)
    for a in (2, 3, 4, 5):  # RESEARCH ENRICH EMAIL ESCALATE
        assert (l_dear[:, a] < l_cheap[:, a]).all()
    assert torch.allclose(l_dear[:, 0], l_cheap[:, 0])  # WAIT unmoved


def test_fly_policy_has_explicit_sensory_calibration_head():
    p = FlyPolicy(20)
    feat = torch.zeros(2, 20)
    obs = torch.ones(2, 16)
    logits, _ = p(feat, obs)
    assert logits.shape == (2, 7)
    assert p.sensory_head.weight.shape == (6, 16)


def test_mlp_baseline_shape():
    p = MLPPolicy(16, 64)
    logits, v = p(torch.zeros(5, 1000), torch.zeros(5, 16))
    assert logits.shape == (5, 7) and v.shape == (5,)


def test_gae_bootstraps_and_resets_at_done():
    from learning.ppo import generalized_advantage
    rewards = torch.tensor([1.0, 0.0, 3.0, 0.0])
    values = torch.tensor([0.5, 0.5, 1.0, 0.0])
    dones = torch.tensor([False, True, False, True])
    adv, ret = generalized_advantage(rewards, values, dones,
                                     gamma=0.9, lam=1.0)
    assert torch.allclose(adv, torch.tensor([0.5, -0.5, 2.0, 0.0]))
    assert torch.allclose(ret, adv + values)


def test_teacher_anchor_exempts_ignore_rows():
    """Exempt teacher-IGNORE rows contribute exactly zero imitation loss
    and zero imitation gradient; the rest of the batch stays anchored."""
    from environment.clay_world import IGNORE
    from learning.ppo import teacher_anchor
    logits = torch.randn(4, 7, requires_grad=True)
    teacher = torch.tensor([IGNORE, 4, IGNORE, 0])
    loss = teacher_anchor(logits, teacher, ignore_anchor_exempt=True)
    # identical to CE computed on only the non-IGNORE rows
    assert torch.allclose(
        loss, torch.nn.functional.cross_entropy(
            logits.detach()[[1, 3]], teacher[[1, 3]]))
    loss.backward()
    assert logits.grad[0].eq(0).all()    # IGNORE rows: no imitation grad
    assert logits.grad[2].eq(0).all()
    assert logits.grad[1].ne(0).any()    # non-IGNORE rows still anchored
    assert logits.grad[3].ne(0).any()


def test_teacher_anchor_all_ignore_batch_is_zero_loss():
    """A minibatch that is all teacher-IGNORE must not produce a NaN or
    a nonzero imitation term — it contributes nothing at all."""
    from environment.clay_world import IGNORE
    from learning.ppo import teacher_anchor
    logits = torch.randn(3, 7, requires_grad=True)
    teacher = torch.full((3,), IGNORE)
    loss = teacher_anchor(logits, teacher, ignore_anchor_exempt=True)
    assert loss.item() == 0.0
    loss.backward()                      # differentiable-safe, zero grads
    assert logits.grad.eq(0).all()


def test_teacher_anchor_default_anchors_every_row():
    """Backward compat: without the flag every row is anchored,
    including teacher-IGNORE rows (the historical behaviour)."""
    from environment.clay_world import IGNORE
    from learning.ppo import teacher_anchor
    logits = torch.randn(4, 7)
    teacher = torch.tensor([IGNORE, 4, IGNORE, 0])
    assert torch.allclose(
        teacher_anchor(logits, teacher),
        torch.nn.functional.cross_entropy(logits, teacher))


def _tiny_traj(n=8, teacher=None):
    traj = {"feat": torch.randn(n, 8), "obs": torch.rand(n, 16),
            "act": torch.randint(0, 7, (n,)), "logp": torch.zeros(n),
            "ret": torch.randn(n), "val": torch.zeros(n)}
    if teacher is not None:
        traj["teacher"] = teacher
    return traj


def test_ppo_update_exempt_makes_all_ignore_teacher_a_noop():
    """End to end through ppo_update: with the exempt flag on an
    all-IGNORE teacher batch the imitation term vanishes, so even a huge
    coefficient moves the policy exactly as a zero coefficient would."""
    from environment.clay_world import IGNORE
    from learning.ppo import ppo_update
    torch.manual_seed(0)
    traj = _tiny_traj(teacher=torch.full((8,), IGNORE))
    p1, p2 = FlyPolicy(8), FlyPolicy(8)
    p2.load_state_dict(p1.state_dict())
    o1 = torch.optim.Adam(p1.parameters(), lr=0.01)
    o2 = torch.optim.Adam(p2.parameters(), lr=0.01)
    # n < MINIBATCH -> one full-batch minibatch per epoch, so randperm
    # order cannot differ between the two updates.
    ppo_update(p1, o1, traj, imitation_coef=0.0, ignore_anchor_exempt=True)
    ppo_update(p2, o2, traj, imitation_coef=100.0,
               ignore_anchor_exempt=True)
    for a, b in zip(p1.parameters(), p2.parameters()):
        assert torch.allclose(a, b)


def test_ppo_update_runs_without_teacher_labels():
    """Backward compat: a trajectory with no teacher labels still
    updates (imitation term absent, exempt flag irrelevant)."""
    from learning.ppo import ppo_update
    policy = FlyPolicy(8)
    opt = torch.optim.Adam(policy.parameters(), lr=0.01)
    ppo_update(policy, opt, _tiny_traj(), ignore_anchor_exempt=True)
    assert all(torch.isfinite(p).all() for p in policy.parameters())


def test_sensor_interface_adapts_gains_from_advantage():
    from learning.encoder import Encoder
    enc = Encoder({0: np.array([0]), **{i: np.array([i]) for i in range(1, 16)}})
    before = enc.gains.copy()
    obs = np.zeros((2, 16), np.float32); obs[:, 0] = [0.1, 1.0]
    enc.adapt(obs, np.array([-1.0, 1.0], np.float32), lr=0.1)
    assert enc.gains[0] > before[0]


def test_dopamine_readout_changes_only_engineered_weights():
    from learning.hebbian import Hebbian
    h = Hebbian(3, 2, lr=0.1)
    h.mark(np.array([1.0, 0.0, 0.0], np.float32), 1)
    delta = h.reward(2.0)
    assert delta[0, 1] > 0 and delta[1, 0] == 0
