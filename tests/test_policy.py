"""Policy mechanics — WAIT decode, margins, shapes."""
import numpy as np
import torch

from learning.policy import FlyPolicy, MLPPolicy, ACTION_ORDER


def test_wait_wins_when_no_commitment():
    p = FlyPolicy(100)
    with torch.no_grad():
        p.readout.weight.zero_()
        p.readout.bias.zero_()
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
        p.readout.weight.zero_(); p.readout.bias.zero_()
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
