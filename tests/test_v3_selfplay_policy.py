"""v3_selfplay.policy: the self-play actor-critic for Kexin's V3 game (her 74-dim state plus a role flag)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from v3_selfplay import policy as sp


def test_input_appends_a_role_flag_to_her_state():
    state = np.arange(74, dtype=np.float32)
    x_att, x_def = sp.policy_input(state, sp.ATTACKER), sp.policy_input(state, sp.DEFENDER)
    assert x_att.shape == (75,) and x_att[-1] == 1.0 and x_def[-1] == 0.0
    assert np.array_equal(x_att[:74], state) and state[-1] == 73.0          # the caller's state is untouched


@pytest.mark.parametrize("state, role", [(np.zeros(73), "attacker"), (np.zeros(74), "receiver")])
def test_input_refuses_another_layout_or_role(state, role):
    with pytest.raises(ValueError):
        sp.policy_input(state, role)


def test_network_has_a_shot_head_a_position_head_and_a_value():
    shots, positions, value = sp.RolePolicy()(torch.zeros(4, sp.INPUT_DIM))
    assert shots.shape == (4, 5) and positions.shape == (4, 3) and value.shape == (4,)


def _policy_preferring(shot, position):
    torch.manual_seed(0)
    net = sp.RolePolicy()
    with torch.no_grad():
        net.shot_head.bias.zero_()
        net.shot_head.bias[shot] = 50.0
        net.position_head.bias.zero_()
        net.position_head.bias[position] = 50.0
    return net


def test_decide_uses_the_head_of_its_role():
    net, x = _policy_preferring(shot=3, position=2), sp.policy_input(np.zeros(74), sp.ATTACKER)
    assert sp.decide(net, x, sp.ATTACKER, argmax=True)[0] == 3
    assert sp.decide(net, x, sp.DEFENDER, argmax=True)[0] == 2
    action, log_prob, _value = sp.decide(net, x, sp.ATTACKER)
    assert action == 3 and log_prob == pytest.approx(0.0, abs=1e-6)


def test_action_distribution_is_a_probability_vector_or_one_hot_for_argmax():
    net, x = sp.RolePolicy(), sp.policy_input(np.zeros(74), sp.DEFENDER)
    probs = sp.action_distribution(net, x, sp.DEFENDER)
    assert probs.shape == (3,) and probs.sum() == pytest.approx(1.0) and (probs > 0).all()
    greedy = sp.action_distribution(net, x, sp.DEFENDER, argmax=True)
    assert greedy.tolist().count(1.0) == 1 and greedy.sum() == 1.0


def test_her_sampler_accepts_our_probabilities():
    """Her evaluator draws with numpy's Generator.choice, which needs the probabilities to sum to 1 in float64."""
    rng = np.random.default_rng(0)
    for seed in range(50):
        torch.manual_seed(seed)
        net, state = sp.RolePolicy(), rng.normal(size=74) * 5
        for role in sp.ROLES:
            probs = sp.action_distribution(net, sp.policy_input(state, role), role)
            assert probs.dtype == np.float64 and abs(probs.sum() - 1.0) <= 1e-12
            rng.choice(len(probs), p=probs)


def test_checkpoints_round_trip_and_name_their_game(tmp_path):
    policies = {1: sp.RolePolicy(), 2: sp.RolePolicy()}
    path = str(tmp_path / "sp.pth")
    sp.save_policies(policies, path, {"seed": 3})
    loaded, meta = sp.load_policies(path)
    x = torch.randn(2, sp.INPUT_DIM)
    for p in (1, 2):
        assert torch.allclose(loaded[p](x)[0], policies[p](x)[0])
    assert meta["game"] == sp.GAME and meta["seed"] == 3 and meta["state_dim"] == 74


def test_loading_refuses_a_checkpoint_from_another_game(tmp_path):
    path = str(tmp_path / "designb.pth")
    sp.save_policies({1: sp.RolePolicy(), 2: sp.RolePolicy()}, path, {})
    meta_path = path + ".json"
    with open(meta_path) as f:
        text = f.read().replace(sp.GAME, "design_b")
    with open(meta_path, "w") as f:
        f.write(text)
    with pytest.raises(ValueError):
        sp.load_policies(path)
