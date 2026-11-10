"""v3_selfplay.ppo and v3_selfplay.train: PPO and the alternating-round schedule, as in our Design B self-play."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from v3_selfplay import policy as sp
from v3_selfplay import ppo
from v3_selfplay import train as tr


def test_gae_matches_a_hand_computation():
    adv, ret = ppo.compute_gae([0.0, 1.0], [0.5, 0.25], gamma=0.9, gae_lambda=0.5)
    d1 = 1.0 - 0.25
    d0 = 0.0 + 0.9 * 0.25 - 0.5
    assert adv == pytest.approx([d0 + 0.9 * 0.5 * d1, d1])
    assert ret == pytest.approx([adv[0] + 0.5, adv[1] + 0.25])


def test_gae_bootstraps_a_cut_rally_from_the_critic():
    adv, _ret = ppo.compute_gae([0.0], [0.2], gamma=0.9, gae_lambda=0.95, bootstrap_value=0.6)
    assert adv == pytest.approx([0.9 * 0.6 - 0.2])


def test_log_prob_comes_from_the_head_that_acted():
    shots = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.2, 0.2]]))
    positions = torch.log(torch.tensor([[0.5, 0.25, 0.25]]))
    lp_att, _ = ppo.role_log_prob_entropy(shots, positions, torch.tensor([2]), torch.tensor([True]))
    lp_def, _ = ppo.role_log_prob_entropy(shots, positions, torch.tensor([0]), torch.tensor([False]))
    assert float(lp_att) == pytest.approx(np.log(0.3)) and float(lp_def) == pytest.approx(np.log(0.5))


def _rally(action, reward, role=sp.DEFENDER):
    x = sp.policy_input(np.zeros(74), role)
    return {"states": [x], "roles": [role], "actions": [action], "old_log_probs": [float(np.log(1 / 3))],
            "old_values": [0.0], "reward": reward, "bootstrap_value": 0.0}


def _cfg(**kw):
    base = dict(ppo_epochs=4, minibatch_size=64, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.0)
    return SimpleNamespace(**{**base, **kw})


def test_an_update_moves_probability_toward_the_rewarded_position():
    torch.manual_seed(0)
    net = sp.RolePolicy()
    with torch.no_grad():
        net.position_head.weight.zero_()
        net.position_head.bias.zero_()
    rallies = [_rally(2, 1.0)] * 8 + [_rally(0, -1.0)] * 8
    batch = ppo.build_batch(rallies, gamma=0.99, gae_lambda=0.95)
    stats = ppo.ppo_update(net, torch.optim.Adam(net.parameters(), lr=1e-2), batch, _cfg())
    probs = sp.action_distribution(net, sp.policy_input(np.zeros(74), sp.DEFENDER), sp.DEFENDER)
    assert probs[2] > 1 / 3 > probs[0]
    assert all(np.isfinite(v) for v in stats.values())


def test_an_empty_batch_is_skipped():
    stats = ppo.ppo_update(sp.RolePolicy(), None, ppo.build_batch([], 0.99, 0.95), _cfg())
    assert all(np.isnan(v) for v in stats.values())


@pytest.mark.parametrize("it, learner, start", [(1, 1, True), (25, 1, False), (26, 2, True), (51, 1, True)])
def test_players_learn_in_alternating_rounds(it, learner, start):
    assert tr.learner_for_iteration(it, 25) == learner and tr.is_round_start(it, 25) == start


def test_rally_seeds_are_reproducible_and_distinct():
    a, b = tr.rally_seeds(0, it=3, n=32), tr.rally_seeds(0, it=3, n=32)
    assert a == b and len(set(a)) == 32 and set(a).isdisjoint(tr.rally_seeds(0, it=4, n=32))
    assert len(tr.rally_seeds(None, it=1, n=4)) == 4


def test_opponents_come_from_the_round_copy_or_the_pool():
    import random
    random.seed(0)
    assert tr.choose_opponents([], "round", 3, snapshot_prob=1.0) == ["round"] * 3
    assert set(tr.choose_opponents(["old"], "round", 50, snapshot_prob=1.0)) == {"old"}
    mixed = tr.choose_opponents(["old"], "round", 200, snapshot_prob=0.5)
    assert 60 < mixed.count("old") < 140


def test_a_worker_rebuilds_both_policies_and_plays_the_seeded_rally(monkeypatch):
    seen = {}

    def fake_rally(env, ppo_ctrl, policies, learner, game, max_steps, seed=None, decide_fn=None):
        seen.update(policies=policies, learner=learner, seed=seed, max_steps=max_steps, env=env)
        return {"ok": True}

    monkeypatch.setattr(tr, "play_rally", fake_rally)
    monkeypatch.setattr(tr, "_W", {"env": "env", "ppo": "ppo", "game": "game"})
    learner, opponent = sp.RolePolicy(), sp.RolePolicy()
    out = tr.run_job((2, learner.state_dict(), opponent.state_dict(), 1234, 600))
    x = torch.randn(3, sp.INPUT_DIM)
    assert out == {"ok": True} and seen["learner"] == 2 and seen["seed"] == 1234 and seen["max_steps"] == 600
    assert torch.allclose(seen["policies"][2](x)[0], learner(x)[0])
    assert torch.allclose(seen["policies"][1](x)[0], opponent(x)[0]) and seen["env"] == "env"


def test_training_runs_end_to_end_on_a_fake_game(tmp_path, monkeypatch):
    calls = []

    def fake_rally(env, ppo_ctrl, policies, learner, game, max_steps, seed=None, decide_fn=None):
        calls.append(learner)
        x = sp.policy_input(np.zeros(74), sp.ATTACKER)
        return {"states": [x], "roles": [sp.ATTACKER], "actions": [1], "old_log_probs": [-1.6],
                "old_values": [0.0], "reward": 1.0, "done": True, "winner": learner, "bootstrap_value": 0.0,
                "steps": 10, "exchanges": 1, "usage": {"attack": np.eye(5, dtype=int)[1], "defend": np.zeros(3, int)}}

    monkeypatch.setattr(tr, "play_rally", fake_rally)
    out = str(tmp_path / "sp.pth")
    args = tr.build_parser().parse_args(["--iterations", "4", "--rallies-per-iter", "2", "--round-iters", "2",
                                         "--snapshot-every", "2", "--seed", "0", "--output", out,
                                         "--log", str(tmp_path / "sp.csv")])
    tr.train(args, env=SimpleNamespace(close=lambda: None), ppo_ctrl=None, game=None)
    assert calls == [1, 1, 1, 1, 2, 2, 2, 2]
    policies, meta = sp.load_policies(out)
    assert set(policies) == {1, 2} and meta["iterations"] == 4 and meta["max_steps"] == 600
    with open(tmp_path / "sp.csv") as f:
        assert len(f.read().strip().splitlines()) == 5                    # header + one row per iteration
