"""Tests for nash_skills.v2.selfplay_per_player (PPO update, batching, training loop)."""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from nash_skills.v2 import selfplay_per_player as spp
from nash_skills.v2.selfplay_attack_defend import MetaPolicy
from tests.test_selfplay_per_player import RecordingTimedEnv, _ZeroPPO
from tests.test_collect_data_attack_defend import TWO_DECISION_SCRIPT


def _args(**kw):
    base = dict(gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, ppo_epochs=4, minibatch_size=8,
                value_coef=0.5, entropy_coef=0.0, lr=1e-2)
    base.update(kw)
    return SimpleNamespace(**base)


def _attack_only_batch(n=8, action=3, adv=1.0):
    return {"states": np.zeros((n, 90), dtype=np.float32), "roles": ["attack"] * n, "actions": [action] * n,
            "old_log_probs": [float(np.log(1 / 5))] * n, "advantages": [adv] * n, "returns": [1.0] * n}


class TestPPOUpdate:
    def test_positive_advantage_raises_the_taken_attack_probability(self):
        torch.manual_seed(0)
        policy = MetaPolicy()
        x = torch.zeros(1, 90)
        before = torch.softmax(policy(x)[0], -1)[0, 3].item()
        batch = _attack_only_batch(n=8)
        batch["actions"] = [3, 0] * 4  # action 3 was good, action 0 bad
        batch["advantages"] = [1.0, -1.0] * 4
        spp.ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=1e-2), batch, _args(value_coef=0.0))
        after = torch.softmax(policy(x)[0], -1)[0, 3].item()
        assert after > before

    def test_attack_only_batch_never_touches_the_preparation_head(self):
        torch.manual_seed(0)
        policy = MetaPolicy()
        w, b = policy.defend_head.weight.clone(), policy.defend_head.bias.clone()
        spp.ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=1e-2),
                       _attack_only_batch(), _args(entropy_coef=0.05))
        assert torch.equal(policy.defend_head.weight, w)
        assert torch.equal(policy.defend_head.bias, b)

    def test_empty_batch_returns_nan_stats(self):
        stats = spp.ppo_update(MetaPolicy(), None, {"states": np.zeros((0, 90)), "roles": [], "actions": [],
                                                   "old_log_probs": [], "advantages": [], "returns": []}, _args())
        assert np.isnan(stats["policy_loss"])


class TestBatching:
    def test_gae_runs_per_rally_with_the_terminal_reward_last(self):
        rally = {"states": [np.zeros(90)] * 2, "roles": ["defend", "attack"], "actions": [1, 2],
                 "old_log_probs": [-1.0, -1.5], "old_values": [0.0, 0.0], "reward": 1.0,
                 "bootstrap_value": 0.0}
        batch = spp.build_batch([rally, rally], gamma=1.0, gae_lambda=1.0)
        assert batch["advantages"] == pytest.approx([1.0, 1.0, 1.0, 1.0])
        assert batch["roles"] == ["defend", "attack"] * 2

    def test_rallies_without_learner_decisions_are_skipped(self):
        empty = {"states": [], "roles": [], "actions": [], "old_log_probs": [], "old_values": [],
                 "reward": -1.0, "bootstrap_value": 0.0}
        assert spp.build_batch([empty], 0.99, 0.95)["advantages"] == []


def test_training_loop_alternates_learners_and_saves_both_policies(tmp_path, monkeypatch):
    monkeypatch.setattr(spp, "make_env", lambda args: RecordingTimedEnv(TWO_DECISION_SCRIPT))
    monkeypatch.setattr(spp, "load_ppo", lambda path: _ZeroPPO())
    out = str(tmp_path / "sp.pth")
    args = spp.build_parser().parse_args([
        "--iterations", "4", "--rallies-per-iter", "2", "--round-iters", "2", "--output", out,
        "--log", str(tmp_path / "log.csv"), "--seed", "0", "--defend-mechanism", "preparation",
        "--save-every", "2", "--snapshot-every", "1"])
    spp.train(args)
    meta = json.load(open(out + ".json"))
    assert meta["per_player"] and meta["skill_profile"] == "aggressive"
    rows = open(tmp_path / "log.csv").read().strip().splitlines()
    learners = [r.split(",")[1] for r in rows[1:]]
    assert learners == ["1", "1", "2", "2"]
