"""Tests for per-player, role-gated Design B self-play (paper Sec. 4.2):
nash_skills.v2.selfplay_rollout and nash_skills.v2.selfplay_per_player."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
import torch.nn as nn

from nash_skills.v2 import selfplay_rollout as sr
from nash_skills.v2.selfplay_attack_defend import MetaPolicy
from tests.test_collect_decision_timing import TimedFakeEnv
from tests.test_collect_data_attack_defend import TWO_DECISION_SCRIPT


class TestLearnerSchedule:
    def test_rounds_alternate_between_players(self):
        got = [sr.learner_for_iteration(it, round_iters=3) for it in range(1, 13)]
        assert got == [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2]

    def test_round_start(self):
        assert [sr.is_round_start(it, 3) for it in range(1, 8)] == [True, False, False, True, False, False, True]


class TestRoleHeadLogProb:
    def test_uses_the_attack_head_for_attack_decisions_and_defend_head_otherwise(self):
        attack_logits = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.3, 0.1]] * 2))
        defend_logits = torch.log(torch.tensor([[0.5, 0.25, 0.25]] * 2))
        actions = torch.tensor([2, 0])
        is_attack = torch.tensor([True, False])
        lp, ent = sr.role_head_log_prob_entropy(attack_logits, defend_logits, actions, is_attack)
        assert lp[0].item() == pytest.approx(np.log(0.3), abs=1e-5)
        assert lp[1].item() == pytest.approx(np.log(0.5), abs=1e-5)
        h_attack = -sum(p * np.log(p) for p in [0.1, 0.2, 0.3, 0.3, 0.1])
        h_defend = -sum(p * np.log(p) for p in [0.5, 0.25, 0.25])
        assert ent[0].item() == pytest.approx(h_attack, abs=1e-5)
        assert ent[1].item() == pytest.approx(h_defend, abs=1e-5)

    def test_attack_index_beyond_the_defend_head_is_safe(self):
        lp, _ = sr.role_head_log_prob_entropy(torch.zeros(1, 5), torch.zeros(1, 3),
                                              torch.tensor([4]), torch.tensor([True]))
        assert lp[0].item() == pytest.approx(np.log(1 / 5), abs=1e-5)


class _FixedPolicy(nn.Module):
    """Always strongly prefers attack index 3 and defend index 2; value 0.25."""

    def forward(self, x):
        n = x.shape[0]
        a = torch.full((n, 5), -20.0)
        a[:, 3] = 20.0
        d = torch.full((n, 3), -20.0)
        d[:, 2] = 20.0
        return a, d, torch.full((n,), 0.25)


class _ZeroPPO:
    def predict(self, _obs, deterministic=True):
        return np.zeros(9, dtype=np.float32), None


class RecordingTimedEnv(TimedFakeEnv):
    def __init__(self, script):
        super().__init__(script)
        self.skill_calls, self.defend_calls = [], []

    def set_skills(self, s1, s2):
        super().set_skills(s1, s2)
        self.skill_calls.append((s1, s2))

    def set_defend_preparation_positions(self, p1, p2, calibrated=False):
        super().set_defend_preparation_positions(p1, p2, calibrated)
        self.defend_calls.append((p1, p2))


def _rally(learner=1):
    env = RecordingTimedEnv(TWO_DECISION_SCRIPT)
    policies = {1: _FixedPolicy(), 2: _FixedPolicy()}
    out = sr.play_rally(env, _ZeroPPO(), policies, learner=learner, device=torch.device("cpu"),
                        defend_mechanism="preparation", max_steps=100, rng=np.random.default_rng(0),
                        init_choice=((0, 0), (0, 0)))
    return env, out


class TestPlayRally:
    def test_learner_records_one_role_head_decision_per_exchange(self):
        _env, out = _rally(learner=1)
        # exchange 1: ball to P2 (P2 attacks, P1 defends); exchange 2: P1 attacks
        assert out["roles"] == ["defend", "attack"]
        assert out["actions"] == [2, 3]
        assert len(out["states"]) == 2 and out["states"][0].shape == (90,)
        assert out["old_values"] == pytest.approx([0.25, 0.25])

    def test_only_the_live_half_changes_at_each_decision(self):
        env, _out = _rally()
        # after decision 1: P2's attack -> right_short (3); P1's attack stays left (0)
        assert env.skill_calls[1] == ("left", "right_short")
        # P1's preparation -> index 2 (center; order is left, right, center); P2's stays left (0)
        assert env.defend_calls[1] == ("center", "left")
        # decision 2: P1 attacks (3); P2 prepares (2)
        assert env.skill_calls[2] == ("right_short", "right_short")
        assert env.defend_calls[2] == ("center", "center")

    def test_terminal_reward_follows_player_identity(self):
        _env, out1 = _rally(learner=1)
        _env, out2 = _rally(learner=2)
        assert out1["reward"] == 1.0 and out2["reward"] == -1.0
        assert out1["done"] and out1["bootstrap_value"] == 0.0

    def test_player2_learner_records_its_own_roles(self):
        _env, out = _rally(learner=2)
        assert out["roles"] == ["attack", "defend"]


class CrossingOnlyEnv(RecordingTimedEnv):
    """The role tracker flips at net crossings, but the ball x-velocity the contact clock reads stays 0,
    so contact timing never fires a decision here."""

    def step(self, action):
        out = super().step(action)
        self._env.data.qvel[-6] = 0.0
        return out


def _rally_timed(timing, learner=1):
    env = CrossingOnlyEnv(TWO_DECISION_SCRIPT)
    kw = {} if timing is None else {"decision_timing": timing}
    out = sr.play_rally(env, _ZeroPPO(), {1: _FixedPolicy(), 2: _FixedPolicy()}, learner=learner,
                        device=torch.device("cpu"), defend_mechanism="preparation", max_steps=100,
                        rng=np.random.default_rng(0), init_choice=((0, 0), (0, 0)), **kw)
    return env, out


class TestDecisionTiming:
    def test_crossing_timing_decides_at_net_crossings_with_the_receiver_attacking(self):
        _env, out = _rally_timed("crossing")
        assert out["roles"] == ["defend", "attack"] and out["exchanges"] == 2

    def test_contact_timing_stays_the_default_and_needs_the_contact_signal(self):
        _env, out = _rally_timed(None)
        assert out["roles"] == [] and out["exchanges"] == 0

    def test_unknown_timing_is_refused(self):
        with pytest.raises(ValueError):
            _rally_timed("bounce")


class TestCheckpoint:
    def test_sidecar_records_the_decision_timing_it_was_given(self, tmp_path):
        path = str(tmp_path / "sp.pth")
        sr.save_per_player({1: MetaPolicy(), 2: MetaPolicy()}, path, {"decision_timing": "crossing"})
        assert sr.load_per_player(path)[1]["decision_timing"] == "crossing"

    def test_round_trip_with_metadata(self, tmp_path):
        policies = {1: MetaPolicy(), 2: MetaPolicy()}
        path = str(tmp_path / "sp.pth")
        sr.save_per_player(policies, path, {"skill_profile": "moderate_aggressive"})
        loaded, meta = sr.load_per_player(path)
        assert meta["per_player"] is True and meta["decision_timing"] == "contact"
        assert meta["skill_profile"] == "moderate_aggressive"
        for p in (1, 2):
            for a, b in zip(policies[p].parameters(), loaded[p].parameters()):
                assert torch.equal(a, b)

    def test_refuses_a_shared_policy_checkpoint(self, tmp_path):
        from nash_skills.v2.selfplay_attack_defend import save_checkpoint
        path = str(tmp_path / "old.pth")
        save_checkpoint(MetaPolicy(), path)
        with pytest.raises(ValueError, match="per-player"):
            sr.load_per_player(path)
