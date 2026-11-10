"""Self-play state without the choice slots (Dvij's pi(s1, s2, role)): state_mode="no_choices"
drops both players' current attack / preparation choices (joint state [74:90]) from the actor input."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from nash_skills.v2 import selfplay_rollout as sr
from nash_skills.v2.selfplay_attack_defend import MetaPolicy
from tests.test_selfplay_per_player import RecordingTimedEnv, _ZeroPPO
from tests.test_collect_data_attack_defend import TWO_DECISION_SCRIPT


def _info():
    return {"diff_pos": np.zeros(3), "diff_quat": np.zeros(4), "target": np.zeros(7),
            "diff_pos_opp": np.zeros(3), "diff_quat_opp": np.zeros(4), "target_opp": np.zeros(7)}


def test_modes_and_dimensions():
    assert sr.STATE_MODES == ("joint", "no_choices")
    assert sr.state_dim_for("joint") == 90 and sr.state_dim_for("no_choices") == 74
    with pytest.raises(ValueError):
        sr.state_dim_for("partial")


def test_no_choices_state_is_the_joint_state_without_the_choice_slots():
    obs = np.random.default_rng(0).normal(size=116).astype(np.float32)
    obs[-2:] = [0.25, 0.75]                      # skill slots (valid one-hot positions)
    a = sr.encode_for_player(1, obs, _info(), {1: [0, 1], 2: [2, 2]}, state_mode="no_choices")
    b = sr.encode_for_player(1, obs, _info(), {1: [4, 0], 2: [1, 1]}, state_mode="no_choices")
    full = sr.encode_for_player(1, obs, _info(), {1: [0, 1], 2: [2, 2]})
    assert a.shape == (74,)
    np.testing.assert_array_equal(a, full[:74])
    np.testing.assert_array_equal(a, b)          # choices do not reach the actor


class _Fixed(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        assert x.shape[-1] == self.dim
        n = x.shape[0]
        return torch.zeros(n, 5), torch.zeros(n, 3), torch.zeros(n)


def test_play_rally_feeds_the_selected_state():
    env = RecordingTimedEnv(TWO_DECISION_SCRIPT)
    out = sr.play_rally(env, _ZeroPPO(), {1: _Fixed(74), 2: _Fixed(74)}, learner=1, device=torch.device("cpu"),
                        defend_mechanism="preparation", max_steps=100, rng=np.random.default_rng(0),
                        init_choice=((0, 0), (0, 0)), state_mode="no_choices")
    assert all(s.shape == (74,) for s in out["states"])


def test_checkpoint_round_trip_keeps_the_state_mode(tmp_path):
    path = str(tmp_path / "sp.pth")
    sr.save_per_player({1: MetaPolicy(state_dim=74), 2: MetaPolicy(state_dim=74)}, path, {"state_mode": "no_choices"})
    policies, meta = sr.load_per_player(path)
    assert meta["state_mode"] == "no_choices"
    assert policies[1](torch.zeros(1, 74))[0].shape == (1, 5)


def test_old_checkpoints_default_to_the_joint_state(tmp_path):
    path = str(tmp_path / "old.pth")
    sr.save_per_player({1: MetaPolicy(), 2: MetaPolicy()}, path)
    policies, meta = sr.load_per_player(path)
    assert meta.get("state_mode", "joint") == "joint"
    assert policies[2](torch.zeros(1, 90))[1].shape == (1, 3)


def test_evaluator_picker_uses_the_checkpoint_state_mode(tmp_path):
    from nash_skills import eval_matchup_attack_defend as emad
    path = str(tmp_path / "sp.pth")
    sr.save_per_player({1: MetaPolicy(state_dim=74), 2: MetaPolicy(state_dim=74)}, path, {"state_mode": "no_choices"})
    pick = emad.make_picker_joint("selfplay-argmax:" + path, None)
    info = {**_info(), "attacker": 1, "current_joints": (0, 0)}
    assert 0 <= pick(1, np.zeros(116, dtype=np.float32), 0, info) < 15


def test_training_cli_records_the_state_mode(tmp_path, monkeypatch):
    from nash_skills.v2 import selfplay_per_player as spp
    monkeypatch.setattr(spp, "make_env", lambda args: RecordingTimedEnv(TWO_DECISION_SCRIPT))
    monkeypatch.setattr(spp, "load_ppo", lambda path: _ZeroPPO())
    out = str(tmp_path / "sp.pth")
    args = spp.build_parser().parse_args([
        "--iterations", "2", "--rallies-per-iter", "2", "--round-iters", "1", "--output", out,
        "--log", str(tmp_path / "log.csv"), "--seed", "0", "--defend-mechanism", "preparation",
        "--state-mode", "no_choices"])
    spp.train(args)
    assert json.load(open(out + ".json"))["state_mode"] == "no_choices"
