"""Self-play with the v3 state (the ball's spin and the rally status added): actor sizes, rollouts,
checkpoints and the evaluator's picker follow the encoder recorded with the checkpoint."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from nash_skills.v2 import selfplay_rollout as sr
from nash_skills.v2.selfplay_attack_defend import MetaPolicy
from nash_skills.v2.state_encoder_attack_defend import ENCODER_V2, ENCODER_V3, drop_choice_slots, encode_ego
from tests.test_collect_data_attack_defend import TWO_DECISION_SCRIPT
from tests.test_collect_encoder_v3 import STATUS
from tests.test_selfplay_per_player import RecordingTimedEnv, _ZeroPPO
from tests.test_selfplay_state_mode import _Fixed, _info


class StatusEnv(RecordingTimedEnv):
    def reset(self, *a, **k):
        obs, info = super().reset(*a, **k)
        return obs, {**info, **STATUS}

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        return obs, reward, done, truncated, {**info, **STATUS}


def _obs():
    obs = np.random.default_rng(0).normal(size=116).astype(np.float32)
    obs[-2:] = [0.25, 0.75]
    return obs


def test_dimensions_by_encoder():
    assert sr.state_dim_for("joint", ENCODER_V3) == 97 and sr.state_dim_for("no_choices", ENCODER_V3) == 81
    assert sr.state_dim_for("joint") == 90 and sr.state_dim_for("no_choices", ENCODER_V2) == 74


def test_v3_actor_input_drops_only_the_choice_slots():
    info = {**_info(), **STATUS}
    a = sr.encode_for_player(1, _obs(), info, {1: [0, 1], 2: [2, 2]}, "no_choices", ENCODER_V3)
    full = encode_ego(_obs(), info, "right", "center", encoder=ENCODER_V3)
    assert a.shape == (81,)
    np.testing.assert_array_equal(a, drop_choice_slots(full))


def test_play_rally_feeds_the_v3_state():
    out = sr.play_rally(StatusEnv(TWO_DECISION_SCRIPT), _ZeroPPO(), {1: _Fixed(81), 2: _Fixed(81)}, learner=1,
                        device=torch.device("cpu"), defend_mechanism="preparation", max_steps=100,
                        rng=np.random.default_rng(0), init_choice=((0, 0), (0, 0)), state_mode="no_choices",
                        encoder=ENCODER_V3)
    assert out["states"] and all(s.shape == (81,) for s in out["states"])


def test_checkpoint_round_trip_keeps_the_encoder(tmp_path):
    path = str(tmp_path / "sp.pth")
    sr.save_per_player({1: MetaPolicy(state_dim=81), 2: MetaPolicy(state_dim=81)}, path,
                       {"state_mode": "no_choices", "encoder": ENCODER_V3})
    policies, meta = sr.load_per_player(path)
    assert meta["encoder"] == ENCODER_V3
    assert policies[1](torch.zeros(1, 81))[0].shape == (1, 5)


def test_unknown_checkpoint_encoder_is_refused(tmp_path):
    path = str(tmp_path / "sp.pth")
    sr.save_per_player({1: MetaPolicy(), 2: MetaPolicy()}, path, {"encoder": "attack_defend_joint_v9"})
    with pytest.raises(ValueError, match="v9"):
        sr.load_per_player(path)


def test_evaluator_picker_uses_the_checkpoint_encoder(tmp_path):
    from nash_skills import eval_matchup_attack_defend as emad
    path = str(tmp_path / "sp.pth")
    sr.save_per_player({1: MetaPolicy(state_dim=81), 2: MetaPolicy(state_dim=81)}, path,
                       {"state_mode": "no_choices", "encoder": ENCODER_V3})
    pick = emad.make_picker_joint("selfplay-argmax:" + path, None)
    info = {**_info(), **STATUS, "attacker": 1, "current_joints": (0, 0)}
    assert 0 <= pick(1, np.zeros(116, dtype=np.float32), 0, info) < 15


def test_training_cli_records_the_encoder(tmp_path, monkeypatch):
    from nash_skills.v2 import selfplay_per_player as spp
    monkeypatch.setattr(spp, "make_env", lambda args: StatusEnv(TWO_DECISION_SCRIPT))
    monkeypatch.setattr(spp, "load_ppo", lambda path: _ZeroPPO())
    out = str(tmp_path / "sp.pth")
    args = spp.build_parser().parse_args([
        "--iterations", "2", "--rallies-per-iter", "2", "--round-iters", "1", "--output", out,
        "--log", str(tmp_path / "log.csv"), "--seed", "0", "--defend-mechanism", "preparation",
        "--state-mode", "no_choices", "--encoder", ENCODER_V3])
    spp.train(args)
    meta = json.load(open(out + ".json"))
    assert meta["encoder"] == ENCODER_V3 and meta["state_mode"] == "no_choices"
