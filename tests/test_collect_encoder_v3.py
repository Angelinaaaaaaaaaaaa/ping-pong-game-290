"""collect_data_attack_defend --encoder: rallies encoded and tagged with the chosen state encoder."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills.v2 import collect_data_attack_defend as cad
from nash_skills.v2.state_encoder_attack_defend import ENCODER_V2, ENCODER_V3
from tests.test_collect_data_attack_defend import FakePPO, TWO_DECISION_SCRIPT
from tests.test_collect_decision_timing import TimedFakeEnv

STATUS = {"ball_angvel": np.array([100.0, -20.0, 5.0]),
          "rally_status": {"last_hitter": "opp", "bounces_since_hit": 0, "receiver_had_chance": False}}


class StatusFakeEnv(TimedFakeEnv):
    """TimedFakeEnv whose info also carries what SkillEnv adds for the v3 encoder."""

    def reset(self):
        obs, info = super().reset()
        return obs, {**info, **STATUS}

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        return obs, reward, done, truncated, {**info, **STATUS}


def _collect(monkeypatch, tmp_path, **kw):
    env = StatusFakeEnv(TWO_DECISION_SCRIPT)
    monkeypatch.setattr(cad, "SkillEnv", lambda **_kw: env)
    monkeypatch.setattr(cad, "PPO", FakePPO)
    return cad.collect(target_decisions=1, output_path=str(tmp_path / "x.pkl"), ppo_path="fake", max_rallies=1,
                       progress_every=0, quiet=True, decision_timing="contact", **kw)


def test_v3_rallies_carry_spin_and_status(monkeypatch, tmp_path):
    rally = _collect(monkeypatch, tmp_path, encoder=ENCODER_V3)[0]
    assert rally["encoder"] == ENCODER_V3
    states = np.array(rally["states"])
    assert states.shape[1] == 97
    np.testing.assert_allclose(states[:, 90], 1.0)  # 100 rad/s x SPIN_SCALE
    assert states[:, 93:95].tolist() == [[0.0, 1.0]] * len(states)


def test_default_encoder_is_unchanged(monkeypatch, tmp_path):
    rally = _collect(monkeypatch, tmp_path)[0]
    assert rally["encoder"] == ENCODER_V2 and np.array(rally["states"]).shape[1] == 90


def test_unknown_encoder_raises(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        _collect(monkeypatch, tmp_path, encoder="attack_defend_joint_v9")
