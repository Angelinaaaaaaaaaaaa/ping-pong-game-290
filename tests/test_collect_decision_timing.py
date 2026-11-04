"""Collector support for decision_timing='contact' (paper Sec. 3.2 exchanges)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills.v2 import collect_data_attack_defend as cad
from tests.test_collect_data_attack_defend import FakeEnv, FakePPO, TWO_DECISION_SCRIPT


class _Data:
    def __init__(self):
        self.qvel = np.zeros(24)


class _Inner:
    def __init__(self):
        self.data = _Data()


class TimedFakeEnv(FakeEnv):
    """FakeEnv that also exposes the ball x-velocity where DecisionClock reads it."""

    def __init__(self, script):
        super().__init__(script)
        self._env = _Inner()

    def set_defend_preparation_positions(self, pos1, pos2, calibrated=False):
        self.defend_target1, self.defend_target2 = pos1, pos2

    def step(self, action):
        self._env.data.qvel[-6] = self._script[self._step_idx][1]
        return super().step(action)


def _collect(monkeypatch, tmp_path, **kw):
    env = TimedFakeEnv(TWO_DECISION_SCRIPT)
    monkeypatch.setattr(cad, "SkillEnv", lambda **_kw: env)
    monkeypatch.setattr(cad, "PPO", FakePPO)
    return cad.collect(target_decisions=1, output_path=str(tmp_path / "x.pkl"), ppo_path="fake",
                       max_rallies=1, progress_every=0, quiet=True, **kw)


def test_contact_timing_decides_at_each_hit_with_the_receiver_attacking(monkeypatch, tmp_path):
    rally = _collect(monkeypatch, tmp_path, decision_timing="contact")[0]
    # ball heads to P2 (P2 attacks), then P2 returns it (P1 attacks)
    assert rally["attacker"] == [2, 1]
    assert len(rally["states"]) == 2
    assert rally["decision_timing"] == "contact"


def test_crossing_timing_is_the_default_and_keeps_its_decisions(monkeypatch, tmp_path):
    rally = _collect(monkeypatch, tmp_path)[0]
    assert rally["decision_timing"] == "crossing"
    assert rally["attacker"] == [2, 1]
    assert len(rally["states"]) == 2


def test_rally_records_the_skill_profile(monkeypatch, tmp_path):
    rally = _collect(monkeypatch, tmp_path, skill_profile="moderate_aggressive")[0]
    assert rally["skill_profile"] == "moderate_aggressive"


def test_unknown_timing_raises(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        _collect(monkeypatch, tmp_path, decision_timing="bounce")
