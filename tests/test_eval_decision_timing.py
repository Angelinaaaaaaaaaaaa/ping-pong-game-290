"""run_matchup_joint with decision_timing='contact', role info for pickers, decision timing
stats, the per-player self-play picker and the ibr-sim strategy."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills import eval_matchup_attack_defend as emad
from nash_skills.v2.joint_action import joint_action_from_index, joint_action_index
from nash_skills.v2.selfplay_attack_defend import MetaPolicy
from nash_skills.v2.selfplay_rollout import save_per_player
from tests.test_eval_matchup_attack_defend_run import FakeEnv, FakeModel, TWO_DECISION_SCRIPT


class _Data:
    def __init__(self):
        self.qvel = np.zeros(24)


class _Inner:
    def __init__(self):
        self.data = _Data()


class TimedEnv(FakeEnv):
    def __init__(self, script):
        super().__init__(script)
        self._env = _Inner()

    def set_defend_preparation_positions(self, p1, p2, calibrated=False):
        self.defend_target1, self.defend_target2 = p1, p2

    def step(self, action):
        self._env.data.qvel[-6] = self._script[self._step_idx][1]
        return super().step(action)


def _run(monkeypatch, s1, s2, **kw):
    env = TimedEnv(TWO_DECISION_SCRIPT)
    monkeypatch.setattr(emad, "SkillEnv", lambda **_kw: env)
    result = emad.run_matchup_joint(s1, s2, FakeModel(), model_p=None, n_episodes=1, max_steps_per_episode=10,
                                    defend_mechanism="preparation", **kw)
    return env, result


def test_contact_timing_passes_the_attacker_and_the_live_env(monkeypatch):
    seen = []

    def fake_make_picker(_strategy, *_a, **_kw):
        def pick(p, _obs, _other, info=None):
            seen.append((p, info["attacker"], info["env"] is not None))
            return 0
        return pick

    monkeypatch.setattr(emad, "make_picker_joint", fake_make_picker)
    _run(monkeypatch, 1, 2, decision_timing="contact")
    # ball heads to P2 (P2 attacks), P2 returns (vel flips at step 4: P1 attacks)
    assert seen == [(1, 2, True), (2, 2, True), (1, 1, True), (2, 1, True)]


def test_decision_seconds_are_recorded_per_player(monkeypatch):
    _env, result = _run(monkeypatch, ("left", "center"), "random", decision_timing="contact")
    assert len(result.decision_seconds[1]) == 2 and len(result.decision_seconds[2]) == 2


def test_unknown_timing_raises(monkeypatch):
    with pytest.raises(ValueError):
        _run(monkeypatch, "random", "random", decision_timing="bounce")


class TestPerPlayerSelfplayPicker:
    def _picker(self, tmp_path, argmax=True):
        path = str(tmp_path / "sp.pth")
        save_per_player({1: MetaPolicy(), 2: MetaPolicy()}, path)
        return emad.make_picker_joint(("selfplay-argmax:" if argmax else "selfplay:") + path, None)

    def _info(self, attacker, current):
        info = {"diff_pos": np.zeros(3), "diff_quat": np.zeros(4), "target": np.zeros(7),
                "diff_pos_opp": np.zeros(3), "diff_quat_opp": np.zeros(4), "target_opp": np.zeros(7)}
        return {**info, "attacker": attacker, "current_joints": current}

    def test_attacker_changes_only_its_attack_target(self, tmp_path):
        pick = self._picker(tmp_path)
        current = (joint_action_index(1, 2), joint_action_index(4, 0))
        new = pick(1, np.zeros(116, dtype=np.float32), current[1], self._info(1, current))
        assert joint_action_from_index(new)[1] == 2  # preparation half untouched

    def test_defender_changes_only_its_preparation(self, tmp_path):
        pick = self._picker(tmp_path, argmax=False)
        current = (joint_action_index(1, 2), joint_action_index(4, 0))
        for _ in range(10):
            new = pick(2, np.zeros(116, dtype=np.float32), current[0], self._info(1, current))
            assert joint_action_from_index(new)[0] == 4  # attack half untouched


def test_ibr_sim_needs_the_low_level_controller():
    with pytest.raises(ValueError, match="ppo"):
        emad.make_picker_joint("ibr-sim", None)
