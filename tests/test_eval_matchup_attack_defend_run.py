"""Tests for nash_skills.eval_matchup_attack_defend.run_matchup_joint.

TDD: written before implementation.

The two-axis analog of eval_matchup.run_matchup -- but that function
can't be reused (hardcoded to env.set_skills only, no defend-position
calls, and a cruder raw ball_x sign-change crossing check). This drives
BOTH env.set_skills AND env.set_defend_reactive_positions via
env._role_tracker transitions, modeled directly on
collect_data_attack_defend.py's already-correct loop (same FakeEnv
pattern as tests/test_collect_data_attack_defend.py: a REAL
DefenderRoleTracker driven by a scripted ball_x/ball_vel_x sequence).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills import eval_matchup_attack_defend as emad
from nash_skills.defend_role import DefenderRoleTracker


def _info():
    return {
        "diff_pos": np.zeros(3), "diff_quat": np.zeros(4), "target": np.zeros(7),
        "diff_pos_opp": np.zeros(3), "diff_quat_opp": np.zeros(4), "target_opp": np.zeros(7),
    }


def _obs(ball_x: float) -> np.ndarray:
    obs = np.zeros(116, dtype=np.float32)
    obs[36] = ball_x
    return obs


class FakeEnv:
    """
    Drives a REAL DefenderRoleTracker from a scripted
    (ball_x, ball_vel_x, done, winner) sequence -- one entry consumed per
    step() call, exactly mirroring tests/test_collect_data_attack_defend.py's
    FakeEnv. Deliberately has NO set_defend_positions method (the
    hard-override mechanism) and NO set_defend_target_positions method
    (the static pose-redirect "Option B" mechanism, also found net-
    harmful) -- only set_defend_reactive_positions (the velocity/
    trajectory-informed mechanism that replaced them both) -- so a
    regression to either older mechanism fails loudly.
    """

    def __init__(self, script):
        self._script = list(script)
        self._step_idx = 0
        self._role_tracker = DefenderRoleTracker()
        self.skill1 = "left"
        self.skill2 = "right"
        self.defend_target1 = None
        self.defend_target2 = None

    def set_skills(self, skill1, skill2):
        self.skill1 = skill1
        self.skill2 = skill2

    def set_defend_reactive_positions(self, pos1, pos2):
        self.defend_target1 = pos1
        self.defend_target2 = pos2

    def reset(self):
        self._step_idx = 0
        self._role_tracker.reset()
        return _obs(0.0), _info()

    def step(self, _action):
        ball_x, ball_vel_x, done, winner = self._script[self._step_idx]
        self._step_idx += 1
        self._role_tracker.update(ball_x, ball_vel_x)
        info = _info()
        if winner is not None:
            info["winner"] = winner
        return _obs(ball_x), 0.0, done, False, info

    def close(self):
        pass


class FakeModel:
    def predict(self, _obs, deterministic=True):
        return np.zeros(9, dtype=np.float32), None


class FakePPO:
    @staticmethod
    def load(_path):
        return FakeModel()


# One decision, ball heads to P2 then the rally ends with ego winning.
EGO_WINS_SCRIPT = [
    (0.1, 4.5, False, None),
    (0.8, 4.5, False, None),
    (1.6, 4.5, False, None),
    (2.9, 4.5, True, "ego"),
]

# Never reaches `done` within the script -- simulates a truncated episode.
NEVER_DONE_SCRIPT = [
    (0.1, 4.5, False, None),
    (0.8, 4.5, False, None),
]


def _patch(monkeypatch, script):
    env = FakeEnv(script)
    monkeypatch.setattr(emad, "SkillEnv", lambda **_kw: env)
    return env, FakeModel()


class TestFixedVsFixedRecordsOutcome:
    def test_ego_win_is_recorded(self, monkeypatch):
        env, ppo = _patch(monkeypatch, EGO_WINS_SCRIPT)
        result = emad.run_matchup_joint(
            ("left", "center"), ("right", "left"), ppo, model_p=None,
            n_episodes=1, max_steps_per_episode=10,
        )
        assert result.episodes == 1
        assert result.ego_wins == 1
        assert result.opp_wins == 0

    def test_uses_fixed_reactive_pose_mechanism_not_older_mechanisms(self, monkeypatch):
        env, ppo = _patch(monkeypatch, EGO_WINS_SCRIPT)
        emad.run_matchup_joint(
            ("left", "center"), ("right", "left"), ppo, model_p=None,
            n_episodes=1, max_steps_per_episode=10,
        )
        # FakeEnv has neither set_defend_positions nor
        # set_defend_target_positions -- if run_matchup_joint had called
        # either instead, this test would already have failed with
        # AttributeError during the call above.
        assert env.defend_target1 is not None
        assert env.defend_target2 is not None


class TestDefendMechanismSwitch:
    def test_preparation_mechanism_routes_to_the_preparation_setter(self, monkeypatch):
        env, ppo = _patch(monkeypatch, EGO_WINS_SCRIPT)
        prep = []
        env.set_defend_preparation_positions = lambda p1, p2: prep.append((p1, p2))
        result = emad.run_matchup_joint(
            ("left", "center"), ("right", "left"), ppo, model_p=None,
            n_episodes=1, max_steps_per_episode=10, defend_mechanism="preparation",
        )
        assert result.episodes == 1
        assert prep[-1] == ("center", "left")
        assert env.defend_target1 is None and env.defend_target2 is None

    def test_unknown_mechanism_raises(self, monkeypatch):
        _env, ppo = _patch(monkeypatch, EGO_WINS_SCRIPT)
        with pytest.raises(ValueError):
            emad.run_matchup_joint(
                ("left", "center"), ("right", "left"), ppo, model_p=None,
                n_episodes=1, max_steps_per_episode=10, defend_mechanism="static_pose",
            )


class TestTruncatedEpisodesCountedSeparately:
    def test_truncated_episode_is_not_a_win_and_is_counted(self, monkeypatch):
        env, ppo = _patch(monkeypatch, NEVER_DONE_SCRIPT * 5)
        result = emad.run_matchup_joint(
            ("left", "center"), ("right", "left"), ppo, model_p=None,
            n_episodes=1, max_steps_per_episode=2, max_attempts=5,
        )
        assert result.episodes == 0
        assert result.truncated_episodes >= 1


class TestRandomStrategyIntegration:
    def test_random_vs_random_runs_without_crashing(self, monkeypatch):
        env, ppo = _patch(monkeypatch, EGO_WINS_SCRIPT)
        result = emad.run_matchup_joint(
            "random", "random", ppo, model_p=None,
            n_episodes=1, max_steps_per_episode=10,
        )
        assert result.episodes == 1


class TestResetTargetsFlag:
    @pytest.mark.parametrize("flag", [True, False])
    def test_reset_targets_is_passed_to_the_env(self, monkeypatch, flag):
        env = FakeEnv(EGO_WINS_SCRIPT)
        captured = {}
        monkeypatch.setattr(emad, "SkillEnv", lambda **kw: (captured.update(kw), env)[1])
        emad.run_matchup_joint(("left", "center"), ("right", "left"), FakeModel(), model_p=None,
                               n_episodes=1, max_steps_per_episode=10, reset_targets=flag)
        assert captured["reset_targets"] is flag


# Two decisions (net crossings), then ego wins.
TWO_DECISION_SCRIPT = [
    (0.1, 4.5, False, None),
    (0.8, 4.5, False, None),
    (1.6, 4.5, False, None),
    (2.5, -4.0, False, None),
    (1.4, -4.0, False, None),
    (0.05, -4.0, True, "ego"),
]


class TestSimultaneousMoves:
    """Paper Sec. 3.2: neither player sees the other's newly sampled action,
    so both pickers must receive the other's PREVIOUS choice."""

    def test_both_pickers_see_only_the_previous_choice(self, monkeypatch):
        env, ppo = _patch(monkeypatch, TWO_DECISION_SCRIPT)
        seen = {1: [], 2: []}
        picks = {1: iter([3, 5]), 2: iter([7, 9])}

        def fake_make_picker(strategy, *_a, **_kw):
            player = strategy

            def pick(p, _obs, other, _info=None):
                seen[p].append(other)
                return next(picks[p])
            return pick

        monkeypatch.setattr(emad, "make_picker_joint", fake_make_picker)
        emad.run_matchup_joint(1, 2, ppo, model_p=None, n_episodes=1, max_steps_per_episode=10)
        assert seen[1] == [0, 7]
        assert seen[2] == [0, 3]


class TestPickersSeeTheActiveChoices:
    """Each new rally restarts both players at joint 0; pickers must be told the
    choices actually in force, not rely on what they picked in the last rally."""

    def test_info_carries_the_current_joints_at_every_decision(self, monkeypatch):
        env, ppo = _patch(monkeypatch, TWO_DECISION_SCRIPT * 2)
        seen = []
        picks = iter([3, 7, 5, 9, 4, 8, 6, 2])

        def fake_make_picker(_strategy, *_a, **_kw):
            def pick(p, _obs, _other, info=None):
                seen.append((p, info["current_joints"]))
                return next(picks)
            return pick

        monkeypatch.setattr(emad, "make_picker_joint", fake_make_picker)
        emad.run_matchup_joint(1, 2, ppo, model_p=None, n_episodes=2, max_steps_per_episode=10)
        assert seen[0] == (1, (0, 0)) and seen[1] == (2, (0, 0))
        assert seen[2] == (1, (3, 7)) and seen[3] == (2, (3, 7))
        assert seen[4] == (1, (0, 0)), "second rally must start from the reset choices"
