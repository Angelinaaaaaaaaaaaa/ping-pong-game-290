"""Tests for the paper-faithful preparation mechanism (Sec. 3.2: the player who
just hit prepares during the opponent's return).

The SkillEnv wiring tests swap the MuJoCo env for a small fake that records
which target-pose calls the wrapper makes, so no simulation is needed.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills.defend_preparation import PreparationRoleTracker
from nash_skills.defend_target_pose import defend_target_pose
import nash_skills.env_wrapper as env_wrapper


class TestPreparationRoleTracker:
    def test_starts_unlatched(self):
        assert PreparationRoleTracker().preparing_player is None

    def test_ball_moving_toward_player2_means_player1_prepares(self):
        tracker = PreparationRoleTracker()
        assert tracker.update(4.0) == 1

    def test_ball_moving_toward_player1_means_player2_prepares(self):
        tracker = PreparationRoleTracker()
        assert tracker.update(-4.0) == 2

    def test_small_velocity_before_first_latch_stays_unlatched(self):
        tracker = PreparationRoleTracker()
        assert tracker.update(0.3) is None
        assert tracker.update(-0.3) is None

    def test_dead_band_holds_the_previous_role_through_a_contact_transient(self):
        tracker = PreparationRoleTracker()
        tracker.update(4.0)
        assert tracker.update(-0.08) == 1
        assert tracker.update(0.4) == 1

    def test_flips_once_the_new_direction_is_clear(self):
        tracker = PreparationRoleTracker()
        tracker.update(4.0)
        assert tracker.update(-0.94) == 2

    def test_threshold_is_strict(self):
        tracker = PreparationRoleTracker(vel_threshold=0.5)
        assert tracker.update(0.5) is None

    def test_reset_clears_the_latch(self):
        tracker = PreparationRoleTracker()
        tracker.update(4.0)
        tracker.reset()
        assert tracker.preparing_player is None


class _Body:
    def __init__(self, xpos):
        self.xpos = np.asarray(xpos, dtype=float)


class _Data:
    def __init__(self):
        self.ball_pos = np.array([1.0, 0.0, 1.0])
        self.qvel = np.zeros(24)

    def body(self, name):
        return _Body(self.ball_pos if name == "ball" else np.zeros(3))


class FakeKukaTennisEnv:
    """Calls the pose-update hooks from step() the way KukaTennisEnv does."""

    def __init__(self, **_kwargs):
        self.data = _Data()
        self.side_target = 1
        self.side_target_opp = -1
        self.curr_target = np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0])
        self.curr_target_opp = self.curr_target.copy()
        self.calls = []

    def update_target_racket_pose(self, **kwargs):
        self.calls.append(("update", kwargs))

    def update_target_racket_pose_opp(self, **kwargs):
        self.calls.append(("update_opp", kwargs))

    def set_target_pose(self, pose):
        self.curr_target = np.asarray(pose)
        self.calls.append(("set", np.asarray(pose)))

    def set_target_pose_opp(self, pose):
        self.curr_target_opp = np.asarray(pose)
        self.calls.append(("set_opp", np.asarray(pose)))

    def step(self, _action):
        self.update_target_racket_pose(y_target=self.side_target * 0.38)
        self.update_target_racket_pose_opp(y_target=self.side_target_opp * 0.38)
        return np.zeros(116), 0.0, False, False, {}

    def reset(self, seed=None):
        self.calls.append(("reset", None))
        return np.zeros(116), {}

    def close(self):
        pass


@pytest.fixture
def skill_env(monkeypatch):
    monkeypatch.setattr(env_wrapper, "KukaTennisEnv", FakeKukaTennisEnv)
    return env_wrapper.SkillEnv()


def _step_with_ball_vel_x(env, vel_x):
    env._env.data.qvel[-6] = vel_x
    env._env.calls.clear()
    env.step(np.zeros(18))
    return env._env.calls


def _kinds(calls):
    return [kind for kind, _ in calls]


def _set_pose(calls, kind):
    poses = [payload for k, payload in calls if k == kind]
    assert len(poses) == 1, f"expected exactly one {kind} call, got {calls}"
    return poses[0]


class TestSkillEnvPreparationWiring:
    def test_unknown_position_name_raises(self, skill_env):
        with pytest.raises(ValueError):
            skill_env.set_defend_preparation_positions("far_left", None)

    def test_player1_moves_to_its_preparation_pose_while_the_ball_moves_away(self, skill_env):
        skill_env.set_defend_preparation_positions("left", None)
        calls = _step_with_ball_vel_x(skill_env, 4.0)
        np.testing.assert_allclose(_set_pose(calls, "set"), defend_target_pose("left"))
        assert "update" not in _kinds(calls)
        assert "update_opp" in _kinds(calls)

    def test_receiver_keeps_normal_ball_interception_with_its_attack_aim(self, skill_env):
        skill_env.set_defend_preparation_positions("left", "right")
        calls = _step_with_ball_vel_x(skill_env, -4.0)
        updates = [kw for kind, kw in calls if kind == "update"]
        assert len(updates) == 1
        assert updates[0]["x_target"] == skill_env._x_target1
        assert "set" not in _kinds(calls)

    def test_player2_gets_its_pose_through_the_mirroring_setter(self, skill_env):
        skill_env.set_defend_preparation_positions(None, "right")
        calls = _step_with_ball_vel_x(skill_env, -4.0)
        np.testing.assert_allclose(_set_pose(calls, "set_opp"), defend_target_pose("right"))
        assert "update_opp" not in _kinds(calls)
        assert "update" in _kinds(calls)

    def test_dead_band_keeps_preparing_through_the_contact_transient(self, skill_env):
        skill_env.set_defend_preparation_positions("center", None)
        _step_with_ball_vel_x(skill_env, 4.0)
        calls = _step_with_ball_vel_x(skill_env, -0.2)
        np.testing.assert_allclose(_set_pose(calls, "set"), defend_target_pose("center"))

    def test_player_without_a_position_is_left_alone(self, skill_env):
        skill_env.set_defend_preparation_positions(None, "center")
        calls = _step_with_ball_vel_x(skill_env, 4.0)
        assert _kinds(calls).count("update") == 1
        assert "set" not in _kinds(calls)

    def test_role_tracker_still_runs_so_callers_can_detect_decisions(self, skill_env):
        skill_env.set_defend_preparation_positions("left", "right")
        _step_with_ball_vel_x(skill_env, 4.0)
        assert skill_env._role_tracker.defending_player == 2

    def test_reset_clears_the_preparation_latch(self, skill_env):
        skill_env.set_defend_preparation_positions("left", None)
        _step_with_ball_vel_x(skill_env, 4.0)
        skill_env.reset()
        assert skill_env._prep_tracker.preparing_player is None

    def test_no_mechanism_set_keeps_the_original_updates(self, skill_env):
        calls = _step_with_ball_vel_x(skill_env, 4.0)
        assert _kinds(calls) == ["update", "update_opp"]


class TestSkillEnvResetTargets:
    """KukaTennisEnv.reset() leaves the racket targets where the last rally put
    them; reset_targets=True restores the construction-time targets first."""

    def test_restores_both_home_targets_before_the_inner_reset(self, monkeypatch):
        monkeypatch.setattr(env_wrapper, "KukaTennisEnv", FakeKukaTennisEnv)
        env = env_wrapper.SkillEnv(reset_targets=True)
        home = env._env.curr_target.copy()
        env._env.curr_target = np.full(7, 5.0)
        env._env.curr_target_opp = np.full(7, -5.0)
        env._env.calls.clear()
        env.reset()
        assert [k for k, _ in env._env.calls] == ["set", "set_opp", "reset"]
        np.testing.assert_allclose(env._env.curr_target, home)
        np.testing.assert_allclose(env._env.curr_target_opp, home)

    def test_default_leaves_targets_alone(self, skill_env):
        skill_env._env.calls.clear()
        skill_env.reset()
        assert [k for k, _ in skill_env._env.calls] == ["reset"]
