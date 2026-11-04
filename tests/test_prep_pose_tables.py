"""Tests for nash_skills.prep_pose_tables (calibrated preparation poses per skill profile)
and their wiring through the 'preparation_cal' defend mechanism."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills import env_wrapper
from nash_skills.defend_target_pose import _DEFEND_QUAT
from nash_skills.prep_pose_tables import (
    CALIBRATED_PREP_POSES, PREP_POSE_CONTROLLER_MD5, calibrated_prep_pose,
)
from tests.test_defend_preparation import FakeKukaTennisEnv, _set_pose, _step_with_ball_vel_x

CAL_DIR = os.path.join(os.path.dirname(__file__), "..", "analysis", "prep_pose_calibration")


def test_both_profiles_have_all_three_positions():
    for profile in ("aggressive", "moderate_aggressive"):
        assert set(CALIBRATED_PREP_POSES[profile]) == {"left", "center", "right"}


def test_pose_is_table_position_plus_the_fixed_orientation():
    pose = calibrated_prep_pose("right", "moderate_aggressive")
    assert pose.shape == (7,)
    np.testing.assert_allclose(pose[:3], CALIBRATED_PREP_POSES["moderate_aggressive"]["right"])
    np.testing.assert_allclose(pose[3:], _DEFEND_QUAT)


def test_unknown_profile_or_position_raises():
    with pytest.raises(ValueError, match="current"):
        calibrated_prep_pose("left", "current")
    with pytest.raises(ValueError):
        calibrated_prep_pose("far_left", "aggressive")


@pytest.mark.parametrize("profile", ["aggressive", "moderate_aggressive"])
def test_table_matches_the_saved_calibration_run(profile):
    with open(os.path.join(CAL_DIR, f"{profile}.json")) as f:
        cal = json.load(f)
    assert cal["ppo_md5"] == PREP_POSE_CONTROLLER_MD5
    for name in ("left", "center", "right"):
        s = cal["summary"][name]
        np.testing.assert_allclose(CALIBRATED_PREP_POSES[profile][name], (s["x"], s["y"], s["z"]), atol=1e-3)


@pytest.fixture
def cal_env(monkeypatch):
    monkeypatch.setattr(env_wrapper, "KukaTennisEnv", FakeKukaTennisEnv)
    return env_wrapper.SkillEnv(skill_profile="moderate_aggressive")


def test_calibrated_preparation_uses_the_profile_table(cal_env):
    cal_env.set_defend_preparation_positions("left", "right", calibrated=True)
    calls = _step_with_ball_vel_x(cal_env, 4.0)
    np.testing.assert_allclose(_set_pose(calls, "set"), calibrated_prep_pose("left", "moderate_aggressive"))
    calls = _step_with_ball_vel_x(cal_env, -4.0)
    np.testing.assert_allclose(_set_pose(calls, "set_opp"), calibrated_prep_pose("right", "moderate_aggressive"))


def test_calibrated_preparation_refuses_an_uncalibrated_profile(monkeypatch):
    monkeypatch.setattr(env_wrapper, "KukaTennisEnv", FakeKukaTennisEnv)
    env = env_wrapper.SkillEnv(skill_profile="current")
    with pytest.raises(ValueError, match="current"):
        env.set_defend_preparation_positions("left", "right", calibrated=True)
