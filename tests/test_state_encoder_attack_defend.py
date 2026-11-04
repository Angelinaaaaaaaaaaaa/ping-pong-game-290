"""Tests for the Design B joint-state encoder (nash_skills.v2.state_encoder_attack_defend).

The raw KukaTennisEnv observation is [P1 qpos, P2 qpos, P1 qvel, P2 qvel, ball, ...]
(mujoco_env_comp.py builds it as concat(qpos[:18], qvel[:18], ...)). The synthetic obs
below gives every raw block distinct values so each encoded slot can be traced to its
source; TestRealEnv checks the same thing against MuJoCo's own qpos/qvel.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import contextlib
import io

import numpy as np
import pytest

from nash_skills.v2.state_encoder_attack_defend import (
    STATE_DIM_ATTACK_DEFEND, STATE_ENCODER_NAME, defend_one_hot, encode_ego, encode_opp,
)
from nash_skills.defend_positions import DEFEND_POSITION_NAMES, N_DEFEND_POSITIONS
from nash_skills.skills import N_SKILLS, skill_index

P1_QPOS = np.arange(1, 10, dtype=np.float32)
P2_QPOS = np.arange(11, 20, dtype=np.float32)
P1_QVEL = np.arange(21, 30, dtype=np.float32)
P2_QVEL = np.arange(31, 40, dtype=np.float32)
BALL_POS = np.array([0.5, 0.1, 1.0], dtype=np.float32)


def _obs(ball_vel_x=-3.0, skill1="left_short", skill2="right"):
    obs = np.zeros(116, dtype=np.float32)
    obs[0:9], obs[9:18], obs[18:27], obs[27:36] = P1_QPOS, P2_QPOS, P1_QVEL, P2_QVEL
    obs[36:39] = BALL_POS
    obs[39:42] = [ball_vel_x, 0.2, 0.5]
    obs[42:114] = 99.0  # low-level action history: not part of the joint state
    obs[-2] = skill_index(skill1) / (N_SKILLS - 1)
    obs[-1] = skill_index(skill2) / (N_SKILLS - 1)
    return obs


def _info():
    return {
        "target": np.arange(41, 48, dtype=float), "diff_pos": np.arange(51, 54, dtype=float),
        "diff_quat": np.arange(61, 65, dtype=float), "target_opp": np.arange(71, 78, dtype=float),
        "diff_pos_opp": np.arange(81, 84, dtype=float), "diff_quat_opp": np.arange(91, 95, dtype=float),
    }


@pytest.fixture
def ego():
    return encode_ego(_obs(), _info(), "left", "center")


class TestLayout:
    def test_dimension_is_unchanged_so_action_slots_stay_at_74_90(self, ego):
        assert STATE_DIM_ATTACK_DEFEND == 90
        assert ego.shape == (90,)

    def test_both_robots_positions_and_velocities_come_from_the_right_raw_blocks(self, ego):
        np.testing.assert_array_equal(ego[0:9], P1_QPOS)
        np.testing.assert_array_equal(ego[9:18], P1_QVEL)
        np.testing.assert_array_equal(ego[18:27], P2_QPOS)
        np.testing.assert_array_equal(ego[27:36], P2_QVEL)

    def test_ball(self, ego):
        np.testing.assert_allclose(ego[36:39], BALL_POS)
        np.testing.assert_allclose(ego[39:42], [-3.0, 0.2, 0.5])

    def test_both_racket_targets_and_tracking_errors(self, ego):
        info = _info()
        np.testing.assert_allclose(ego[42:49], info["target"])
        np.testing.assert_allclose(ego[49:52], info["diff_pos"])
        np.testing.assert_allclose(ego[52:56], info["diff_quat"])
        np.testing.assert_allclose(ego[56:63], info["target_opp"])
        np.testing.assert_allclose(ego[63:66], info["diff_pos_opp"])
        np.testing.assert_allclose(ego[66:70], info["diff_quat_opp"])

    def test_action_history_is_not_encoded(self, ego):
        assert not np.any(ego == 99.0)

    @pytest.mark.parametrize("vx, expected", [(-3.0, [1, 0]), (3.0, [0, 1]), (0.0, [0, 0])])
    def test_attacker_is_the_player_the_ball_moves_toward(self, vx, expected):
        state = encode_ego(_obs(ball_vel_x=vx), _info(), "left", "center")
        np.testing.assert_array_equal(state[70:72], expected)

    def test_acting_player_slot(self):
        np.testing.assert_array_equal(encode_ego(_obs(), _info(), "left", "center")[72:74], [1, 0])
        np.testing.assert_array_equal(encode_opp(_obs(), _info(), "left", "center")[72:74], [0, 1])

    def test_choices_are_player_ordered(self, ego):
        assert ego[74:79].argmax() == skill_index("left_short")
        assert ego[79:84].argmax() == skill_index("right")
        np.testing.assert_array_equal(ego[84:87], defend_one_hot("left"))
        np.testing.assert_array_equal(ego[87:90], defend_one_hot("center"))

    def test_both_players_see_the_same_joint_state_except_who_is_acting(self):
        e = encode_ego(_obs(), _info(), "right", "left")
        o = encode_opp(_obs(), _info(), "right", "left")
        mask = np.ones(90, dtype=bool)
        mask[72:74] = False
        np.testing.assert_array_equal(e[mask], o[mask])

    def test_encoder_has_a_new_version_tag(self):
        assert STATE_ENCODER_NAME != "attack_defend"


class TestDefendOneHot:
    def test_is_single_hot_and_distinct(self):
        hots = [defend_one_hot(p) for p in DEFEND_POSITION_NAMES]
        for h in hots:
            assert h.shape == (N_DEFEND_POSITIONS,) and h.sum() == 1.0
        assert len({h.argmax() for h in hots}) == N_DEFEND_POSITIONS

    def test_unknown_position_raises(self):
        with pytest.raises(Exception):
            defend_one_hot("far_left")


class TestRealEnv:
    """Regression guard for the original bug: compare against MuJoCo's own state."""

    def test_slots_match_mujoco_qpos_and_qvel(self):
        from nash_skills.env_wrapper import SkillEnv
        env = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile="aggressive")
        np.random.seed(0)
        obs, info = env.reset()
        rng = np.random.default_rng(0)
        for _ in range(30):
            with contextlib.redirect_stdout(io.StringIO()):
                obs, _r, _d, _t, info = env.step(rng.uniform(-1, 1, 18).astype(np.float32))
        data = env._env.data
        state = encode_ego(obs, info, "left", "right")
        np.testing.assert_allclose(state[0:9], data.qpos[0:9], atol=1e-5)
        np.testing.assert_allclose(state[9:18], data.qvel[0:9], atol=1e-5)
        np.testing.assert_allclose(state[18:27], data.qpos[9:18], atol=1e-5)
        np.testing.assert_allclose(state[27:36], data.qvel[9:18], atol=1e-5)
        env.close()
