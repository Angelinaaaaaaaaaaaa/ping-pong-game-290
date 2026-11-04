"""
Design B joint state (90 dims), following the paper's x_t = (ball, z^1, z^2, attacker)
plus both players' current (attack, preparation) choices.

KukaTennisEnv's raw observation (mujoco_env_comp.py:601/738) is
concat(qpos[:18], qvel[:18], ...):

    [0:9] P1 qpos   [9:18] P2 qpos   [18:27] P1 qvel   [27:36] P2 qvel
    [36:39] ball position   [39:42] ball velocity
    [42:78] P1 action history   [78:114] P2 action history   [114:116] skills

The earlier version of this encoder reused the legacy 74-dim block of
nash_skills/v2/state_encoder.py, which reads that observation as
[P1 qpos, P1 qvel, P2 qpos, P2 qvel]: P1's vector got P2's positions in its
"velocity" slots and P2's vector got P1's velocities in its "position" slots
(scripts/check_obs_layout.py shows it on a live env). This version reads the
blocks by their real positions and keeps both robots, as the paper's joint state does.

Layout (identical for both players except the acting-player slot):
    [0:9]   P1 qpos (gantry x, y; 7 arm joints)    [9:18]  P1 qvel
    [18:27] P2 qpos                                 [27:36] P2 qvel
    [36:39] ball position                           [39:42] ball velocity
    [42:49] P1 racket target pose   [49:52] P1 position error   [52:56] P1 orientation error
    [56:63] P2 racket target pose   [63:66] P2 position error   [66:70] P2 orientation error
    [70:72] attacker one-hot: the player the ball is moving toward (paper's iota_t)
    [72:74] acting-player one-hot: whose decision this vector is for
    [74:79] P1 attack one-hot   [79:84] P2 attack one-hot
    [84:87] P1 preparation one-hot   [87:90] P2 preparation one-hot

The choice slots stay at [74:90], where nash_skills.v2.joint_action writes them.
Raw-world coordinates are kept (no mirroring); the low-level action history is
dropped to make room for the second robot.
"""
import numpy as np

from nash_skills.defend_positions import N_DEFEND_POSITIONS, defend_position_index
from nash_skills.skills import N_SKILLS

__all__ = ["encode_ego", "encode_opp", "defend_one_hot", "STATE_DIM_ATTACK_DEFEND", "STATE_ENCODER_NAME"]

STATE_DIM_ATTACK_DEFEND = 90
STATE_ENCODER_NAME = "attack_defend_joint_v2"

_RAW_P1_QPOS, _RAW_P2_QPOS = slice(0, 9), slice(9, 18)
_RAW_P1_QVEL, _RAW_P2_QVEL = slice(18, 27), slice(27, 36)
_RAW_BALL_POS, _RAW_BALL_VEL = slice(36, 39), slice(39, 42)
_ATTACKER, _ACTING = slice(70, 72), slice(72, 74)
_P1_ATTACK = slice(74, 74 + N_SKILLS)
_P2_ATTACK = slice(_P1_ATTACK.stop, _P1_ATTACK.stop + N_SKILLS)
_P1_DEFEND = slice(_P2_ATTACK.stop, _P2_ATTACK.stop + N_DEFEND_POSITIONS)
_P2_DEFEND = slice(_P1_DEFEND.stop, _P1_DEFEND.stop + N_DEFEND_POSITIONS)


def defend_one_hot(position: str) -> np.ndarray:
    """3-dim one-hot vector for a defend-position name, in DEFEND_POSITION_NAMES order."""
    out = np.zeros(N_DEFEND_POSITIONS, dtype=np.float32)
    out[defend_position_index(position)] = 1.0
    return out


def _skill_one_hot_from_scalar(value: float) -> np.ndarray:
    out = np.zeros(N_SKILLS, dtype=np.float32)
    out[int(round(float(value) * (N_SKILLS - 1)))] = 1.0
    return out


def _attacker_one_hot(ball_vel_x: float) -> np.ndarray:
    if ball_vel_x < 0:
        return np.array([1.0, 0.0], dtype=np.float32)
    if ball_vel_x > 0:
        return np.array([0.0, 1.0], dtype=np.float32)
    return np.zeros(2, dtype=np.float32)


def _joint_state(obs, info, p1_defend, p2_defend, acting_player):
    out = np.zeros(STATE_DIM_ATTACK_DEFEND, dtype=np.float32)
    out[0:9], out[9:18] = obs[_RAW_P1_QPOS], obs[_RAW_P1_QVEL]
    out[18:27], out[27:36] = obs[_RAW_P2_QPOS], obs[_RAW_P2_QVEL]
    out[36:39], out[39:42] = obs[_RAW_BALL_POS], obs[_RAW_BALL_VEL]
    out[42:49], out[49:52], out[52:56] = info["target"][:7], info["diff_pos"][:3], info["diff_quat"][:4]
    out[56:63], out[63:66], out[66:70] = info["target_opp"][:7], info["diff_pos_opp"][:3], info["diff_quat_opp"][:4]
    out[_ATTACKER] = _attacker_one_hot(float(obs[_RAW_BALL_VEL][0]))
    out[_ACTING] = [1.0, 0.0] if acting_player == 1 else [0.0, 1.0]
    out[_P1_ATTACK] = _skill_one_hot_from_scalar(obs[-2])
    out[_P2_ATTACK] = _skill_one_hot_from_scalar(obs[-1])
    out[_P1_DEFEND] = defend_one_hot(p1_defend)
    out[_P2_DEFEND] = defend_one_hot(p2_defend)
    return out


def encode_ego(obs: np.ndarray, info: dict, ego_defend: str, opp_defend: str) -> np.ndarray:
    """Joint state for player 1's decision (also the view Phi and Q are trained on)."""
    return _joint_state(obs, info, ego_defend, opp_defend, acting_player=1)


def encode_opp(obs: np.ndarray, info: dict, ego_defend: str, opp_defend: str) -> np.ndarray:
    """Joint state for player 2's decision: same content, acting-player slot set to P2.
    ego_defend / opp_defend are player 1's / player 2's preparation, as in encode_ego."""
    return _joint_state(obs, info, ego_defend, opp_defend, acting_player=2)
