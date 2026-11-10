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

Version attack_defend_joint_v3 (97 dims) appends the rest of the paper's x_t (Sec. 3.2):
    [90:93] ball angular velocity omega_t (world frame, rad/s x SPIN_SCALE)
    [93:95] rally status ell_t: last-hitter one-hot (player 1, player 2; zeros before the first contact)
    [95]    table bounces since that contact   [96] receiver has had a legal chance to return
read from the info keys "ball_angvel" and "rally_status" that nash_skills.env_wrapper.SkillEnv adds.
Everything in [0:90] is identical to v2.
"""
import numpy as np

from nash_skills.defend_positions import N_DEFEND_POSITIONS, defend_position_index
from nash_skills.skills import N_SKILLS

__all__ = ["encode_ego", "encode_opp", "defend_one_hot", "STATE_DIM_ATTACK_DEFEND", "STATE_ENCODER_NAME",
           "ENCODER_V2", "ENCODER_V3", "ENCODER_DIMS", "SPIN_SCALE", "state_dim", "drop_choice_slots"]

ENCODER_V2 = "attack_defend_joint_v2"
ENCODER_V3 = "attack_defend_joint_v3"
ENCODER_DIMS = {ENCODER_V2: 90, ENCODER_V3: 97}
STATE_DIM_ATTACK_DEFEND = ENCODER_DIMS[ENCODER_V2]
STATE_ENCODER_NAME = ENCODER_V2
SPIN_SCALE = 0.01  # the ball spins at ~100 rad/s; keep the inputs O(1) like the other slots
_CHOICE_SLOTS = slice(74, 90)
_SPIN, _LAST_HITTER, _BOUNCES_SINCE_HIT, _RECEIVER_HAD_CHANCE = slice(90, 93), slice(93, 95), 95, 96
_HITTER_ONE_HOT = {"ego": (1.0, 0.0), "opp": (0.0, 1.0)}


def state_dim(encoder: str) -> int:
    if encoder not in ENCODER_DIMS:
        raise ValueError(f"Unknown state encoder {encoder!r}. Choose from: {sorted(ENCODER_DIMS)}")
    return ENCODER_DIMS[encoder]


def drop_choice_slots(state: np.ndarray) -> np.ndarray:
    """The state without both players' current choices [74:90] (self-play's pi(x, role) input)."""
    return np.concatenate([state[:_CHOICE_SLOTS.start], state[_CHOICE_SLOTS.stop:]])

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


def _fill_spin_and_status(out, info):
    missing = [k for k in ("ball_angvel", "rally_status") if k not in info]
    if missing:
        raise KeyError(f"encoder {ENCODER_V3} needs info keys {missing}: step the env through "
                       f"nash_skills.env_wrapper.SkillEnv, which adds them")
    status = info["rally_status"]
    out[_SPIN] = np.asarray(info["ball_angvel"], dtype=np.float32)[:3] * SPIN_SCALE
    out[_LAST_HITTER] = _HITTER_ONE_HOT.get(status["last_hitter"], (0.0, 0.0))
    out[_BOUNCES_SINCE_HIT] = float(status["bounces_since_hit"])
    out[_RECEIVER_HAD_CHANCE] = float(status["receiver_had_chance"])


def _joint_state(obs, info, p1_defend, p2_defend, acting_player, encoder=ENCODER_V2):
    out = np.zeros(state_dim(encoder), dtype=np.float32)
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
    if encoder == ENCODER_V3:
        _fill_spin_and_status(out, info)
    return out


def encode_ego(obs: np.ndarray, info: dict, ego_defend: str, opp_defend: str,
               encoder: str = ENCODER_V2) -> np.ndarray:
    """Joint state for player 1's decision (also the view Phi and Q are trained on)."""
    return _joint_state(obs, info, ego_defend, opp_defend, acting_player=1, encoder=encoder)


def encode_opp(obs: np.ndarray, info: dict, ego_defend: str, opp_defend: str,
               encoder: str = ENCODER_V2) -> np.ndarray:
    """Joint state for player 2's decision: same content, acting-player slot set to P2.
    ego_defend / opp_defend are player 1's / player 2's preparation, as in encode_ego."""
    return _joint_state(obs, info, ego_defend, opp_defend, acting_player=2, encoder=encoder)
