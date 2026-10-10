"""Role-based, attacker-relative V3 state encoding.

The encoder keeps the V2 state width (76 values) while changing the meaning
of the slots so that they are independent of physical P1/P2 identity:

    first role  = current attacker
    second role = current defender

All position-like values are represented in the current attacker's canonical
frame.  The canonical frame is P1's local/world frame.  P2 conversion uses
the same 180-degree table reflection already used by V3 semantic/world and
gantry transforms.

The fixed-width layout is:

    0:9    attacker qpos (gantry in canonical frame, arm joints local)
    9:18   attacker qvel (gantry in canonical frame, arm velocities local)
    18:27  defender qpos
    27:36  defender qvel
    36:39  ball position
    39:42  ball velocity
    42:49  attacker target pose (position + quaternion)
    49:56  defender target pose
    56:59  attacker target position error
    59:62  defender target position error
    62:68  attacker gantry history (three most recent x/y frames)
    68:74  defender gantry history (three most recent x/y frames)
    74     attacker skill
    75     defender skill

The old 4-frame full pose history and quaternion tracking errors cannot all
fit alongside both role robot blocks at 76 dimensions.  The retained history
is the spatial gantry history; target poses retain their full position and
orientation information.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from nash_skills.skills import N_SKILLS, skill_index
from v3.constants import GantryTarget, gantry_to_world_position
from v3.racket_target_planner import mirror_position_to_world, mirror_velocity_to_local


STATE_DIM = 76
HISTORY = 4
HISTORY_FRAMES_RETAINED = 3
STATE_ENCODER_NAME = "v3_role_symmetric_canonical_v1"

ATTACKER_QPOS = slice(0, 9)
ATTACKER_QVEL = slice(9, 18)
DEFENDER_QPOS = slice(18, 27)
DEFENDER_QVEL = slice(27, 36)
BALL_POS = slice(36, 39)
BALL_VEL = slice(39, 42)
ATTACKER_TARGET = slice(42, 49)
DEFENDER_TARGET = slice(49, 56)
ATTACKER_DIFF_POS = slice(56, 59)
DEFENDER_DIFF_POS = slice(59, 62)
ATTACKER_HISTORY = slice(62, 68)
DEFENDER_HISTORY = slice(68, 74)
ATTACKER_SKILL = 74
DEFENDER_SKILL = 75

_REFLECTION = np.diag([-1.0, -1.0, 1.0])


def normalized_skill(skill: str | int | float) -> float:
    if isinstance(skill, str):
        index = skill_index(skill)
    else:
        index = skill
    return float(index) / float(N_SKILLS - 1)


def _validate_inputs(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    raw = np.asarray(obs, dtype=np.float32).reshape(-1)
    if raw.shape[0] != 116:
        raise ValueError(f"Expected 116-dim raw observation, got {raw.shape[0]}")
    required = ("diff_pos", "diff_pos_opp", "target", "target_opp")
    missing = [key for key in required if key not in info]
    if missing:
        raise KeyError(f"Missing state-encoder info fields: {missing}")
    return raw


def _reflect_position(position: np.ndarray) -> np.ndarray:
    return np.asarray(mirror_position_to_world(np.asarray(position, dtype=float)), dtype=float)


def _reflect_vector(vector: np.ndarray) -> np.ndarray:
    # Reuse the established P2 x/y sign transform.  z is unchanged.
    return np.asarray(mirror_velocity_to_local(np.asarray(vector, dtype=float)), dtype=float)


def _local_position_to_world(player: int, position: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=float).copy()
    if player == 1:
        return position
    if player == 2:
        return _reflect_position(position)
    raise ValueError("player must be 1 or 2")


def _world_position_to_attacker(position: np.ndarray, attacker: int) -> np.ndarray:
    if attacker == 1:
        return np.asarray(position, dtype=float).copy()
    if attacker == 2:
        return _reflect_position(position)
    raise ValueError("attacker must be 1 or 2")


def _local_vector_to_attacker(vector: np.ndarray, player: int, attacker: int) -> np.ndarray:
    value = np.asarray(vector, dtype=float).copy()
    if player == 2:
        value = _reflect_vector(value)
    if attacker == 2:
        value = _reflect_vector(value)
    return value


def _local_pose_to_attacker(pose: np.ndarray, player: int, attacker: int) -> np.ndarray:
    """Convert a player's local target pose into the attacker frame."""
    value = np.asarray(pose, dtype=float).reshape(7).copy()
    if player == 2:
        value = _reflect_pose(value)
    if attacker == 2:
        value = _reflect_pose(value)
    return value


def _reflect_pose(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=float).reshape(7).copy()
    value[:3] = _reflect_position(value[:3])
    value[3:] = R.from_matrix(_REFLECTION @ R.from_quat(value[3:]).as_matrix()).as_quat()
    return value


def _canonical_qpos(local_qpos: np.ndarray, player: int, attacker: int) -> np.ndarray:
    value = np.asarray(local_qpos, dtype=float).reshape(9).copy()
    world_gantry = gantry_to_world_position(
        player,
        GantryTarget(float(value[0]), float(value[1])),
    )
    canonical_gantry = _world_position_to_attacker(
        np.array([world_gantry.x, world_gantry.y, 0.0], dtype=float),
        attacker,
    )
    value[0:2] = canonical_gantry[:2]
    # Arm joint coordinates are already local to the robot chain.  The role
    # swap changes ownership, not the joint-coordinate convention.
    return value


def _canonical_qvel(local_qvel: np.ndarray, player: int, attacker: int) -> np.ndarray:
    value = np.asarray(local_qvel, dtype=float).reshape(9).copy()
    value[0:2] = _local_vector_to_attacker(value[0:2], player, attacker)
    return value


def _canonical_history(raw_history: np.ndarray, player: int, attacker: int) -> np.ndarray:
    history = np.asarray(raw_history, dtype=float).reshape(HISTORY, 9)
    retained = history[-HISTORY_FRAMES_RETAINED:, :2]
    output = np.empty_like(retained)
    for index, local_gantry in enumerate(retained):
        world = gantry_to_world_position(
            player,
            GantryTarget(float(local_gantry[0]), float(local_gantry[1])),
        )
        canonical = _world_position_to_attacker(
            np.array([world.x, world.y, 0.0], dtype=float),
            attacker,
        )
        output[index] = canonical[:2]
    return output.reshape(-1)


def _role_raw_slices(player: int) -> tuple[slice, slice, slice]:
    if player == 1:
        return slice(0, 9), slice(9, 18), slice(42, 78)
    if player == 2:
        return slice(18, 27), slice(27, 36), slice(78, 114)
    raise ValueError("player must be 1 or 2")


def _role_info(info: dict[str, Any], player: int) -> tuple[np.ndarray, np.ndarray]:
    if player == 1:
        return (
            np.asarray(info["target"], dtype=float)[:7],
            np.asarray(info["diff_pos"], dtype=float)[:3],
        )
    if player == 2:
        return (
            np.asarray(info["target_opp"], dtype=float)[:7],
            np.asarray(info["diff_pos_opp"], dtype=float)[:3],
        )
    raise ValueError("player must be 1 or 2")


def encode_canonical_state(
    obs: np.ndarray,
    info: dict[str, Any],
    attacker_id: int,
    attack_skill: str | int | float | None = None,
    defender_skill: str | int | float | None = None,
) -> np.ndarray:
    """Encode a raw environment state using attacker/defender roles."""
    raw = _validate_inputs(obs, info)
    if attacker_id not in (1, 2):
        raise ValueError("attacker_id must be 1 or 2")
    defender_id = 3 - attacker_id
    attacker_qpos_slice, attacker_qvel_slice, attacker_history_slice = _role_raw_slices(attacker_id)
    defender_qpos_slice, defender_qvel_slice, defender_history_slice = _role_raw_slices(defender_id)

    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[ATTACKER_QPOS] = _canonical_qpos(raw[attacker_qpos_slice], attacker_id, attacker_id)
    state[ATTACKER_QVEL] = _canonical_qvel(raw[attacker_qvel_slice], attacker_id, attacker_id)
    state[DEFENDER_QPOS] = _canonical_qpos(raw[defender_qpos_slice], defender_id, attacker_id)
    state[DEFENDER_QVEL] = _canonical_qvel(raw[defender_qvel_slice], defender_id, attacker_id)

    ball_position = _world_position_to_attacker(raw[36:39], attacker_id)
    ball_velocity = _local_vector_to_attacker(raw[39:42], 1, attacker_id)
    state[BALL_POS] = ball_position
    state[BALL_VEL] = ball_velocity

    attacker_target, attacker_diff = _role_info(info, attacker_id)
    defender_target, defender_diff = _role_info(info, defender_id)
    state[ATTACKER_TARGET] = _local_pose_to_attacker(attacker_target, attacker_id, attacker_id)
    state[DEFENDER_TARGET] = _local_pose_to_attacker(defender_target, defender_id, attacker_id)
    state[ATTACKER_DIFF_POS] = _local_vector_to_attacker(attacker_diff, attacker_id, attacker_id)
    state[DEFENDER_DIFF_POS] = _local_vector_to_attacker(defender_diff, defender_id, attacker_id)

    state[ATTACKER_HISTORY] = _canonical_history(raw[attacker_history_slice], attacker_id, attacker_id)
    state[DEFENDER_HISTORY] = _canonical_history(raw[defender_history_slice], defender_id, attacker_id)

    physical_skills = {1: float(raw[-2]), 2: float(raw[-1])}
    state[ATTACKER_SKILL] = normalized_skill(attack_skill) if attack_skill is not None else physical_skills[attacker_id]
    state[DEFENDER_SKILL] = normalized_skill(defender_skill) if defender_skill is not None else physical_skills[defender_id]
    return state


def state_for_attacker(
    obs: np.ndarray,
    info: dict[str, Any],
    attacker_id: int,
    attack_skill: str | int | float | None = None,
) -> np.ndarray:
    """Compatibility wrapper used by V3 collection and policy evaluation."""
    return encode_canonical_state(obs, info, attacker_id, attack_skill=attack_skill)


STATE_FIELD_NAMES = [
    *[f"attacker_qpos_{index}" for index in range(9)],
    *[f"attacker_qvel_{index}" for index in range(9)],
    *[f"defender_qpos_{index}" for index in range(9)],
    *[f"defender_qvel_{index}" for index in range(9)],
    "ball_x", "ball_y", "ball_z",
    "ball_vx", "ball_vy", "ball_vz",
    *[f"attacker_target_{index}" for index in range(7)],
    *[f"defender_target_{index}" for index in range(7)],
    "attacker_diff_x", "attacker_diff_y", "attacker_diff_z",
    "defender_diff_x", "defender_diff_y", "defender_diff_z",
    "attacker_history_gantry_0_x", "attacker_history_gantry_0_y",
    "attacker_history_gantry_1_x", "attacker_history_gantry_1_y",
    "attacker_history_gantry_2_x", "attacker_history_gantry_2_y",
    "defender_history_gantry_0_x", "defender_history_gantry_0_y",
    "defender_history_gantry_1_x", "defender_history_gantry_1_y",
    "defender_history_gantry_2_x", "defender_history_gantry_2_y",
    "attacker_skill", "defender_skill",
]

if len(STATE_FIELD_NAMES) != STATE_DIM:
    raise AssertionError(f"State field names have {len(STATE_FIELD_NAMES)} entries, expected {STATE_DIM}")


__all__ = [
    "ATTACKER_SKILL",
    "DEFENDER_SKILL",
    "STATE_DIM",
    "STATE_ENCODER_NAME",
    "STATE_FIELD_NAMES",
    "encode_canonical_state",
    "normalized_skill",
    "state_for_attacker",
]
