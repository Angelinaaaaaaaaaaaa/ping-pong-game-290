"""Shared V3 state/action model-input schema.

The V3 state is deliberately action-free.  Candidate actions are represented
only by the appended one-hot blocks, so every Q/Phi counterfactual uses the
same physical state prefix.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from nash_skills.skills import SKILL_NAMES
from v3.constants import DEFENSE_POSITIONS
from v3.state_encoder import STATE_DIM, STATE_ENCODER_NAME


STATE_SCHEMA_VERSION = STATE_ENCODER_NAME
ACTION_SCHEMA_VERSION = "v3_joint_action_onehot_v1"
MODEL_INPUT_SCHEMA_VERSION = "v3_state74_action_onehot_v1"
ACTION_DIM = len(SKILL_NAMES) + len(DEFENSE_POSITIONS)
MODEL_INPUT_DIM = STATE_DIM + ACTION_DIM

if tuple(SKILL_NAMES) != ("left", "left_short", "center_safe", "right_short", "right"):
    raise AssertionError(f"Unexpected V3 skill order: {SKILL_NAMES!r}")
if tuple(DEFENSE_POSITIONS) != ("left", "center", "right"):
    raise AssertionError(f"Unexpected V3 defense order: {DEFENSE_POSITIONS!r}")
if STATE_DIM != 74 or ACTION_DIM != 8 or MODEL_INPUT_DIM != 82:
    raise AssertionError("V3 state/action schema dimensions are inconsistent")


def _index(value: str | int, names: tuple[str, ...] | list[str]) -> int:
    index = int(value) if isinstance(value, (int, np.integer)) else names.index(value)
    if not 0 <= index < len(names):
        raise ValueError(f"Action index {index} is outside [0, {len(names)})")
    return index


def build_joint_action_input(
    state: np.ndarray,
    attack_skill: str | int,
    defense_position: str | int,
) -> np.ndarray:
    """Build the canonical 82D input without mutating ``state``."""
    physical_state = np.asarray(state, dtype=np.float32).reshape(-1)
    if physical_state.shape != (STATE_DIM,):
        raise ValueError(f"Expected {STATE_DIM}-dim state, got {physical_state.shape}")

    attack_index = _index(attack_skill, list(SKILL_NAMES))
    defense_index = _index(defense_position, list(DEFENSE_POSITIONS))
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[attack_index] = 1.0
    action[len(SKILL_NAMES) + defense_index] = 1.0
    return np.concatenate((physical_state.copy(), action)).astype(np.float32, copy=False)


def build_joint_action_batch(
    states: np.ndarray,
    attack_skill: str | int,
    defense_position: str | int,
) -> np.ndarray:
    """Vectorized equivalent of :func:`build_joint_action_input`."""
    values = np.asarray(states, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != STATE_DIM:
        raise ValueError(f"Expected states with shape (N, {STATE_DIM}), got {values.shape}")
    if len(values) == 0:
        return np.empty((0, MODEL_INPUT_DIM), dtype=np.float32)
    action = build_joint_action_input(values[0], attack_skill, defense_position)[STATE_DIM:]
    return np.concatenate((values.copy(), np.repeat(action[None, :], len(values), axis=0)), axis=1)


def iter_action_pairs() -> Iterable[tuple[str, str]]:
    for attack_skill in SKILL_NAMES:
        for defense_position in DEFENSE_POSITIONS:
            yield attack_skill, defense_position


__all__ = [
    "ACTION_DIM",
    "ACTION_SCHEMA_VERSION",
    "MODEL_INPUT_DIM",
    "MODEL_INPUT_SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "build_joint_action_batch",
    "build_joint_action_input",
    "iter_action_pairs",
]
