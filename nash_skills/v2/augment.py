"""
Data augmentation for 5-skill Nash potential training.

Implements ego/opponent flip: for each decided rally, creates a mirrored
version where the opponent becomes ego and vice versa. This exploits the
symmetry of the game to double the effective dataset size and balance the
ego/opp win counts.

Skill mirroring: left↔right, left_short↔right_short, center_safe stays.

The flipped state encoding is approximate: diff_pos, diff_quat, and target
(dims 24-38) are zeroed because the opponent's tracking error is not stored
in raw_obs. All other fields (joints, velocities, ball, history) are exact.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from typing import Optional

import numpy as np

from nash_skills.skills import skill_index, N_SKILLS

__all__ = ["mirror_skill", "flip_rally", "augment_with_flip"]

_RAW_OBS_DIM = 116

_MIRROR_MAP: dict[str, str] = {
    "left":         "right",
    "left_short":   "right_short",
    "center_safe":  "center_safe",
    "right_short":  "left_short",
    "right":        "left",
}


def mirror_skill(skill: str) -> str:
    """Return the laterally-mirrored counterpart of a skill name."""
    return _MIRROR_MAP[skill]  # raises KeyError for unknown skills


def flip_rally(rally: dict) -> Optional[dict]:
    """
    Create a mirrored version of a rally by swapping ego/opponent roles.

    Returns None for truncated rallies (winner == 0 or None) — there is no
    decided outcome to flip.

    The returned dict is a NEW object; the input is not mutated.
    Raises ValueError if any raw_obs entry is not 116-dimensional.
    """
    winner = rally.get("winner")
    if winner not in (0, 1, 2):
        return None

    flip_skill1 = mirror_skill(rally["skill2"])
    flip_skill2 = mirror_skill(rally["skill1"])
    flip_winner = 3 - winner  # 1 → 2, 2 → 1

    flip_states = []
    flip_raw = []
    for obs_raw in rally["raw_obs"]:
        obs_raw = np.asarray(obs_raw, dtype=np.float32)
        if obs_raw.shape[0] < _RAW_OBS_DIM:
            raise ValueError(
                f"raw_obs must be {_RAW_OBS_DIM}-dim, got {obs_raw.shape[0]}"
            )

        s = np.zeros(76, dtype=np.float32)

        # Kinematic state: original opp becomes new ego
        s[0:2]   = obs_raw[18:20]    # opp gantry  → ego gantry
        s[2:9]   = obs_raw[20:27]    # opp arm     → ego arm
        s[9:18]  = obs_raw[27:36]    # opp qvel    → ego qvel

        # Ball position and velocity: absolute coords, same for both players
        s[18:21] = obs_raw[36:39]
        s[21:24] = obs_raw[39:42]

        # Tracking error (diff_pos[24:27], diff_quat[27:31], target[31:38]):
        # not stored in raw_obs for the opponent → left as zeros

        # Pose history: original opp history → new ego history
        s[38:74] = obs_raw[78:114]

        # Skill indices (normalised): apply mirroring
        s[74] = skill_index(flip_skill1) / (N_SKILLS - 1)  # new ego skill
        s[75] = skill_index(flip_skill2) / (N_SKILLS - 1)  # new opp skill

        flip_states.append(s)
        flip_raw.append(obs_raw.copy())  # deep copy: isolate from original

    return {
        "skill1": flip_skill1,
        "skill2": flip_skill2,
        "states": flip_states,
        "raw_obs": flip_raw,
        "winner": flip_winner,
    }


def augment_with_flip(rallies: list) -> list:
    """
    Augment a rally list by appending a mirrored copy of each decided rally.

    Truncated rallies (winner == 0 or None) are kept as-is without a flip.
    The original list and its rally dicts are not mutated.
    """
    result = []
    for rally in rallies:
        result.append(rally)
        flipped = flip_rally(rally)
        if flipped is not None:
            result.append(flipped)
    return result
