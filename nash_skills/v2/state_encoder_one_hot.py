"""
One-hot skill-identity variant of the 76-dim v2 state encoder
(nash_skills.v2.state_encoder) for the main Nash-Phi pipeline.

The legacy encoder represents each player's committed skill as a single
scalar skill_index/(N_SKILLS-1) (nash_skills.skills.skill_index) at fixed
indices _SKILL_EGO=74 / _SKILL_OPP=75. That ordinal scalar was flagged
earlier in this project as a likely contributor to the Phi-vs-win-rate
correlation problem: signal dilution, and an arbitrary ordinal distance
between skills that aren't actually ordered along any single axis (skill
2 isn't "between" skills 1 and 3 in any meaningful sense).

This module REPLACES those 2 scalar dims with a 5-dim one-hot block per
player (10 dims total), keeping every other dim byte-for-byte identical
to the legacy encoder -- literally reused by slicing the legacy
encoder's own output, not reimplemented, so there is no risk of drift
between the two.

This is a NEW, separate, explicitly-named module. It does NOT modify
nash_skills/v2/state_encoder.py -- STATE_DIM there stays 76, and every
existing checkpoint trained on that encoder keeps loading exactly as
before. Only code that explicitly imports THIS module opts into one-hot.
See analysis/experiment_plan.md Sec. 5 for the decision not to mutate
the shared encoder in place.

Layout:
  [0:74]    identical to nash_skills.v2.state_encoder's encode_ego/
            encode_opp indices [0:74] (gantry, arm, qvel, ball pos/vel,
            diff_pos, diff_quat, target, pose history) -- everything
            EXCEPT the 2 scalar skill-identity dims, which legacy puts
            at [74] and [75] and this module drops.
  [74:79]   ego skill one-hot (5 dims, nash_skills.skills.SKILL_NAMES
            order)
  [79:84]   opp skill one-hot (5 dims, same order)

STATE_DIM_ONE_HOT = 84

Calling convention matches the legacy encoder exactly -- encode_ego(obs,
info) / encode_opp(obs, info), no extra parameters -- so switching a
caller over is a one-line import change. Skill identity is recovered
from the same obs[-2]/obs[-1] scalars the legacy encoder reads (set by
nash_skills.env_wrapper.SkillEnv._encode_obs_skills), by inverting
skill_index(name)/(N_SKILLS-1). For N_SKILLS=5 every one of the 5
possible values (0, 0.25, 0.5, 0.75, 1.0) is exactly representable in
float32, so this round-trips with no rounding ambiguity.
"""
import numpy as np

from nash_skills.v2.state_encoder import encode_ego as _encode_ego_legacy, encode_opp as _encode_opp_legacy
from nash_skills.skills import N_SKILLS, skill_index

__all__ = ["encode_ego", "encode_opp", "skill_one_hot", "STATE_DIM_ONE_HOT"]

_LEGACY_PRE_SKILL = 74  # legacy dims [0:74] reused verbatim; [74:76] (the
                        # 2 scalar skill slots) are dropped and replaced

STATE_DIM_ONE_HOT = _LEGACY_PRE_SKILL + 2 * N_SKILLS  # 74 + 5 + 5 = 84


def skill_one_hot(skill: str) -> np.ndarray:
    """5-dim one-hot vector for a skill name, in SKILL_NAMES order."""
    out = np.zeros(N_SKILLS, dtype=np.float32)
    out[skill_index(skill)] = 1.0
    return out


def _skill_index_from_scalar(value: float) -> int:
    return int(round(float(value) * (N_SKILLS - 1)))


def _skill_one_hot_from_scalar(value: float) -> np.ndarray:
    out = np.zeros(N_SKILLS, dtype=np.float32)
    out[_skill_index_from_scalar(value)] = 1.0
    return out


def encode_ego(obs: np.ndarray, info: dict) -> np.ndarray:
    """
    84-dim ego state: legacy dims [0:74] unchanged, plus one-hot
    (ego_skill, opp_skill) appended at [74:84] in place of the legacy
    scalar skill slots. Skill identity is read from obs[-2]/obs[-1],
    exactly like the legacy encoder.
    """
    legacy = _encode_ego_legacy(obs, info)
    out = np.zeros(STATE_DIM_ONE_HOT, dtype=np.float32)
    out[:_LEGACY_PRE_SKILL] = legacy[:_LEGACY_PRE_SKILL]
    out[_LEGACY_PRE_SKILL:_LEGACY_PRE_SKILL + N_SKILLS] = _skill_one_hot_from_scalar(obs[-2])
    out[_LEGACY_PRE_SKILL + N_SKILLS:] = _skill_one_hot_from_scalar(obs[-1])
    return out


def encode_opp(obs: np.ndarray, info: dict) -> np.ndarray:
    """
    84-dim opponent-perspective state: legacy dims [0:74] unchanged
    (opponent's own qpos/qvel/history per nash_skills.v2.state_encoder.
    encode_opp), plus one-hot (ego_skill, opp_skill) appended at [74:84]
    in the SAME slot order as encode_ego -- matches the legacy
    convention of keeping ego/opp skill slots in fixed positions for
    both perspectives.
    """
    legacy = _encode_opp_legacy(obs, info)
    out = np.zeros(STATE_DIM_ONE_HOT, dtype=np.float32)
    out[:_LEGACY_PRE_SKILL] = legacy[:_LEGACY_PRE_SKILL]
    out[_LEGACY_PRE_SKILL:_LEGACY_PRE_SKILL + N_SKILLS] = _skill_one_hot_from_scalar(obs[-2])
    out[_LEGACY_PRE_SKILL + N_SKILLS:] = _skill_one_hot_from_scalar(obs[-1])
    return out
