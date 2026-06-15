"""
High-level state encoder for the redesigned Nash pipeline (v2).

The low-level PPO receives a 68-dim slice of the full 116-dim env observation.
The high-level Q / potential models need a richer representation to reason
about game-theoretic skill selection.  This module defines that representation.

State layout (from ego player's perspective)
============================================

The raw 116-dim env obs is:
  [0:9]       ego qpos (joint positions)
                qpos[0:2] = gantry (x, y) — base position
                qpos[2:9] = arm joint angles (7 DOF)
  [9:18]      ego qvel (joint velocities)
  [18:27]     opp qpos
  [27:36]     opp qvel
  [36:39]     ball position (x, y, z)
  [39:42]     ball velocity (vx, vy, vz)
  [42:78]     ego pose history (HISTORY=4 frames × 9 dims = 36)
  [78:114]    opp pose history (36)
  [114]       ego skill index (normalised, 0.0–1.0)
  [115]       opp skill index (normalised, 0.0–1.0)

The info dict from env.step() provides:
  diff_pos     (3,)  — ego racket position error to current target
  diff_quat    (4,)  — ego racket orientation error
  target       (7,)  — ego current target pose [pos(3) + quat(4)]
  diff_pos_opp (3,)  — same for opponent
  diff_quat_opp(4,)
  target_opp   (7,)

Encoded state (same dims for ego and opp)
==========================================

  [0:2]     gantry position (qpos[0:2])
  [2:9]     arm joint angles (qpos[2:9])     ← NEW: full joint info
  [9:18]    joint velocities (qvel[0:9])
  [18:21]   ball position (x, y, z)
  [21:24]   ball velocity (vx, vy, vz)
  [24:27]   diff_pos to current target       (racket tracking error)
  [27:31]   diff_quat to current target
  [31:38]   target pose (7)
  [38:74]   ego pose history (36)
  [74]      ego skill index (normalised)
  [75]      opp skill index (normalised)

STATE_DIM = 76

For the opponent's encoding, the same layout is used but populated from
the opponent's qpos/qvel/history/target fields.  The skill slots are kept
in the same positions (ego skill, opp skill) so the model sees the full
joint strategy even when encoding from the opponent perspective.

Why this design?
  - Gantry position is critical: it determines where the arm can reach,
    directly affecting whether the robot can get to the ball in time.
  - Full joint angles give the model information about arm configuration,
    which is correlated with racket pose and tracking error.
  - Ball position + velocity are the primary game-state signals.
  - Pose history is kept for temporal context (same as PPO observation).
  - Target pose tells the model where the PPO is currently aiming.
  - Skill indices tell the model what strategy each player has committed to.
"""

import numpy as np

HISTORY = 4    # number of pose history frames (must match env)

# Derived layout constants
_GANTRY_SLICE  = slice(0, 2)    # qpos[0:2]
_ARM_SLICE     = slice(2, 9)    # qpos[2:9]
_QVEL_SLICE    = slice(9, 18)   # qvel[0:9]
_BALL_POS      = slice(18, 21)
_BALL_VEL      = slice(21, 24)
_DIFF_POS      = slice(24, 27)
_DIFF_QUAT     = slice(27, 31)
_TARGET        = slice(31, 38)
_HIST          = slice(38, 74)  # 36 = HISTORY * 9
_SKILL_EGO     = 74
_SKILL_OPP     = 75

STATE_DIM = 76  # total encoded state size


def encode_ego(obs: np.ndarray, info: dict) -> np.ndarray:
    """
    Encode the raw 116-dim obs + info into a STATE_DIM-dim ego state vector.

    Parameters
    ----------
    obs  : (116,) float32  — raw observation from KukaTennisEnv.step()
    info : dict            — info dict from KukaTennisEnv.step()

    Returns
    -------
    (STATE_DIM,) float32
    """
    out = np.zeros(STATE_DIM, dtype=np.float32)
    out[_GANTRY_SLICE] = obs[0:2]       # ego gantry
    out[_ARM_SLICE]    = obs[2:9]       # ego arm joints
    out[_QVEL_SLICE]   = obs[9:18]      # ego joint velocities
    out[_BALL_POS]     = obs[36:39]     # ball position
    out[_BALL_VEL]     = obs[39:42]     # ball velocity
    out[_DIFF_POS]     = info["diff_pos"]
    out[_DIFF_QUAT]    = info["diff_quat"][:4]
    out[_TARGET]       = info["target"][:7]
    out[_HIST]         = obs[42:78]     # ego pose history
    out[_SKILL_EGO]    = obs[-2]        # ego normalised skill index
    out[_SKILL_OPP]    = obs[-1]        # opp normalised skill index
    return out


def encode_opp(obs: np.ndarray, info: dict) -> np.ndarray:
    """
    Encode the raw 116-dim obs + info into a STATE_DIM-dim opponent state vector.

    Uses opponent's qpos/qvel (obs[18:36]) and opponent's history (obs[78:114]).
    Skill indices are kept in the same positions as encode_ego so both players'
    models see the same strategy context.

    Parameters
    ----------
    obs  : (116,) float32
    info : dict

    Returns
    -------
    (STATE_DIM,) float32
    """
    out = np.zeros(STATE_DIM, dtype=np.float32)
    out[_GANTRY_SLICE] = obs[18:20]     # opp gantry (qpos[9:11] remapped)
    out[_ARM_SLICE]    = obs[20:27]     # opp arm joints
    out[_QVEL_SLICE]   = obs[27:36]     # opp joint velocities
    out[_BALL_POS]     = obs[36:39]     # ball position (same ball)
    out[_BALL_VEL]     = obs[39:42]     # ball velocity
    out[_DIFF_POS]     = info["diff_pos_opp"]
    out[_DIFF_QUAT]    = info["diff_quat_opp"][:4]
    out[_TARGET]       = info["target_opp"][:7]
    out[_HIST]         = obs[78:114]    # opp pose history
    out[_SKILL_EGO]    = obs[-2]        # ego normalised skill index
    out[_SKILL_OPP]    = obs[-1]        # opp normalised skill index
    return out
