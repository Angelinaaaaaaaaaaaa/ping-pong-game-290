"""
Shared observation index constants for the 5-skill pipeline.

The full observation vector produced by KukaTennisEnv is 116-dimensional:
  [0:9]            ego joint positions (qpos)
  [9:18]           ego joint velocities (qvel)
  [18:27]          opponent joint positions (qpos)
  [27:36]          opponent joint velocities (qvel)
  [36:39]          ball position (x, y, z)
  [39:42]          ball velocity (vx, vy, vz)
  [42:42+H*9]      ego pose history  (H = HISTORY = 4  → 36 dims)
  [78:78+H*9]      opponent pose history              → 36 dims
  [114]            side_target / normalised skill index for ego
  [115]            side_target_opp / normalised skill index for opponent

Total: 9+9+9+9+3+3+36+36+1+1 = 116

Usage:
    from nash_skills.obs_constants import (
        EGO_QPOS, EGO_QVEL, OPP_QPOS, OPP_QVEL,
        BALL_POS, BALL_VEL, EGO_HIST, OPP_HIST,
        SKILL_EGO, SKILL_OPP, OBS_DIM, HISTORY,
    )

Note: The PPO policy receives a 68-dimensional slice, NOT the full 116-dim obs.
      Constants below refer to the full 116-dim obs used by value/potential models.
"""

HISTORY = 4    # number of pose history frames stored per player

# Ego player
EGO_QPOS = slice(0, 9)
EGO_QVEL = slice(9, 18)

# Opponent
OPP_QPOS = slice(18, 27)
OPP_QVEL = slice(27, 36)

# Ball
BALL_POS = slice(36, 39)
BALL_VEL = slice(39, 42)

# Pose history
_HIST_START = 42
EGO_HIST    = slice(_HIST_START,               _HIST_START + HISTORY * 9)
OPP_HIST    = slice(_HIST_START + HISTORY * 9, _HIST_START + 2 * HISTORY * 9)

# Skill / normalised skill index (last two entries)
SKILL_EGO = -2   # index, not slice — scalar
SKILL_OPP = -1   # index, not slice — scalar

OBS_DIM = 116
