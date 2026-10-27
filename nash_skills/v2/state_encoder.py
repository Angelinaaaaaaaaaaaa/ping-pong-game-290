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
  [9:18]      opp qpos
  [18:27]     ego qvel (joint velocities)
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
  linear: [74] ego skill, [75] opp skill (index / 4)
  onehot: [74:79] ego skill, [79:84] opp skill

STATE_DIM = 76 (linear, for the original Q/Phi pipeline)
STATE_DIM_ONEHOT = 84

NOTE (2026-10-26): the raw-observation slicing in encode_ego/encode_opp was off by
one 9-wide block and has been corrected.  Checkpoints trained before this date saw
different quantities in the same slots -- the ego vector carried the opponent's
qpos where it claimed its own velocities, and the opponent vector carried no
positions at all -- so they cannot be compared against anything trained after it.
`encode_joint` below is the version that also adds the opponent's state and
canonicalises frames; this pair stays ego-only, which is what it always claimed.

The original linear encoding is the default for the existing Q/Phi pipeline
and its 76-dimensional checkpoints. Five-skill self-play explicitly chooses
either that representation or one-hot, preserving both experiment families.
An MLP can learn non-monotone responses from scalar indices, but one-hot avoids
making numerical distance between skill indices an input assumption.

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

from nash_skills.defender import DEFENDER_NAMES, N_DEFENDER

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
_N_SKILLS      = 5
_SKILL_EGO     = slice(74, 74 + _N_SKILLS)
_SKILL_OPP     = slice(74 + _N_SKILLS, 74 + 2 * _N_SKILLS)

STATE_DIM = 76  # original linear encoding and Q/Phi checkpoint contract
STATE_DIM_ONEHOT = 74 + 2 * _N_SKILLS  # 84
SKILL_ENCODINGS = ("linear", "onehot")


def state_dim_for(skill_encoding: str) -> int:
    if skill_encoding == "linear":
        return STATE_DIM
    if skill_encoding == "onehot":
        return STATE_DIM_ONEHOT
    raise ValueError(f"Unknown skill encoding: {skill_encoding!r}")


def skill_encoding_from_state_dict(state_dict: dict) -> str:
    """Identify a 5-skill self-play checkpoint by its first layer width."""
    width = int(state_dict["net.0.weight"].shape[1])
    for encoding in SKILL_ENCODINGS:
        if width == state_dim_for(encoding):
            return encoding
    raise ValueError(
        f"Unsupported self-play input width: {width}. This loader only knows "
        f"the ego-centric contract ({STATE_DIM} linear / {STATE_DIM_ONEHOT} "
        f"onehot); `encode_joint` below is {STATE_DIM_JOINT}-wide and has no "
        f"checkpoints yet. A 90-wide checkpoint is attack_defend_joint_v2 and "
        f"also needs that pipeline's model class (trunk + attack/defend heads "
        f"+ critic), not MetaPolicy.")

# Public aliases for the one-hot layout. The linear layout uses [74] and [75].
SKILL_EGO_SLICE = _SKILL_EGO
SKILL_OPP_SLICE = _SKILL_OPP
N_SKILL_SLOTS = _N_SKILLS


def _one_hot(normalised_index: float) -> np.ndarray:
    """Undo env_wrapper's `skill_index / (N_SKILLS - 1)` and re-expand it.

    The env still hands us the scalar, so decode it here rather than changing
    the observation contract.  Values are exact multiples of 1/(N-1), so
    rounding recovers the index; anything out of range is left all-zero rather
    than silently aliasing onto a real skill.
    """
    out = np.zeros(_N_SKILLS, dtype=np.float32)
    idx = int(round(float(normalised_index) * (_N_SKILLS - 1)))
    if 0 <= idx < _N_SKILLS:
        out[idx] = 1.0
    return out


def encode_ego(obs: np.ndarray, info: dict, skill_encoding: str = "linear") -> np.ndarray:
    """
    Encode the raw 116-dim obs + info using the selected skill representation.

    Parameters
    ----------
    obs  : (116,) float32  — raw observation from KukaTennisEnv.step()
    info : dict            — info dict from KukaTennisEnv.step()

    Returns
    -------
    (76,) for linear or (84,) for onehot, float32
    """
    out = np.zeros(state_dim_for(skill_encoding), dtype=np.float32)
    out[_GANTRY_SLICE] = obs[0:2]       # ego gantry
    out[_ARM_SLICE]    = obs[2:9]       # ego arm joints
    # obs is concatenate([qpos[:18], qvel[:18], ...]), so BOTH players' positions
    # come first and the velocities follow: obs[0:9] p1 qpos | obs[9:18] p2 qpos |
    # obs[18:27] p1 qvel | obs[27:36] p2 qvel.  This line used to read obs[9:18],
    # which is player 2's qpos, not player 1's velocities -- so the ego vector
    # silently carried the opponent's gantry and arm in a slot labelled as its own
    # velocity, and nobody's velocities at all.  `_build_ppo_obs` in the self-play
    # trainers slices the same observation correctly and is the easiest cross-check.
    out[_QVEL_SLICE]   = obs[18:27]     # ego joint velocities
    out[_BALL_POS]     = obs[36:39]     # ball position
    out[_BALL_VEL]     = obs[39:42]     # ball velocity
    out[_DIFF_POS]     = info["diff_pos"]
    out[_DIFF_QUAT]    = info["diff_quat"][:4]
    out[_TARGET]       = info["target"][:7]
    out[_HIST]         = obs[42:78]     # ego pose history
    if skill_encoding == "linear":
        out[74:76] = obs[-2:]
    else:
        out[_SKILL_EGO] = _one_hot(obs[-2])
        out[_SKILL_OPP] = _one_hot(obs[-1])
    return out


def encode_opp(obs: np.ndarray, info: dict, skill_encoding: str = "linear") -> np.ndarray:
    """
    Encode the raw 116-dim obs + info from the opponent's perspective.

    Uses opponent's qpos/qvel (obs[18:36]) and opponent's history (obs[78:114]).
    Skill indices are kept in the same positions as encode_ego so both players'
    models see the same strategy context.

    Parameters
    ----------
    obs  : (116,) float32
    info : dict

    Returns
    -------
    (76,) for linear or (84,) for onehot, float32
    """
    out = np.zeros(state_dim_for(skill_encoding), dtype=np.float32)
    # Same correction as encode_ego, mirrored: these used to read obs[18:20] and
    # obs[20:27], which are player *1*'s velocities, so the opponent vector had no
    # position information whatsoever -- both players' velocities and nobody's
    # positions.  Only the velocity line below was accidentally right.
    out[_GANTRY_SLICE] = obs[9:11]      # opp gantry (qpos[9:11])
    out[_ARM_SLICE]    = obs[11:18]     # opp arm joints
    out[_QVEL_SLICE]   = obs[27:36]     # opp joint velocities
    out[_BALL_POS]     = obs[36:39]     # ball position (same ball)
    out[_BALL_VEL]     = obs[39:42]     # ball velocity
    out[_DIFF_POS]     = info["diff_pos_opp"]
    out[_DIFF_QUAT]    = info["diff_quat_opp"][:4]
    out[_TARGET]       = info["target_opp"][:7]
    out[_HIST]         = obs[78:114]    # opp pose history
    if skill_encoding == "linear":
        out[74:76] = obs[-2:]
    else:
        out[_SKILL_EGO] = _one_hot(obs[-2])
        out[_SKILL_OPP] = _one_hot(obs[-1])
    return out


# ---------------------------------------------------------------------------
# joint_v1: the four corrections, as one caller-indexed encoder
# ---------------------------------------------------------------------------
#
# Status (2026-10-27).  This is the project's only encoder of ours that carries
# both robots and the ball, i.e. the only one that can express the policy as
# Dvij writes it -- pi(s1, s2, b) -- and the only one for which sec 4.2's "the
# actor receives the same physical information available to the potential-based
# planner" could be true.  `encode_ego`/`encode_opp` above are ego-only: the
# opponent's body is absent and so is either player's ready position, so the
# 84-dim policy is closer to pi(s_i, b, a1, a2) with the skills lagged one
# exchange.
#
# The convergence target is `attack_defend_joint_v2` (90-dim, Angelina), whose
# layout is known from the input BatchNorm of her pushed Phi weights:
#
#     0:18  robot A qpos+qvel     42:56  robot A target + dpos + dquat
#    18:36  robot B qpos+qvel     56:70  robot B target + dpos + dquat
#    36:42  ball pos+vel          70:72  attacker one-hot (varies, 50/50)
#                                 72:74  acting-player one-hot (constant)
#                                 74:84  two 5-way skill one-hots
#                                 84:90  two 3-way position one-hots
#
# so 124 = 90 + 36 (the prev_actions block below) - 2 (one role pair instead of
# her two).  Her *code* is not pushed -- only results, weights and launch
# scripts -- so until it is, this encoder is what we have, and the 36-dim
# history is the one block to drop when aligning (measured max|v| = 0.039
# against 0.393 for the joint velocities that are its physical consequence;
# Kexin's v3 cut the same block to 12 and hers to 0).
#
#
# `encode_ego`/`encode_opp` above are kept untouched so the existing results
# stay reproducible.  They have four defects, fixed here together because
# applying only the first makes the strategic picture worse rather than better:
#
#   1. Slice order.  The raw observation is
#          obs[0:9] p1 qpos | obs[9:18] p2 qpos | obs[18:27] p1 qvel | obs[27:36] p2 qvel
#      -- concatenate([qpos[:18], qvel[:18], ...]) in mujoco_env_comp -- but the
#      encoders above index it as [p1 qpos, p1 qvel, p2 qpos, p2 qvel].  The net
#      effect is that `encode_ego` loads p2's qpos into the slot it calls "own
#      joint velocities", and `encode_opp` loads p1's qvel into the slots it
#      calls "opponent gantry/arm".  So player 1 acted on both players'
#      positions and no velocities, and player 2 on both velocities and no
#      positions.  (`_build_ppo_obs` in the self-play trainers slices the same
#      observation correctly, which is the easiest cross-check.)
#
#   2. The opponent's state was never intended to be present.  Correcting (1)
#      alone therefore *removes* the opponent's body, which the attacker/defender
#      game needs: "hit where the opponent is not" is unavailable without it.
#      So both robots' qpos and qvel are included deliberately.
#
#   3. Frames.  Exactly one quantity in the observation is world-frame: the
#      ball (`data.body('ball').xpos` and `data.qvel[-6:-3]`).  It was never
#      converted, so both players received the identical world-frame ball even
#      though player 2's base is rotated 180 degrees -- a given ball y meant
#      opposite physical sides to the two of them.  It is converted here.
#      Everything else -- qpos (gantry included), qvel, prev_actions, and the
#      racket targets/errors in `info` -- is already expressed in the owning
#      player's own frame and needs no conversion.  Verified on the model: both
#      gantries read qpos -0.5 at reset while their rackets sit at world
#      x = -0.243 and x = +3.243, and driving p2's local +y moves its racket
#      toward world -y.  So the local coordinate is already frame-canonical and
#      mirroring it is a category error.  Angelina's 90-dim encoder agrees:
#      slots 0 and 18 of its input BatchNorm both have mean -0.76.
#
#   4. Choice slots were ordered by player id, so slot "ego skill" held player
#      1's choice in *both* vectors.  Here they are ordered self-then-other.
#
# All four matter specifically because one shared set of weights serves both
# players: a slot whose meaning depends on the caller has no single correct
# interpretation, and the stable thing for the network to learn is to ignore it.

TABLE_SHIFT = 1.5       # mujoco_env_comp.TABLE_SHIFT; table centre in x

# Caller-relative layout.  One table, because the bug this encoder exists to fix
# came from four index expressions drifting apart: get one wrong and the slot
# silently holds another quantity.  "self" is always the calling player.
_JGS = {
    "self_qpos":   slice(0, 9),      # gantry(2) in caller frame + arm joints(7)
    "self_qvel":   slice(9, 18),
    "other_qpos":  slice(18, 27),
    "other_qvel":  slice(27, 36),
    "ball_pos":    slice(36, 39),
    "ball_vel":    slice(39, 42),
    "self_dpos":   slice(42, 45),    # racket tracking error, position
    "self_dquat":  slice(45, 49),    # ... and orientation
    "self_target": slice(49, 56),    # racket target pose (pos 3 + quat 4)
    "other_dpos":  slice(56, 59),
    "other_dquat": slice(59, 63),
    "other_target": slice(63, 70),   # what the opponent's solver has committed to
    "self_hist":   slice(70, 106),   # caller's action history, 4 frames x 9
    "role":        slice(106, 108),  # [am I attacker, am I defender]
    "self_skill":  slice(108, 113),
    "self_pos":    slice(113, 116),
    "other_skill": slice(116, 121),
    "other_pos":   slice(121, 124),
}
STATE_DIM_JOINT = 124
STATE_ENCODER_NAME = "ego_centric_joint_v2"

_RAW_QPOS = {1: slice(0, 9),   2: slice(9, 18)}
_RAW_QVEL = {1: slice(18, 27), 2: slice(27, 36)}
_RAW_HIST = {1: slice(42, 78), 2: slice(78, 114)}


def _mirror_xy(x, y):
    """World <-> player-2 frame.  An involution, so one function serves both."""
    return 2.0 * TABLE_SHIFT - x, -y


def encode_joint(obs: np.ndarray, info: dict, player: int, *,
                 attacker_id: int, self_pos: str, other_pos: str) -> np.ndarray:
    """124-dim state for `player`, both robots, everything in `player`'s frame.

    Fixes four things the ego-centric pair above could not, and which matter
    together because one shared network serves both players:

      1. Slice order (see the NOTE in the module docstring).
      2. The opponent's body, racket target and tracking error are present --
         the game's premise is "hit where the opponent is not", and the
         opponent's committed racket pose is a finer statement of the incoming
         shot than its five-way skill label.
      3. Frames.  The ball is the only world-frame quantity in the
         observation, and it was never converted -- so a given ball y meant
         opposite physical sides to the two players, whose bases are rotated
         180 degrees from each other.  It is converted here.  qpos (gantry
         included), qvel, prev_actions and the `info` racket targets are
         already in the owning player's own frame, so they are passed through.
      4. Choices are ordered self-then-other, and the *defender positions* are
         included.  Those were never in the observation at all, so until now the
         only trace of a preposition was where the gantry had got to -- which
         lags the commitment while the controller drives toward it, and the
         attacker decides inside exactly that window.

    `self_pos` / `other_pos` are defender position names; they come from
    SkillEnv.defender_positions because the observation has no room for them.
    `attacker_id` comes from the strike-latched role.
    """
    if player not in (1, 2):
        raise ValueError(f"player must be 1 or 2, got {player!r}")
    if attacker_id not in (1, 2):
        raise ValueError(f"attacker_id must be 1 or 2, got {attacker_id!r}")
    other = 3 - player
    mirror = (player == 2)          # convert world-frame quantities for p2
    out = np.zeros(STATE_DIM_JOINT, dtype=np.float32)

    # --- robots: all nine qpos entries are joint coordinates in the owning
    # player's own frame -- the gantry rail included, not just the seven arm
    # joints.  Two symmetric robots read the same qpos, so these slots are
    # already caller-relative and must be passed through unchanged.  Mirroring
    # the gantry pair (as this encoder first did) treats a rail coordinate as a
    # world position: it offsets `other_qpos` by 2*TABLE_SHIFT and so puts the
    # opponent's gantry around 3.0-3.5 in a slot whose `self_` counterpart
    # stays inside +/-0.5.
    out[_JGS["self_qpos"]] = obs[_RAW_QPOS[player]]
    out[_JGS["self_qvel"]] = obs[_RAW_QVEL[player]]
    out[_JGS["other_qpos"]] = obs[_RAW_QPOS[other]]
    out[_JGS["other_qvel"]] = obs[_RAW_QVEL[other]]

    # --- ball: world frame, so it converts for player 2 only
    bx, by = (_mirror_xy(obs[36], obs[37]) if mirror else (obs[36], obs[37]))
    out[_JGS["ball_pos"]] = (bx, by, float(obs[38]))
    bvx, bvy = ((-obs[39], -obs[40]) if mirror else (obs[39], obs[40]))
    out[_JGS["ball_vel"]] = (bvx, bvy, float(obs[41]))

    # --- racket tracking, both players.  info's plain keys are player 1's and
    # the *_opp ones player 2's, and each is already in that player's own frame
    # -- `target` reads [0, 0, 0.9] while `target_opp` reads [-0.02, -0.49,
    # 0.86], both about the owner's own base rather than about the table.  So
    # neither is mirrored; only the self/other *ordering* is caller-relative.
    key = lambda base, p: (f"{base}_opp" if p == 2 else base)
    for who, tag in ((player, "self"), (other, "other")):
        out[_JGS[f"{tag}_target"]] = np.asarray(info[key("target", who)], dtype=np.float32)[:7]
        out[_JGS[f"{tag}_dpos"]] = np.asarray(info[key("diff_pos", who)], dtype=np.float32)[:3]
        out[_JGS[f"{tag}_dquat"]] = np.asarray(info[key("diff_quat", who)], dtype=np.float32)[:4]
    out[_JGS["self_hist"]] = obs[_RAW_HIST[player]]

    # --- role: the same physical state means different things depending on
    # whether the caller is about to hit or about to wait.  Recoverable from the
    # ball's velocity sign, but cheap to state outright.
    out[_JGS["role"]] = (1.0, 0.0) if player == attacker_id else (0.0, 1.0)

    # --- choices, self then other
    raw_skill = {1: float(obs[-2]), 2: float(obs[-1])}
    out[_JGS["self_skill"]] = _one_hot(raw_skill[player])
    out[_JGS["other_skill"]] = _one_hot(raw_skill[other])
    for tag, pos in (("self_pos", self_pos), ("other_pos", other_pos)):
        if pos is not None:
            out[_JGS[tag]][DEFENDER_NAMES.index(pos)] = 1.0
    return out
