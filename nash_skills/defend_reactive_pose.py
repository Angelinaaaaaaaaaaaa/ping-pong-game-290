"""
Velocity/trajectory-informed defend-position pre-positioning -- the fix
for Option B's (nash_skills.defend_target_pose) root cause.

Why defend_target_pose() had to be replaced, not patched: direct reading
of mujoco_env_comp.KukaTennisEnv.update_target_racket_pose(_opp) (this
session) showed its x_target/y_target parameters do NOT bias where the
racket moves to intercept the ball -- the interception position is
computed purely from the ball's own predicted trajectory (bounce
physics), with zero input from x_target/y_target. Those parameters only
select the racket's ORIENTATION, which determines where the ball lands
AFTER being hit (the outgoing shot's aim point) -- exactly how every
attack skill already uses them. There is no existing hook to say "prefer
the left side while still tracking the real ball."

That is why defend_target_pose() had to fully REPLACE the target with a
static, ball-independent pose, and why that caused the discontinuity
root-caused this session: the static pose and the real ball-reactive
target can differ by 0.5+ units (ten times the 0.05 convergence
tolerance) by the time control hands back, with too little remaining
flight-time to close the gap.

Fix: predict the SAME ball-trajectory interception point every step
(ported from update_target_racket_pose's own physics, not reimplemented
from scratch -- see predict_ball_interception / solve_orientation
docstrings), then add a LATERAL BIAS to that real, continuously-updating
prediction instead of replacing it outright. The bias fades to zero as
the ball approaches (ball_pos[0] -> 0), so early positioning is nudged
toward the chosen side while the final contact point converges to the
true interception point -- there is no freeze-then-jump, because the
target was never ball-independent in the first place.

Bias magnitude/fade thresholds are first-guess constants, explicitly not
yet empirically tuned (see BIAS_MAGNITUDE, FADE_FAR_X, FADE_NEAR_X) --
meant to be checked the same way DEFEND_POSE_X/Y/Z were in
defend_target_pose.py, not assumed correct.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from nash_skills.skills import TABLE_SHIFT

__all__ = [
    "predict_ball_interception",
    "solve_orientation",
    "lateral_bias",
    "reactive_defend_pose",
    "BIAS_MAGNITUDE",
    "FADE_FAR_X",
    "FADE_NEAR_X",
]

# First-guess: not yet empirically tuned. Deliberately smaller than the
# ~0.6-0.75 absolute racket-pose y found for a FULL defend position
# (defend_target_pose.py's DEFEND_POSE_Y) -- this is a NUDGE on top of a
# real, already-correct prediction, not a replacement for it.
BIAS_MAGNITUDE = 0.15

# Fade schedule: bias is full strength while the ball is still far away
# (just crossed the net, ball_pos[0] near FADE_FAR_X) and linearly fades
# to zero as the ball nears the racket's reach plane (ball_pos[0] near
# FADE_NEAR_X) -- so the FINAL contact point is the true, unbiased
# interception point. First-guess thresholds, not yet empirically tuned.
FADE_FAR_X = 1.5
FADE_NEAR_X = 0.3

# UPDATE: there is no "neutral" aim to default to -- root-caused this
# session. The role tracker still labels a player "defending" at the
# exact instant they make contact (it only flips to "attacker"
# afterward, once the hit has already happened), so whatever aim
# orientation is solved for during "defending" IS the aim of that
# player's actual outgoing shot. A hardcoded neutral default meant
# neither player's chosen attack skill ever governed their shot --
# every exchange converged on the same safe, maximally-returnable
# placement, which is what caused universal (~100%) truncation under
# both-sided activation (confirmed: sweeping BIAS_MAGNITUDE to 0.0
# changed nothing, which is what prompted re-examining this instead of
# the position-bias logic). reactive_defend_pose() now requires the
# caller to pass the SAME x_target/y_target the defending player's own
# current attack skill would use via the normal (non-defending)
# branch -- see nash_skills.env_wrapper.SkillEnv._reactive_pose_for_player.


def predict_ball_interception(ball_pos, ball_vel, *, bounce_factor=1.0, table_z=0.56, g=-9.81,
                               x_dis=-1.8 + TABLE_SHIFT, z_dis=0.9, vx_dis=4.5, vy_dis=2.0):
    """
    Predict where the ball will be at the racket's reach plane, and its
    velocity there. Returns (pos, v_future) -- pos is a (3,) array
    [x, y, z], v_future is a (3, 1) array -- or None if the guard
    conditions mean no target should be computed this step (ball not
    incoming, too close, too fast vertically, or below the table).

    Ported verbatim from the position-prediction portion of
    mujoco_env_comp.KukaTennisEnv.update_target_racket_pose (lines
    352-394 as read this session) -- NOT reimplemented from scratch, to
    keep bit-for-bit physics parity with every existing attack skill and
    the unmodified "off" baseline. Caller is responsible for the same
    frame flip update_target_racket_pose_opp applies before calling the
    original (negate y, mirror x as 2*TABLE_SHIFT - x, negate vel x/y)
    when computing for player 2 -- this function is frame-agnostic, it
    just reproduces the math given whatever frame its inputs are in.
    """
    ball_pos = np.asarray(ball_pos, dtype=float)
    ball_vel_col = np.zeros((3, 1))
    ball_vel_col[:, 0] = np.asarray(ball_vel, dtype=float)

    if (ball_vel_col[0, 0] > 0 or ball_pos[0] < 0.3
            or ball_vel_col[2, 0] > 8. or ball_pos[2] < table_z + 0.1):
        return None

    v_table = -np.sqrt(max(ball_vel_col[2, 0] * ball_vel_col[2, 0] - 2 * g * (ball_pos[2] - table_z), 0.))
    x_table = ball_vel_col[0, 0] * (v_table - ball_vel_col[2, 0]) / g + ball_pos[0]
    if x_table > 0:
        x_racket = np.linspace(-0.7, min(0.4, x_table), 100)
        T = (x_racket - ball_pos[0]) / ball_vel_col[0, 0]
        y = ball_pos[1] + ball_vel_col[1, 0] * T
        z = ball_pos[2] + ball_vel_col[2, 0] * T + 0.5 * g * T * T
        v_future = np.zeros((3, 100))
        v_future[0, :] = ball_vel_col[0, 0]
        v_future[1, :] = ball_vel_col[1, 0]
        v_future[2, :] = ball_vel_col[2, 0] + g * T
        vz_bounce = -np.sqrt(-2 * g * max(ball_pos[2] - table_z, 0) + ball_vel_col[2, 0] * ball_vel_col[2, 0])
        t_bounce = (vz_bounce - ball_vel_col[2, 0]) / g
        t_remaining = T - t_bounce
        v_future[2, :] = -vz_bounce + g * t_remaining
        z = table_z + 0.5 * g * t_remaining * t_remaining - vz_bounce * t_remaining * bounce_factor
    else:
        x_racket = np.linspace(-0.7, 0.4, 100)
        T = (x_racket - ball_pos[0]) / ball_vel_col[0, 0]
        y = ball_pos[1] + ball_vel_col[1, 0] * T
        z = ball_pos[2] + ball_vel_col[2, 0] * T + 0.5 * g * T * T
        v_future = np.zeros((3, 100))
        v_future[0, :] = ball_vel_col[0, 0]
        v_future[1, :] = ball_vel_col[1, 0]
        v_future[2, :] = ball_vel_col[2, 0] + g * T

    t_dis = (x_racket - x_dis) / vx_dis
    z_dis_arr = z_dis + vy_dis * t_dis + 0.5 * g * t_dis * t_dis
    errs = (z - z_dis_arr) ** 2
    idx = np.argmin(errs)
    pos = np.array([x_racket[idx], y[idx], z[idx]])
    v_future_sel = v_future[:, idx:idx + 1]
    return pos, v_future_sel


def solve_orientation(pos, v_future, x_target, y_target, *, table_z=0.56, g=-9.81):
    """
    Solve for the racket orientation (and the small positional offset
    that goes with it) that reflects the predicted incoming velocity
    v_future toward (x_target, y_target), given contact position pos.
    Returns (adjusted_pos, quat).

    Ported verbatim from the orientation-solving portion of
    update_target_racket_pose (lines 396-422 as read this session).
    """
    pos = np.asarray(pos, dtype=float).copy()
    yzr = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-0.1, 0.1, 100))
    yzr = np.array(yzr).reshape(2, -1).T
    xyzr = np.concatenate([np.ones((yzr.shape[0], 1)), yzr], axis=1)
    xyzr = xyzr / np.linalg.norm(xyzr, axis=1)[:, None]
    ball_vels = np.tile(v_future.T, (100 * 100, 1)).T
    ball_reflected_vels = ball_vels - 2 * np.sum(ball_vels * xyzr.T, axis=0, keepdims=True) * xyzr.T
    vz_hits = -np.sqrt(-2 * g * max(pos[2] - table_z, 0) + ball_reflected_vels[2] * ball_reflected_vels[2])
    t_hits = (vz_hits - ball_reflected_vels[2]) / g
    x_hits = pos[0] + ball_reflected_vels[0] * t_hits
    y_hits = pos[1] + ball_reflected_vels[1] * t_hits
    costs = (x_hits - x_target) ** 2 + (y_hits - y_target) ** 2
    idx = np.argmin(costs)
    z_axis = xyzr[idx]
    theta_z = np.arctan2(pos[2], pos[1])
    x_axis = np.array([0., np.cos(theta_z), np.sin(theta_z)])
    x_axis[0] = (-x_axis[1] * z_axis[1] - x_axis[2] * z_axis[2]) / z_axis[0]
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot = np.array([x_axis, y_axis, z_axis]).T
    quat = R.from_matrix(rot).as_quat()
    pos[2] -= 0.2 * np.sin(theta_z)
    pos[1] -= 0.2 * np.cos(theta_z)
    pos[0] -= 0.1
    return pos, quat


def lateral_bias(defend_name: str, ball_pos_x: float) -> float:
    """
    Signed lateral bias to add to the predicted interception y, for a
    named defend position, faded by how close the ball already is.
    left -> positive (matches the real sign confirmed empirically this
    session via scripts/trace_defend_y_sign.py: receiving a left-aimed
    shot produces positive real racket-pose y); right -> negative;
    center -> always zero.
    """
    if defend_name == "center" or defend_name is None:
        return 0.0
    if defend_name not in ("left", "right"):
        raise ValueError(f"Unknown defend position '{defend_name}'. Choose from: left, center, right")

    fade = (ball_pos_x - FADE_NEAR_X) / (FADE_FAR_X - FADE_NEAR_X)
    fade = min(max(fade, 0.0), 1.0)
    sign = 1.0 if defend_name == "left" else -1.0
    return sign * BIAS_MAGNITUDE * fade


def reactive_defend_pose(ball_pos, ball_vel, defend_name, x_target, y_target):
    """
    Full defend-position pose for this step, or None if the underlying
    trajectory prediction's guard triggered (caller should leave the
    previous pose unchanged this step, exactly as
    update_target_racket_pose_opp's own guard does by simply not calling
    set_target_pose_opp).

    x_target/y_target are REQUIRED -- the defending player's own
    current attack skill's aim (the same values the normal,
    non-defending branch would pass to update_target_racket_pose(_opp)).
    There is no neutral default: the role tracker still labels this
    player "defending" at the exact instant their hit lands, so this
    orientation IS their outgoing shot's aim, not a placeholder.
    """
    predicted = predict_ball_interception(ball_pos, ball_vel)
    if predicted is None:
        return None
    pos, v_future = predicted
    pos = pos.copy()
    pos[1] += lateral_bias(defend_name, float(ball_pos[0]))
    adjusted_pos, quat = solve_orientation(pos, v_future, x_target, y_target)
    return np.concatenate([adjusted_pos, quat])
