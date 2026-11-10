"""
Defend-position pre-positioning via the SAME racket-pose-target channel
every attack skill already uses, instead of overriding the PPO's gantry
ACTION directly.

Why: the hard-override and soft-bias mechanisms
(nash_skills.defend_steering / defend_soft_bias) both override the
ACTION the PPO's own policy outputs -- found net-harmful in every tested
configuration (scripts/mechanism_validation_matched.py,
scripts/soft_bias_sweep.py: harm present at every intervention strength,
even against a random attacker, even when the override released control
well before the rally ended). The low-level PPO
(logs/best_model_tracker1/best_model) was trained purely as a pose
tracker -- its observation is diff_pos/diff_quat/target
(nash_skills.v2.collect_data._build_ppo_obs), it never sees the ball or
opponent directly, and it has only ever been allowed to decide its own
action in response to a target pose. Overriding that action is therefore
out-of-distribution for it. Every one of the 5 existing attack skills
instead works by giving this SAME policy a DIFFERENT racket-pose target
(mujoco_env_comp.KukaTennisEnv.update_target_racket_pose ->
set_target_pose) and letting it decide the arm+gantry motion itself.
This module expresses a defend position the same way: as a racket-pose
target, via set_target_pose/set_target_pose_opp directly (which have no
ball-reactive guards, unlike update_target_racket_pose, so they can be
called unconditionally during pre-positioning) -- the policy's
input->output contract never changes, only the target does.

set_target_pose_opp (unchanged, existing code) already handles player-2
mirroring (position AND rotation) -- defend_target_pose() always builds
the PLAYER-1-FRAME pose; pass that SAME pose to set_target_pose_opp for
player 2 and let its existing mirroring do the work, exactly like
update_target_racket_pose_opp's x_target already does for skills.

Orientation choice: there is no existing "default" racket orientation
anywhere in this codebase divorced from a specific ball trajectory --
update_target_racket_pose always computes orientation from the ball's
predicted reflection. For a pre-positioning pose (no ball reaction yet),
this uses a simple, fixed, face-forward orientation (racket normal
pointing toward the opponent, +x) as a first-guess default. This is the
one part of the design NOT grounded in an existing convention -- it is
meant to be checked empirically (does contact/success rate look sane
once the target is reached) rather than assumed correct; see
scripts/mechanism_validation_target_pose.py.

UPDATE 1: the first-guess DEFEND_POSE_X/DEFEND_POSE_Z (-0.3, 0.66) were
checked exactly this way and found badly wrong -- scripts/
trace_real_defend_target.py recorded the REAL curr_target_opp at
closest-approach-to-contact across 6 unmodified (no-override) rallies:
x ranged -0.239..-0.022, z ranged 0.820..0.872. The old z=0.66 was
~0.16-0.21 too low -- on its own sufficient to explain the zero-contact
result (racket parked well below where the ball actually is). Both
constants are now the midpoint of that measured range. Orientation
itself is still the fixed first-guess (not yet similarly corrected) --
the position/height error was large enough to fully explain the
observed failure on its own, so it was fixed and retested first.

UPDATE 2: fixing x/z alone did NOT close the gap at full validation scale
(avg_delta_tested_wr stayed at -0.308, same magnitude as before the x/z
fix) -- scripts/trace_defend_y_sign.py then checked the remaining
untested variable, the lateral (y) component, which this module had been
borrowing directly from defend_positions.get_defend_position() since the
beginning. That function's values are explicitly documented as
gantry-qpos targets (a different consumer, world_defend_qpos / the old
hard-override mechanism) -- never verified to share a scale with the
racket-pose-target space this module actually feeds. Two hypotheses were
tested empirically against 20 real (no-override) rallies per attack
direction, filtered to phases where the policy actually converged
(diff_pos_opp_norm < 0.05):
  (a) SIGN flip -- refuted. attack="left" produced real curr_target_opp.y
      consistently POSITIVE (19/20 phases in [0.43, 0.92], one outlier at
      -0.51; median 0.63) and attack="right" consistently NEGATIVE (20/20
      in [-0.89, -0.54], median -0.75) -- the same sign as
      get_defend_position's left=+0.35/right=-0.35, not flipped.
  (b) MAGNITUDE mismatch -- confirmed. The real magnitude needed (~0.63
      for left, ~0.75 for right) is roughly double get_defend_position's
      0.35 in both directions -- consistent with 0.35 being calibrated
      for the gantry-qpos joint range, not the racket-pose Cartesian
      range skills.py's own side_target*0.38 (0.38-0.65 depending on
      profile) already lives in.
DEFEND_POSE_Y below replaces the borrowed get_defend_position() lookup
with these directly-measured, racket-pose-specific values -- the same
fix pattern as DEFEND_POSE_X/Z, now applied to the one axis that was
still shared with the other (gantry-qpos) consumer.

DefendPoseEngagementTracker mirrors nash_skills.defend_engagement.
DefendEngagementTracker's latch/reset shape exactly (gantry-qpos
proximity is still a reasonable, already-validated "have I finished
pre-positioning" proxy) -- kept as a SEPARATE instance/class so this
mechanism's engagement state is never shared with or can disturb the
hard-override mechanism's already-validated one.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from nash_skills.defend_positions import DEFEND_POSITION_NAMES

__all__ = [
    "defend_target_pose",
    "DefendPoseEngagementTracker",
    "DEFEND_POSE_X",
    "DEFEND_POSE_Z",
    "DEFEND_POSE_Y",
]

DEFEND_POSE_X = -0.13  # empirically measured, not a guess: midpoint of
                        # curr_target_opp.x at closest-approach-to-contact
                        # across 6 real (no-override) rallies, range was
                        # -0.239..-0.022 (scripts/trace_real_defend_target.py)
DEFEND_POSE_Z = 0.85   # same source, real range was 0.820..0.872 -- the
                        # old guess (0.66) was ~0.16-0.21 too low, which on
                        # its own fully explained the zero-contact result in
                        # scripts/mechanism_validation_target_pose.py: the
                        # racket was simply parked far below where the ball
                        # actually is

# Empirically measured (scripts/trace_defend_y_sign.py, see UPDATE 2
# above) -- NOT the same as defend_positions.get_defend_position()'s y,
# which is calibrated for a different consumer (gantry qpos) and turned
# out to be roughly half the magnitude the racket-pose-target space
# actually needs. "center" is left at 0.0 (not contradicted by any
# measurement -- there is no "attack=center" skill to measure against,
# and 0.0 remains the sensible midpoint between the measured left/right
# values, -0.75..0.63).
DEFEND_POSE_Y = {
    "left": 0.63,    # median of 19/20 real phases in [0.43, 0.92]; one
                      # outlier at -0.51 excluded (single rally, not
                      # representative -- see trace_defend_y_sign.py output)
    "center": 0.0,
    "right": -0.75,  # median of 20/20 real phases in [-0.89, -0.54]
}

# First-guess, fixed orientation: racket normal (z-axis) pointing toward
# the opponent (+x), racket "top" (x-axis) pointing up (+z). See module
# docstring -- not grounded in an existing convention, meant to be
# checked empirically.
_Z_AXIS = np.array([1.0, 0.0, 0.0])
_X_AXIS = np.array([0.0, 0.0, 1.0])
_Y_AXIS = np.cross(_Z_AXIS, _X_AXIS)
_DEFEND_ROTATION_MATRIX = np.array([_X_AXIS, _Y_AXIS, _Z_AXIS]).T
_DEFEND_QUAT = R.from_matrix(_DEFEND_ROTATION_MATRIX).as_quat()


def defend_target_pose(position_name: str) -> np.ndarray:
    """
    7-dim [x, y, z, qx, qy, qz, qw] racket-pose target for a defend
    position, in PLAYER-1 FRAME. Do not mirror this yourself for player
    2 -- pass it straight to KukaTennisEnv.set_target_pose_opp, which
    already mirrors both position and rotation.
    """
    if position_name not in DEFEND_POSITION_NAMES:
        raise ValueError(
            f"Unknown defend position '{position_name}'. Choose from: {DEFEND_POSITION_NAMES}"
        )
    y = DEFEND_POSE_Y[position_name]
    pos = np.array([DEFEND_POSE_X, y, DEFEND_POSE_Z])
    return np.concatenate([pos, _DEFEND_QUAT])


class DefendPoseEngagementTracker:
    """
    Per-player latch: has this player's racket reached the defend
    target POSE for the current defending phase
    (nash_skills.defend_role.DefenderRoleTracker, unchanged, is the sole
    source of phase boundaries)? Checks proximity of the actual racket
    end-effector position (env_wrapper.py passes
    data.body('tennis_racket'/'tennis_racket_opp').xpos) against
    defend_target_pose(defend_name)[:3] -- the SAME pose being commanded
    via set_target_pose/set_target_pose_opp while this latch is open.

    UPDATE: originally checked GANTRY QPOS against
    defend_positions.world_defend_qpos instead (mirroring
    nash_skills.defend_engagement.DefendEngagementTracker, where that
    really is the target being driven toward). For Option B it isn't --
    the policy drives toward a racket-POSE target via its own arm+gantry
    coupling, with no guaranteed relationship to world_defend_qpos's
    value. Root-caused this session (scripts/trace_defend_y_sign.py +
    direct reading of env_wrapper.py's patched_update/patched_update_opp):
    the qpos check almost never satisfied its tolerance, so "done" never
    latched, so use_defend_target stayed True for the entire engagement,
    so the real ball-reactive update_target_racket_pose(_opp) call was
    never reached (env_wrapper.py's patched functions `return` early
    instead of falling through to it) -- the racket sat frozen at the
    static pre-positioning pose through the whole rally, regardless of
    which pose. That fully explains why avg_delta_tested_wr stayed
    roughly constant (-0.24..-0.31) across three different pose-value
    fixes: control never hands back to tracking, so the target value
    never mattered. Checking the racket's actual pose against what's
    actually commanded fixes the handoff itself.

    Tolerance default raised from 0.01 (appropriate for a tight 2D
    gantry-qpos check) to 0.05 (3D Euclidean, racket-pose space) --
    matching the "plausible real contact" bar already used throughout
    this session's trace scripts (trace_real_defend_target.py,
    trace_defend_y_sign.py) to decide whether the policy actually
    converged near a commanded pose.
    """

    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance
        self._done = {1: False, 2: False}
        self._ever_engaged = {1: False, 2: False}
        self._prev_defending_player = None

    def reset(self):
        self._done = {1: False, 2: False}
        self._ever_engaged = {1: False, 2: False}
        self._prev_defending_player = None

    def is_done(self, player: int) -> bool:
        return self._done.get(player, False)

    def ever_engaged(self, player: int) -> bool:
        return self._ever_engaged.get(player, False)

    def should_steer(self, player: int, defending_player, racket_pos, defend_name) -> bool:
        self._clear_done_on_new_phase(defending_player)

        if defending_player != player or defend_name is None:
            return False
        if self._done[player]:
            return False

        self._ever_engaged[player] = True

        target = defend_target_pose(defend_name)[:3]
        dx = target[0] - racket_pos[0]
        dy = target[1] - racket_pos[1]
        dz = target[2] - racket_pos[2]
        if (dx * dx + dy * dy + dz * dz) ** 0.5 <= self.tolerance:
            self._done[player] = True
            return False

        return True

    def _clear_done_on_new_phase(self, defending_player):
        if defending_player is not None and defending_player != self._prev_defending_player:
            self._done[defending_player] = False
        self._prev_defending_player = defending_player
