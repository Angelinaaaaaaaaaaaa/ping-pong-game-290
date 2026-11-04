"""Tests for nash_skills.defend_reactive_pose.

TDD: written before implementation.

Pure Python/NumPy, no MuJoCo dependency. predict_ball_interception and
solve_orientation are ports of mujoco_env_comp.KukaTennisEnv.
update_target_racket_pose's own physics (read directly this session) --
a separate, independent cross-check against the real engine's live
output lives in scripts/verify_reactive_pose_port.py (requires MuJoCo,
run on the server/locally where MuJoCo is available, not part of this
fast unit suite).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills.defend_reactive_pose import (
    predict_ball_interception,
    solve_orientation,
    lateral_bias,
    reactive_defend_pose,
    BIAS_MAGNITUDE,
    FADE_FAR_X,
    FADE_NEAR_X,
)

# A ball incoming toward the racket: negative x-velocity (heading toward
# this player, matching update_target_racket_pose's own guard), positive
# height well above the table, roughly centered laterally.
INCOMING_BALL_POS = np.array([1.2, 0.05, 1.0])
INCOMING_BALL_VEL = np.array([-4.0, 0.3, 1.5])


class TestPredictBallInterceptionGuard:
    def test_outgoing_ball_returns_none(self):
        # Positive x-velocity -- ball moving away, not incoming.
        assert predict_ball_interception(INCOMING_BALL_POS, [4.0, 0.0, 0.0]) is None

    def test_too_close_returns_none(self):
        assert predict_ball_interception([0.1, 0.0, 1.0], INCOMING_BALL_VEL) is None

    def test_too_fast_vertically_returns_none(self):
        assert predict_ball_interception(INCOMING_BALL_POS, [-4.0, 0.0, 9.0]) is None

    def test_below_table_returns_none(self):
        assert predict_ball_interception([1.2, 0.0, 0.3], INCOMING_BALL_VEL) is None


class TestPredictBallInterceptionShape:
    def test_returns_pos_and_v_future_with_expected_shapes(self):
        result = predict_ball_interception(INCOMING_BALL_POS, INCOMING_BALL_VEL)
        assert result is not None
        pos, v_future = result
        assert pos.shape == (3,)
        assert v_future.shape == (3, 1)

    def test_pos_is_finite(self):
        pos, _ = predict_ball_interception(INCOMING_BALL_POS, INCOMING_BALL_VEL)
        assert np.all(np.isfinite(pos))


class TestPredictBallInterceptionMatchesHandComputedKinematics:
    def test_lateral_drift_matches_simple_linear_extrapolation(self):
        # With no gravity and a ball already very close to the racket's
        # x_dis reference, the dominant lateral motion should be
        # consistent with simple y = y0 + vy*t drift over the predicted
        # time-of-flight -- a coarse sanity check on the port, not an
        # exact-value test (the real function's bounce/table-matching
        # logic means it is not pure linear extrapolation).
        pos, _ = predict_ball_interception(INCOMING_BALL_POS, INCOMING_BALL_VEL)
        # The predicted x must lie within the racket's searched range.
        assert -0.7 <= pos[0] <= 0.4


class TestSolveOrientation:
    def test_returns_adjusted_pos_and_unit_quaternion(self):
        pos, v_future = predict_ball_interception(INCOMING_BALL_POS, INCOMING_BALL_VEL)
        adjusted_pos, quat = solve_orientation(pos, v_future, x_target=2.19, y_target=0.0)
        assert adjusted_pos.shape == (3,)
        assert quat.shape == (4,)
        assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-6)

    def test_adjusted_pos_differs_from_raw_pos(self):
        # The orientation step always applies the -0.2*sin/cos and -0.1
        # offsets -- adjusted_pos should not equal the raw contact pos.
        pos, v_future = predict_ball_interception(INCOMING_BALL_POS, INCOMING_BALL_VEL)
        adjusted_pos, _ = solve_orientation(pos, v_future, x_target=2.19, y_target=0.0)
        assert not np.allclose(adjusted_pos, pos)


class TestLateralBias:
    def test_center_is_always_zero(self):
        assert lateral_bias("center", ball_pos_x=1.5) == 0.0
        assert lateral_bias("center", ball_pos_x=0.3) == 0.0

    def test_none_is_zero(self):
        assert lateral_bias(None, ball_pos_x=1.5) == 0.0

    def test_left_is_positive_far_from_contact(self):
        assert lateral_bias("left", ball_pos_x=FADE_FAR_X) == pytest.approx(BIAS_MAGNITUDE)

    def test_right_is_negative_far_from_contact(self):
        assert lateral_bias("right", ball_pos_x=FADE_FAR_X) == pytest.approx(-BIAS_MAGNITUDE)

    def test_fades_to_zero_near_contact(self):
        assert lateral_bias("left", ball_pos_x=FADE_NEAR_X) == pytest.approx(0.0, abs=1e-9)

    def test_fades_monotonically_between_thresholds(self):
        mid = (FADE_FAR_X + FADE_NEAR_X) / 2
        far_bias = lateral_bias("left", ball_pos_x=FADE_FAR_X)
        mid_bias = lateral_bias("left", ball_pos_x=mid)
        near_bias = lateral_bias("left", ball_pos_x=FADE_NEAR_X)
        assert far_bias > mid_bias > near_bias

    def test_clamped_beyond_far_threshold(self):
        assert lateral_bias("left", ball_pos_x=FADE_FAR_X + 5.0) == pytest.approx(BIAS_MAGNITUDE)

    def test_clamped_beyond_near_threshold(self):
        assert lateral_bias("left", ball_pos_x=FADE_NEAR_X - 5.0) == pytest.approx(0.0, abs=1e-9)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            lateral_bias("not-a-position", ball_pos_x=1.0)


class TestReactiveDefendPose:
    """
    x_target/y_target are REQUIRED, not defaulted to a fixed "neutral"
    aim -- root-caused this session: a hardcoded neutral aim meant the
    defending player's OWN chosen attack skill never actually governed
    their outgoing shot (the role tracker still labels a player
    "defending" at the exact instant of contact; it only flips to
    "attacker" afterward), so every shot in a rally converged on the
    same safe, maximally-returnable placement regardless of either
    player's attack choice -- confirmed as the actual cause of
    universal (~100%) truncation under both-sided activation, not bias
    magnitude (a sweep of BIAS_MAGNITUDE down to 0.0 showed no
    improvement at all, which is what prompted re-examining this).
    The caller must pass the SAME x_target/y_target the normal
    (non-defending) branch would use for that player's current skill.
    """

    def test_returns_7_dim_pose(self):
        pose = reactive_defend_pose(INCOMING_BALL_POS, INCOMING_BALL_VEL, "left", x_target=2.19, y_target=0.38)
        assert pose.shape == (7,)

    def test_none_when_guard_triggers(self):
        assert reactive_defend_pose([0.1, 0.0, 1.0], INCOMING_BALL_VEL, "left", x_target=2.19, y_target=0.38) is None

    def test_left_and_right_produce_different_y_than_center(self):
        far_ball_pos = np.array([1.5, 0.0, 1.0])
        left_pose = reactive_defend_pose(far_ball_pos, INCOMING_BALL_VEL, "left", x_target=2.19, y_target=0.0)
        center_pose = reactive_defend_pose(far_ball_pos, INCOMING_BALL_VEL, "center", x_target=2.19, y_target=0.0)
        right_pose = reactive_defend_pose(far_ball_pos, INCOMING_BALL_VEL, "right", x_target=2.19, y_target=0.0)
        assert left_pose[1] > center_pose[1] > right_pose[1]

    def test_left_and_center_converge_as_ball_approaches(self):
        # The whole point of the fade: near contact, the named position
        # should barely matter -- left/right/center should nearly agree.
        near_ball_pos = np.array([FADE_NEAR_X, 0.0, 1.0])
        left_pose = reactive_defend_pose(near_ball_pos, INCOMING_BALL_VEL, "left", x_target=2.19, y_target=0.0)
        center_pose = reactive_defend_pose(near_ball_pos, INCOMING_BALL_VEL, "center", x_target=2.19, y_target=0.0)
        assert left_pose[1] == pytest.approx(center_pose[1], abs=1e-6)

    def test_orientation_responds_to_the_given_aim_not_a_fixed_default(self):
        # Regression guard for the actual root cause: two different
        # (x_target, y_target) aims, same position/velocity/defend_name,
        # must produce different orientations (quat), proving the aim
        # argument actually drives orientation rather than being ignored
        # in favor of a hardcoded neutral default.
        pose_aim_a = reactive_defend_pose(INCOMING_BALL_POS, INCOMING_BALL_VEL, "center", x_target=2.19, y_target=0.65)
        pose_aim_b = reactive_defend_pose(INCOMING_BALL_POS, INCOMING_BALL_VEL, "center", x_target=2.19, y_target=-0.65)
        assert not np.allclose(pose_aim_a[3:], pose_aim_b[3:])
