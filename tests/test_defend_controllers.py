"""Tests for nash_skills.defend_controllers.compute_action.

TDD: written before implementation.

This module exists to run a strict, instrumented A/B between this
project's original defend-steering design and a teammate's structurally
similar "V3" pipeline, which showed healthier results (sustained
rallies, high return-success) despite overriding the same two gantry
action components. It isolates two independent design axes rather than
guessing at a single fix:

  - approach law while short of target: "bangbang" (this project's
    original _steer_toward -- full speed the whole way, lands exactly on
    target in one discrete step) vs "proportional" (the teammate's
    style -- action = clip(error * gain, -1, 1), smooth deceleration).
  - what happens once the target is reached: "release" (this project's
    current nash_skills.defend_engagement design -- stop overriding,
    return None, hand gantry control fully back to the PPO controller)
    vs "freeze" (the teammate's style -- keep overriding with an
    explicit (0, 0) command for the rest of the defending phase).

Four named controllers cover the full 2x2 factorial. Pure Python, no
MuJoCo dependency.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from nash_skills.defend_controllers import compute_action, CONTROLLER_NAMES, PROPORTIONAL_GAIN

GANTRY_STEP = 0.01
TOLERANCE = 0.01


class TestUnknownController:
    def test_raises_on_unknown_name(self):
        with pytest.raises(ValueError):
            compute_action("nonsense", (0.0, 0.0), (1.0, 1.0), GANTRY_STEP, TOLERANCE, False)


class TestBangbangApproach:
    def test_full_speed_when_far(self):
        action, reached = compute_action(
            "bangbang_release", current_qpos=(-0.5, 0.0), target_qpos=(-0.5, 0.35),
            gantry_step=GANTRY_STEP, tolerance=TOLERANCE, reached_before=False,
        )
        assert action == (0.0, 1.0)  # saturated toward +y
        assert reached is False

    def test_exact_landing_within_one_gantry_step(self):
        # error (0.005) < gantry_step (0.01) -- unclipped, lands exactly.
        action, reached = compute_action(
            "bangbang_release", current_qpos=(-0.5, 0.345), target_qpos=(-0.5, 0.35),
            gantry_step=GANTRY_STEP, tolerance=0.001, reached_before=False,
        )
        assert action == pytest.approx((0.0, 0.5))


class TestProportionalApproach:
    def test_scales_with_gain_and_clips(self):
        # error=0.35 in y; gain=8 -> 2.8, clipped to 1.0
        action, reached = compute_action(
            "proportional_release", current_qpos=(-0.5, 0.0), target_qpos=(-0.5, 0.35),
            gantry_step=GANTRY_STEP, tolerance=TOLERANCE, reached_before=False,
        )
        assert action == (0.0, 1.0)
        assert reached is False

    def test_unclipped_in_linear_zone(self):
        # error=0.05 in y; gain=8 -> 0.4, well under 1.0 -- not clipped.
        action, _ = compute_action(
            "proportional_release", current_qpos=(-0.5, 0.30), target_qpos=(-0.5, 0.35),
            gantry_step=GANTRY_STEP, tolerance=TOLERANCE, reached_before=False,
        )
        assert action == pytest.approx((0.0, 0.05 * PROPORTIONAL_GAIN))


class TestReleaseOnceReached:
    def test_release_returns_none_action_once_within_tolerance(self):
        action, reached = compute_action(
            "bangbang_release", current_qpos=(-0.5, 0.35), target_qpos=(-0.5, 0.35),
            gantry_step=GANTRY_STEP, tolerance=TOLERANCE, reached_before=False,
        )
        assert action is None
        assert reached is True

    def test_release_stays_none_once_reached_before_even_if_now_far(self):
        # Simulates the PPO's own motion dragging the gantry away again
        # during the return stroke -- release must not re-engage.
        action, reached = compute_action(
            "bangbang_release", current_qpos=(-0.5, -0.5), target_qpos=(-0.5, 0.35),
            gantry_step=GANTRY_STEP, tolerance=TOLERANCE, reached_before=True,
        )
        assert action is None
        assert reached is True


class TestFreezeOnceReached:
    def test_freeze_returns_zero_action_once_within_tolerance(self):
        action, reached = compute_action(
            "bangbang_freeze", current_qpos=(-0.5, 0.35), target_qpos=(-0.5, 0.35),
            gantry_step=GANTRY_STEP, tolerance=TOLERANCE, reached_before=False,
        )
        assert action == (0.0, 0.0)
        assert reached is True

    def test_freeze_stays_zero_even_if_now_far(self):
        # Unlike release, freeze keeps actively commanding zero -- this
        # is the key behavioral difference under test.
        action, reached = compute_action(
            "bangbang_freeze", current_qpos=(-0.5, -0.5), target_qpos=(-0.5, 0.35),
            gantry_step=GANTRY_STEP, tolerance=TOLERANCE, reached_before=True,
        )
        assert action == (0.0, 0.0)
        assert reached is True

    def test_proportional_freeze_also_zeroes(self):
        action, reached = compute_action(
            "proportional_freeze", current_qpos=(-0.5, 0.35), target_qpos=(-0.5, 0.35),
            gantry_step=GANTRY_STEP, tolerance=TOLERANCE, reached_before=False,
        )
        assert action == (0.0, 0.0)
        assert reached is True


class TestAllFourNamesExist:
    def test_exactly_four_controllers(self):
        assert set(CONTROLLER_NAMES) == {
            "bangbang_release", "proportional_release",
            "bangbang_freeze", "proportional_freeze",
        }

    def test_every_named_controller_runs_without_error(self):
        for name in CONTROLLER_NAMES:
            action, reached = compute_action(
                name, (-0.5, 0.0), (-0.5, 0.35), GANTRY_STEP, TOLERANCE, False,
            )
            assert reached is False
            assert action is not None  # still approaching, far from target
