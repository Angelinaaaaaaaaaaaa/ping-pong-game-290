"""Tests for nash_skills.defend_steering.

TDD: written before implementation. Pure Python, no MuJoCo -- covers the
steering math that drives a defending player's gantry toward their chosen
defend position, reusing the existing per-step gantry_step scale rather
than teleporting qpos directly.

`compute_defend_steering_action` takes the already-resolved
`defending_player` (1, 2, or None) as a plain argument instead of
deriving it from an instantaneous ball_vel_x reading -- that
determination is a separate, stateful concern now owned by
nash_skills.defend_role.DefenderRoleTracker (tests/test_defend_role.py),
specifically so the fragile instantaneous-velocity-sign check can't leak
into this pure steering math. This module only answers "given who's
defending and where they want to be, what action gets them there."

The live wiring that reads real qpos/qvel/ball position from MuJoCo,
drives the DefenderRoleTracker, and calls this each step
(nash_skills/env_wrapper.py) is NOT independently verified here -- that
needs an actual rally. It has been verified against a live PPO-driven
trace (scripts/trace_defend_engagement.py): steering converges smoothly
to the unmirrored target with no overshoot, across repeated
engage/disengage cycles within a single rally.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from nash_skills.defend_steering import compute_defend_steering_action


GANTRY_STEP = 0.01  # matches gantry_speed_scale=1.0 * 10.0/1000.0 in mujoco_env_comp.py


class TestWhichPlayerGetsSteered:
    def test_player2_defending_steers_only_player2(self):
        a1, a2 = compute_defend_steering_action(
            defending_player=2, qpos1=(-0.5, 0.0), qpos2=(-0.5, 0.0),
            defend1="left", defend2="left", gantry_step=GANTRY_STEP,
        )
        assert a1 is None          # player 1 is attacking, not steered
        assert a2 is not None      # player 2 is defending, steered

    def test_player1_defending_steers_only_player1(self):
        a1, a2 = compute_defend_steering_action(
            defending_player=1, qpos1=(-0.5, 0.0), qpos2=(-0.5, 0.0),
            defend1="left", defend2="left", gantry_step=GANTRY_STEP,
        )
        assert a1 is not None
        assert a2 is None

    def test_no_defender_latched_yet_steers_no_one(self):
        # DefenderRoleTracker returns None before the initial ambiguous
        # reset/serve velocity resolves -- must not steer anyone then.
        a1, a2 = compute_defend_steering_action(
            defending_player=None, qpos1=(-0.5, 0.0), qpos2=(-0.5, 0.0),
            defend1="left", defend2="left", gantry_step=GANTRY_STEP,
        )
        assert a1 is None and a2 is None


class TestSteeringDirection:
    def test_steers_toward_target_not_away(self):
        # player 2 defending, target "left" = (-0.5, 0.35), currently at (-0.5, 0.0)
        # needs +y movement
        _, a2 = compute_defend_steering_action(
            defending_player=2, qpos1=(-0.5, 0.0), qpos2=(-0.5, 0.0),
            defend1=None, defend2="left", gantry_step=GANTRY_STEP,
        )
        assert a2[1] > 0

    def test_already_at_target_gives_near_zero_action(self):
        _, a2 = compute_defend_steering_action(
            defending_player=2, qpos1=(-0.5, 0.0), qpos2=(-0.5, 0.35),
            defend1=None, defend2="left", gantry_step=GANTRY_STEP,
        )
        assert a2[1] == pytest.approx(0.0, abs=1e-9)


class TestClipping:
    def test_large_distance_clips_to_unit_action(self):
        # far from target relative to gantry_step -> action saturates at +/-1
        _, a2 = compute_defend_steering_action(
            defending_player=2, qpos1=(-0.5, 0.0), qpos2=(-0.5, -10.0),
            defend1=None, defend2="left", gantry_step=GANTRY_STEP,
        )
        assert a2[1] == pytest.approx(1.0)

    def test_action_never_exceeds_unit_bounds(self):
        _, a2 = compute_defend_steering_action(
            defending_player=2, qpos1=(-0.5, 0.0), qpos2=(100.0, 100.0),
            defend1=None, defend2="right", gantry_step=GANTRY_STEP,
        )
        assert -1.0 <= a2[0] <= 1.0
        assert -1.0 <= a2[1] <= 1.0


class TestNoDefendPositionSet:
    def test_none_defend_position_means_no_override_even_while_defending(self):
        a1, a2 = compute_defend_steering_action(
            defending_player=2, qpos1=(-0.5, 0.0), qpos2=(-0.5, 0.0),
            defend1="left", defend2=None, gantry_step=GANTRY_STEP,
        )
        assert a2 is None  # player 2 is defending but chose no defend position

    def test_both_none_returns_no_overrides_at_all(self):
        a1, a2 = compute_defend_steering_action(
            defending_player=2, qpos1=(-0.5, 0.0), qpos2=(-0.5, 0.0),
            defend1=None, defend2=None, gantry_step=GANTRY_STEP,
        )
        assert a1 is None and a2 is None
