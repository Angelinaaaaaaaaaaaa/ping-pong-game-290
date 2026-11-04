"""Tests for nash_skills.defend_target_pose.

TDD: written before implementation.

Option B for the attacker/defender mechanism: express a defend position
as a racket-POSE target, fed through the SAME already-trained target-
tracking interface every attack skill already uses
(mujoco_env_comp.KukaTennisEnv.set_target_pose / set_target_pose_opp),
instead of overriding the PPO's gantry ACTION directly (the hard-override
/ soft-bias mechanisms already found net-harmful in
mechanism_validation_matched.py / soft_bias_sweep.py). The PPO's own
policy decides the action in response; only its input target changes --
exactly how the 5 existing attack skills already work.

defend_target_pose(name) builds the 7-dim [x,y,z,qx,qy,qz,qw] pose;
set_target_pose_opp (unchanged, existing code) handles player-2
mirroring, so this always builds the PLAYER-1-FRAME pose regardless of
which player will receive it.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills.defend_target_pose import (
    defend_target_pose,
    DefendPoseEngagementTracker,
    DEFEND_POSE_X,
    DEFEND_POSE_Z,
    DEFEND_POSE_Y,
)
from nash_skills.defend_positions import DEFEND_POSITION_NAMES


class TestDefendTargetPose:
    def test_returns_7_dim_array(self):
        pose = defend_target_pose("left")
        assert pose.shape == (7,)

    def test_position_uses_defend_pose_x_and_z_constants(self):
        pose = defend_target_pose("center")
        assert pose[0] == pytest.approx(DEFEND_POSE_X)
        assert pose[2] == pytest.approx(DEFEND_POSE_Z)

    def test_lateral_y_matches_defend_pose_y_constants(self):
        # NOT the same as defend_positions.get_defend_position()'s y --
        # that's a different (gantry-qpos) consumer's scale. See
        # defend_target_pose.py's "UPDATE 2" docstring: the racket-pose
        # y was empirically found to need roughly double that magnitude.
        for name in DEFEND_POSITION_NAMES:
            pose = defend_target_pose(name)
            assert pose[1] == pytest.approx(DEFEND_POSE_Y[name])

    def test_lateral_y_is_not_the_gantry_qpos_value(self):
        # Regression guard for the bug this fix addresses: left/right's
        # racket-pose y must NOT equal defend_positions.py's gantry-qpos
        # y (0.35/-0.35) -- that was the magnitude mismatch that left
        # Option B net-harmful even after the x/z fix.
        from nash_skills.defend_positions import get_defend_position
        for name in ("left", "right"):
            pose = defend_target_pose(name)
            _, gantry_y = get_defend_position(name)
            assert pose[1] != pytest.approx(gantry_y)

    def test_quaternion_is_a_unit_quaternion(self):
        pose = defend_target_pose("right")
        quat = pose[3:]
        assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-6)

    def test_same_orientation_for_every_position(self):
        # Only the lateral position should vary -- the first-guess
        # orientation choice is fixed, deliberately not position-dependent.
        quats = [defend_target_pose(name)[3:] for name in DEFEND_POSITION_NAMES]
        for q in quats[1:]:
            np.testing.assert_allclose(q, quats[0])

    def test_unknown_position_raises(self):
        with pytest.raises(ValueError):
            defend_target_pose("nonexistent")


class TestDefendPoseEngagementTracker:
    """
    UPDATE: originally mirrored DefendEngagementTracker's latch/reset
    shape with a GANTRY-QPOS proximity check against world_defend_qpos --
    but that's a different consumer's target, in a different space, than
    what Option B actually commands (a racket-POSE target via
    set_target_pose/set_target_pose_opp). Root-caused this session: with
    the qpos check, "done" almost never fired (qpos rarely lands near
    world_defend_qpos when the policy is instead driving toward a
    racket-pose target with no relation to that qpos value), so
    use_defend_target stayed True for the whole engagement, defend_target_
    pose()'s STATIC pose permanently overrode the ball-reactive
    update_target_racket_pose(_opp) call every step (see env_wrapper.py's
    patched_update/patched_update_opp -- the real, ball-tracking update is
    never invoked at all while use_defend_target is True), and the racket
    sat frozen at the pre-positioning point through the entire rally --
    fully explaining why delta_tested_wr stayed roughly constant
    (-0.24..-0.31) across three different pose-value fixes: the specific
    target value never mattered because control never handed back to
    ball-reactive tracking in the first place.

    Fix: check proximity in the SAME space being commanded -- the actual
    racket end-effector position (nash_skills.env_wrapper.py now passes
    data.body('tennis_racket'/'tennis_racket_opp').xpos) against
    defend_target_pose(defend_name)[:3], not qpos against
    world_defend_qpos. Once within tolerance, "done" latches and
    env_wrapper.py's patched_update(_opp) falls through to the real
    ball-reactive call for the rest of the phase.
    """

    def setUp_tracker(self):
        return DefendPoseEngagementTracker()

    def test_steers_while_far_from_target(self):
        t = self.setUp_tracker()
        far = (0.0, 0.0, 0.0)
        assert t.should_steer(2, defending_player=2, racket_pos=far, defend_name="left") is True

    def test_reaching_target_marks_done_and_stops(self):
        t = self.setUp_tracker()
        far = (0.0, 0.0, 0.0)
        at_target = tuple(defend_target_pose("left")[:3])
        t.should_steer(2, defending_player=2, racket_pos=far, defend_name="left")
        result = t.should_steer(2, defending_player=2, racket_pos=at_target, defend_name="left")
        assert result is False
        assert t.is_done(2) is True

    def test_does_not_reengage_once_done_same_phase(self):
        t = self.setUp_tracker()
        far = (0.0, 0.0, 0.0)
        at_target = tuple(defend_target_pose("left")[:3])
        t.should_steer(2, defending_player=2, racket_pos=far, defend_name="left")
        t.should_steer(2, defending_player=2, racket_pos=at_target, defend_name="left")
        result = t.should_steer(2, defending_player=2, racket_pos=far, defend_name="left")
        assert result is False

    def test_reset_clears_latch_for_new_rally(self):
        t = self.setUp_tracker()
        far = (0.0, 0.0, 0.0)
        at_target = tuple(defend_target_pose("left")[:3])
        t.should_steer(2, defending_player=2, racket_pos=far, defend_name="left")
        t.should_steer(2, defending_player=2, racket_pos=at_target, defend_name="left")
        t.reset()
        assert t.is_done(2) is False

    def test_no_defend_name_never_engages(self):
        t = self.setUp_tracker()
        far = (0.0, 0.0, 0.0)
        assert t.should_steer(2, defending_player=2, racket_pos=far, defend_name=None) is False

    def test_not_currently_defending_never_engages(self):
        t = self.setUp_tracker()
        far = (0.0, 0.0, 0.0)
        assert t.should_steer(1, defending_player=2, racket_pos=far, defend_name="left") is False
