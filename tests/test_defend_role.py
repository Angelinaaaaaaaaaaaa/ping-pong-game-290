"""Tests for nash_skills.defend_role.DefenderRoleTracker.

TDD: written before implementation.

Why this exists: the original role check was a single instantaneous
`ball_vel_x >= 0` read, recomputed from scratch every step. A live trace
(scripts/trace_defend_engagement.py, run with real PPO-driven hitting)
confirmed this is fragile exactly where expected -- right at the step a
racket hits the ball, ball_vel_x passes through a small/noisy value
(observed: -0.0805) before settling into the new direction's cruising
speed (observed: -3.7992 ten steps later). A single bad read at that
moment, or a brief ambiguous reading right at reset/serve, could
misassign or flip-flop the defender.

This module fixes that with two separate rules instead of one fragile
read:
  - Initial latch: who defends first is only decided once ball_vel_x
    clearly exceeds `vel_threshold` in magnitude. Before that, no one is
    latched yet (defending_player is None) and no steering should be
    applied to anyone.
  - Subsequent flips: once latched, the defender only changes when the
    ball's x-position actually crosses the net plane (TABLE_SHIFT) --
    not merely when velocity sign changes. This gives a large hysteresis
    margin around the noisy contact moment.

Pure Python, no MuJoCo dependency -- fully unit tested here.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nash_skills.defend_role import DefenderRoleTracker, DEFAULT_VEL_THRESHOLD


NET_X = 1.5  # matches TABLE_SHIFT


class TestInitialLatch:
    def test_clear_positive_velocity_latches_player2_as_defender(self):
        tracker = DefenderRoleTracker(net_x=NET_X, vel_threshold=0.5)
        role = tracker.update(ball_x=0.1, ball_vel_x=4.5)
        assert role == 2

    def test_clear_negative_velocity_latches_player1_as_defender(self):
        tracker = DefenderRoleTracker(net_x=NET_X, vel_threshold=0.5)
        role = tracker.update(ball_x=2.7, ball_vel_x=-4.5)
        assert role == 1

    def test_ambiguous_near_zero_velocity_does_not_latch_anyone(self):
        tracker = DefenderRoleTracker(net_x=NET_X, vel_threshold=0.5)
        role = tracker.update(ball_x=1.5, ball_vel_x=0.01)
        assert role is None

    def test_ambiguous_reading_does_not_permanently_lock_in_wrong_defender(self):
        # Right after reset, a step or two of near-zero velocity (e.g.
        # serve impulse not yet fully applied) must not get stuck --
        # once a clear reading arrives, it must latch correctly, not be
        # blocked by the earlier ambiguous no-op.
        tracker = DefenderRoleTracker(net_x=NET_X, vel_threshold=0.5)
        assert tracker.update(ball_x=0.0, ball_vel_x=0.0) is None
        assert tracker.update(ball_x=0.0, ball_vel_x=0.02) is None
        assert tracker.update(ball_x=0.01, ball_vel_x=-0.1) is None
        # Now a clear, large reading arrives -- must latch on this, not
        # be stuck from the earlier ambiguous reads.
        role = tracker.update(ball_x=0.05, ball_vel_x=4.5)
        assert role == 2

    def test_boundary_at_exactly_threshold_does_not_latch(self):
        # Spec says "clearly exceeds" -- exactly at the threshold is not
        # clear, so it should not latch (strict inequality).
        tracker = DefenderRoleTracker(net_x=NET_X, vel_threshold=0.5)
        role = tracker.update(ball_x=0.1, ball_vel_x=0.5)
        assert role is None

    def test_just_above_threshold_latches(self):
        tracker = DefenderRoleTracker(net_x=NET_X, vel_threshold=0.5)
        role = tracker.update(ball_x=0.1, ball_vel_x=0.51)
        assert role == 2


class TestFlipOnNetCrossing:
    def test_role_flips_after_ball_crosses_net_to_other_side(self):
        tracker = DefenderRoleTracker(net_x=NET_X, vel_threshold=0.5)
        # Latch: ball heading toward player 2, starts on player 1's side.
        assert tracker.update(ball_x=0.1, ball_vel_x=4.5) == 2
        # Still on player 1's side (below net) -- no flip yet.
        assert tracker.update(ball_x=1.0, ball_vel_x=4.5) == 2
        # Crosses the net to player 2's side -- still no flip (it's
        # already heading to -- and now arriving at -- player 2, who is
        # correctly still the defender).
        assert tracker.update(ball_x=1.6, ball_vel_x=4.5) == 2
        # Ball gets hit back by player 2: velocity sign flips, but the
        # ball is still on player 2's side of the net (hasn't crossed
        # back yet) -- role must NOT flip yet.
        assert tracker.update(ball_x=2.8, ball_vel_x=-0.08) == 2
        assert tracker.update(ball_x=2.4, ball_vel_x=-3.8) == 2
        # Ball finally crosses back to player 1's side -- NOW it flips.
        role = tracker.update(ball_x=1.4, ball_vel_x=-3.8)
        assert role == 1

    def test_velocity_sign_change_alone_without_crossing_does_not_flip(self):
        # This is the core robustness property: a noisy/transient
        # velocity reading right at a hit (sign change) must not flip
        # the role on its own if the ball hasn't actually crossed the
        # net plane yet.
        tracker = DefenderRoleTracker(net_x=NET_X, vel_threshold=0.5)
        assert tracker.update(ball_x=0.1, ball_vel_x=4.5) == 2
        assert tracker.update(ball_x=2.8, ball_vel_x=4.5) == 2
        # Sign flips (simulated contact noise) but ball_x stays on
        # player 2's side of the net -- must still read defender as 2.
        role = tracker.update(ball_x=2.75, ball_vel_x=-0.05)
        assert role == 2

    def test_role_flips_back_and_forth_across_multiple_crossings(self):
        # Mirrors the live trace: a single rally can have several
        # defend engagements. Each full crossing must flip correctly,
        # with no drift or stuck state across repeated cycles.
        tracker = DefenderRoleTracker(net_x=NET_X, vel_threshold=0.5)
        assert tracker.update(ball_x=0.1, ball_vel_x=4.5) == 2
        assert tracker.update(ball_x=2.8, ball_vel_x=4.5) == 2
        assert tracker.update(ball_x=1.4, ball_vel_x=-3.8) == 1   # 1st flip
        assert tracker.update(ball_x=0.1, ball_vel_x=-3.8) == 1
        assert tracker.update(ball_x=1.6, ball_vel_x=3.5) == 2    # 2nd flip
        assert tracker.update(ball_x=2.8, ball_vel_x=3.5) == 2
        role = tracker.update(ball_x=1.4, ball_vel_x=-3.9)        # 3rd flip
        assert role == 1

    def test_no_flip_while_staying_on_same_side_at_cruising_speed(self):
        tracker = DefenderRoleTracker(net_x=NET_X, vel_threshold=0.5)
        assert tracker.update(ball_x=0.1, ball_vel_x=4.5) == 2
        for x in (0.5, 0.9, 1.3, 1.7, 2.1, 2.5):
            assert tracker.update(ball_x=x, ball_vel_x=4.5) == 2


class TestResetBetweenRallies:
    def test_reset_clears_latch_for_a_new_rally(self):
        tracker = DefenderRoleTracker(net_x=NET_X, vel_threshold=0.5)
        assert tracker.update(ball_x=0.1, ball_vel_x=4.5) == 2
        tracker.reset()
        assert tracker.defending_player is None
        # A fresh ambiguous reading right after reset must not resolve
        # to the previous rally's latched value.
        assert tracker.update(ball_x=0.0, ball_vel_x=0.0) is None

    def test_reset_allows_relatching_to_a_different_player_next_rally(self):
        tracker = DefenderRoleTracker(net_x=NET_X, vel_threshold=0.5)
        assert tracker.update(ball_x=0.1, ball_vel_x=4.5) == 2
        tracker.reset()
        role = tracker.update(ball_x=2.7, ball_vel_x=-4.5)
        assert role == 1


class TestDefaults:
    def test_default_vel_threshold_is_well_clear_of_observed_contact_noise(self):
        # Live trace observed ~0.08 magnitude noise right at a hit;
        # cruising speed observed ~3.5-4.6. The default threshold must
        # sit well above the noise and well below cruising speed.
        assert 0.08 < DEFAULT_VEL_THRESHOLD < 3.5

    def test_default_net_x_matches_table_shift(self):
        tracker = DefenderRoleTracker()
        assert tracker.net_x == 1.5
