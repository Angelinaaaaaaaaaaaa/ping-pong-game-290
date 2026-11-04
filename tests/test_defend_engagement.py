"""Tests for nash_skills.defend_engagement.DefendEngagementTracker.

TDD: written before implementation.

Why this exists: a live 6-combination experiment (attack={left,right} x
defend={left,center,right}, real PPO hitting, fixed DefenderRoleTracker)
came back with the defender winning almost never in EVERY combination
(opp_wr 0-1/10), and a no-override control (set_defend_positions(None,
None)) showed the opposite: without any gantry override, the defender
essentially never loses outright (ego_wins=0/10). The defend-steering
override was overriding the defender's gantry for the ENTIRE defending
phase -- including the moment of the actual return swing, when the PPO
controller needs free gantry control. Root cause: the override never
released control back to PPO.

Fix: defend_position is a PRE-POSITIONING action only. Steer toward the
target while the defending phase is new; once the target is reached
(within DEFEND_REACHED_TOLERANCE -- chosen to match the default
gantry_step, since the steering law in defend_steering._steer_toward
closes any smaller gap exactly on the next step once unclipped), latch
"done" and stop overriding for the REST of that defending phase, even if
the PPO's own racket/arm motion subsequently drags the gantry away from
the target again while executing the stroke. The latch only clears when
DefenderRoleTracker reports a genuinely new defending phase for that
player (a transition into them, which by DefenderRoleTracker's own
construction only happens after a real net crossing).

Pure Python, no MuJoCo dependency -- fully unit tested here, mirroring
the existing defend_role.py / defend_steering.py pattern. The live wiring
lives in nash_skills/env_wrapper.py.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nash_skills.defend_engagement import DefendEngagementTracker, DEFEND_REACHED_TOLERANCE


FAR = (-0.5, 0.0)      # distance 0.35 from "left" target (-0.5, 0.35)
LEFT_TARGET = (-0.5, 0.35)
NEAR = (-0.5, 0.345)   # distance 0.005 from target -- within tolerance
AT_TARGET = (-0.5, 0.35)


class TestSteeringActiveBeforeTargetReached:
    def test_steers_while_far_from_target(self):
        tracker = DefendEngagementTracker()
        assert tracker.should_steer(2, defending_player=2, qpos=FAR, defend_name="left") is True

    def test_steers_while_within_tolerance_but_not_yet_marked_done(self):
        # Just below the boundary -- still "not yet processed" the first
        # time it's seen; should_steer marks it done and returns False in
        # the SAME call once within tolerance (no extra corrective nudge
        # needed -- see test_steering_disabled_after_target_reached).
        tracker = DefendEngagementTracker()
        result = tracker.should_steer(2, defending_player=2, qpos=NEAR, defend_name="left")
        assert result is False  # already within tolerance this call
        assert tracker.is_done(2) is True


class TestSteeringDisabledAfterTargetReached:
    def test_reaching_target_marks_done_and_stops_steering(self):
        tracker = DefendEngagementTracker()
        assert tracker.should_steer(2, defending_player=2, qpos=FAR, defend_name="left") is True
        # Now within tolerance.
        result = tracker.should_steer(2, defending_player=2, qpos=AT_TARGET, defend_name="left")
        assert result is False
        assert tracker.is_done(2) is True

    def test_subsequent_calls_in_same_phase_stay_disabled(self):
        tracker = DefendEngagementTracker()
        tracker.should_steer(2, defending_player=2, qpos=FAR, defend_name="left")
        tracker.should_steer(2, defending_player=2, qpos=AT_TARGET, defend_name="left")
        # Still "defending_player=2", same phase -- repeated queries at
        # the target must keep returning False.
        for _ in range(5):
            assert tracker.should_steer(2, defending_player=2, qpos=AT_TARGET, defend_name="left") is False


class TestNoReengageWhenPPOMovesAwayDuringReturnStroke:
    def test_does_not_reengage_when_qpos_drifts_far_from_target_same_phase(self):
        tracker = DefendEngagementTracker()
        tracker.should_steer(2, defending_player=2, qpos=FAR, defend_name="left")
        tracker.should_steer(2, defending_player=2, qpos=AT_TARGET, defend_name="left")
        assert tracker.is_done(2) is True

        # PPO's own arm/racket motion drags the gantry far from the
        # target while executing the stroke -- still the SAME defending
        # phase (defending_player hasn't changed). Must NOT re-engage.
        drifted = (-0.5, -0.2)  # far from (-0.5, 0.35) again
        result = tracker.should_steer(2, defending_player=2, qpos=drifted, defend_name="left")
        assert result is False
        assert tracker.is_done(2) is True


class TestResetOnNewDefendingPhase:
    def test_done_clears_when_role_transitions_back_to_same_player(self):
        tracker = DefendEngagementTracker()
        tracker.should_steer(2, defending_player=2, qpos=FAR, defend_name="left")
        tracker.should_steer(2, defending_player=2, qpos=AT_TARGET, defend_name="left")
        assert tracker.is_done(2) is True

        # A real net crossing happens: role goes to player 1 for a while...
        tracker.should_steer(1, defending_player=1, qpos=(-0.5, 0.0), defend_name="left")
        # ...then crosses back -- player 2 is defending again, a NEW phase.
        result = tracker.should_steer(2, defending_player=2, qpos=FAR, defend_name="left")
        assert result is True
        assert tracker.is_done(2) is False

    def test_done_does_not_clear_while_role_stays_on_same_player(self):
        # Calling should_steer repeatedly with the SAME defending_player
        # must not look like "a new phase" each time.
        tracker = DefendEngagementTracker()
        tracker.should_steer(2, defending_player=2, qpos=FAR, defend_name="left")
        tracker.should_steer(2, defending_player=2, qpos=AT_TARGET, defend_name="left")
        assert tracker.is_done(2) is True
        tracker.should_steer(2, defending_player=2, qpos=AT_TARGET, defend_name="left")
        assert tracker.is_done(2) is True  # still done, not cleared

    def test_reset_clears_all_state_for_a_new_rally(self):
        tracker = DefendEngagementTracker()
        tracker.should_steer(2, defending_player=2, qpos=FAR, defend_name="left")
        tracker.should_steer(2, defending_player=2, qpos=AT_TARGET, defend_name="left")
        assert tracker.is_done(2) is True

        tracker.reset()
        assert tracker.is_done(2) is False
        assert tracker.is_done(1) is False
        # A fresh phase right after reset must be treated as new, not
        # blocked by stale "prev defending player" bookkeeping.
        result = tracker.should_steer(2, defending_player=2, qpos=FAR, defend_name="left")
        assert result is True


class TestDisablingDefendPositionsLeavesBehaviorUnchanged:
    def test_no_defend_name_never_steers_even_while_defending(self):
        tracker = DefendEngagementTracker()
        result = tracker.should_steer(2, defending_player=2, qpos=FAR, defend_name=None)
        assert result is False
        assert tracker.is_done(2) is False  # never engaged, nothing to latch

    def test_no_defend_name_is_insensitive_to_phase_changes(self):
        tracker = DefendEngagementTracker()
        for _ in range(3):
            assert tracker.should_steer(2, defending_player=2, qpos=FAR, defend_name=None) is False


class TestNotCurrentlyDefending:
    def test_attacker_is_never_steered_regardless_of_done_state(self):
        tracker = DefendEngagementTracker()
        # Player 1 is attacking (defending_player=2) -- never steered,
        # regardless of distance or defend_name.
        result = tracker.should_steer(1, defending_player=2, qpos=FAR, defend_name="left")
        assert result is False

    def test_no_defender_latched_yet_steers_no_one(self):
        tracker = DefendEngagementTracker()
        assert tracker.should_steer(1, defending_player=None, qpos=FAR, defend_name="left") is False
        assert tracker.should_steer(2, defending_player=None, qpos=FAR, defend_name="left") is False


class TestEverEngagedDiagnostic:
    def test_ever_engaged_false_until_first_steer(self):
        tracker = DefendEngagementTracker()
        assert tracker.ever_engaged(2) is False
        tracker.should_steer(2, defending_player=2, qpos=FAR, defend_name="left")
        assert tracker.ever_engaged(2) is True

    def test_ever_engaged_stays_true_after_done_latches(self):
        tracker = DefendEngagementTracker()
        tracker.should_steer(2, defending_player=2, qpos=FAR, defend_name="left")
        tracker.should_steer(2, defending_player=2, qpos=AT_TARGET, defend_name="left")
        assert tracker.ever_engaged(2) is True
        assert tracker.is_done(2) is True

    def test_ever_engaged_false_when_defend_name_is_none(self):
        tracker = DefendEngagementTracker()
        tracker.should_steer(2, defending_player=2, qpos=FAR, defend_name=None)
        assert tracker.ever_engaged(2) is False

    def test_reset_clears_ever_engaged(self):
        tracker = DefendEngagementTracker()
        tracker.should_steer(2, defending_player=2, qpos=FAR, defend_name="left")
        assert tracker.ever_engaged(2) is True
        tracker.reset()
        assert tracker.ever_engaged(2) is False


class TestToleranceConstant:
    def test_tolerance_matches_default_gantry_step(self):
        # Default gantry_step at gantry_speed_scale=1.0 is 0.01 -- the
        # steering law closes any smaller gap exactly on the next step
        # once unclipped, so "within one gantry_step" is the principled
        # definition of "arrived" (see module docstring).
        assert DEFEND_REACHED_TOLERANCE == 0.01


class TestIntegrationWithDefenderRoleTracker:
    """
    Composition smoke test: drives DefenderRoleTracker and
    DefendEngagementTracker together through a synthetic multi-crossing
    rally (mirroring the live trace's rally 0: three separate defending
    phases for player 2 within one rally), confirming the role tracker's
    own behavior is unaffected by layering engagement tracking on top,
    and that engagement correctly re-arms across all three phases.
    """

    def test_three_defend_phases_in_one_rally_each_reset_and_complete(self):
        from nash_skills.defend_role import DefenderRoleTracker

        roles = DefenderRoleTracker(net_x=1.5, vel_threshold=0.5)
        engagement = DefendEngagementTracker()

        # Phase 1: ball heading to player 2 (left of net), steer, reach target.
        assert roles.update(ball_x=0.1, ball_vel_x=4.5) == 2
        assert engagement.should_steer(2, 2, qpos=FAR, defend_name="left") is True
        assert roles.update(ball_x=2.8, ball_vel_x=4.5) == 2
        assert engagement.should_steer(2, 2, qpos=AT_TARGET, defend_name="left") is False
        assert engagement.is_done(2) is True

        # Hit happens, ball heads back -- still player 2's side of the
        # net for a couple of steps (no flip yet, per DefenderRoleTracker).
        assert roles.update(ball_x=2.75, ball_vel_x=-0.05) == 2  # noisy contact, no flip
        assert engagement.should_steer(2, 2, qpos=(-0.5, -0.1), defend_name="left") is False  # still done

        # Ball crosses back over the net -- role flips to player 1.
        # (FAR is specifically far from the "left" target; player 1 here
        # defends to "center" = (-0.5, 0.0), so use a point far from
        # THAT target instead -- LEFT_TARGET = (-0.5, 0.35) is 0.35 away.)
        assert roles.update(ball_x=1.4, ball_vel_x=-3.8) == 1
        assert engagement.should_steer(1, 1, qpos=LEFT_TARGET, defend_name="center") is True

        # Ball crosses the net again, heading back to player 2 -- role
        # flips back to 2: a genuinely new defending phase, must re-arm.
        assert roles.update(ball_x=1.6, ball_vel_x=3.5) == 2
        result = engagement.should_steer(2, 2, qpos=FAR, defend_name="left")
        assert result is True
        assert engagement.is_done(2) is False
