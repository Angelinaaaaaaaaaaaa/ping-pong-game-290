"""Tests for nash_skills.decision_clock: when a strategic decision happens and who attacks."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills.decision_clock import DECISION_TIMINGS, DecisionClock, live_halves


class _RoleTracker:
    def __init__(self):
        self.defending_player = None


class _Data:
    def __init__(self):
        self.qvel = np.zeros(24)


class _Inner:
    def __init__(self):
        self.data = _Data()


class FakeEnv:
    def __init__(self):
        self._env = _Inner()
        self._role_tracker = _RoleTracker()

    def ball_vel_x(self, v):
        self._env.data.qvel[-6] = v


def test_timings():
    assert DECISION_TIMINGS == ("crossing", "contact")
    with pytest.raises(ValueError):
        DecisionClock("bounce")


class TestContactTiming:
    def test_fires_at_first_latch_with_the_receiver_as_attacker(self):
        env, clock = FakeEnv(), DecisionClock("contact")
        clock.reset()
        env.ball_vel_x(4.0)  # ball moving toward player 2 -> player 2 receives and attacks
        assert clock.update(env) == 2

    def test_does_not_fire_again_while_the_ball_keeps_its_direction(self):
        env, clock = FakeEnv(), DecisionClock("contact")
        env.ball_vel_x(4.0)
        clock.update(env)
        assert clock.update(env) is None
        env.ball_vel_x(0.1)  # contact transient inside the dead-band
        assert clock.update(env) is None

    def test_fires_at_each_return_and_alternates_the_attacker(self):
        env, clock = FakeEnv(), DecisionClock("contact")
        env.ball_vel_x(4.0)
        clock.update(env)
        env.ball_vel_x(-3.0)  # player 2 returned it
        assert clock.update(env) == 1
        env.ball_vel_x(3.0)
        assert clock.update(env) == 2

    def test_reset_rearms_the_first_latch(self):
        env, clock = FakeEnv(), DecisionClock("contact")
        env.ball_vel_x(4.0)
        clock.update(env)
        clock.reset()
        assert clock.update(env) == 2


class TestCrossingTiming:
    def test_fires_when_the_role_tracker_changes_with_its_receiver_as_attacker(self):
        env, clock = FakeEnv(), DecisionClock("crossing")
        clock.reset()
        assert clock.update(env) is None
        env._role_tracker.defending_player = 2  # legacy name: the player the ball heads to
        assert clock.update(env) == 2
        assert clock.update(env) is None
        env._role_tracker.defending_player = 1
        assert clock.update(env) == 1


def test_live_halves_are_attack_for_the_attacker_and_defend_for_the_other():
    assert live_halves(attacker=1) == {1: "attack", 2: "defend"}
    assert live_halves(attacker=2) == {1: "defend", 2: "attack"}
