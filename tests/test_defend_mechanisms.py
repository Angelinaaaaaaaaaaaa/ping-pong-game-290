"""Tests for nash_skills.defend_mechanisms, the switch between the reactive
(receiver) and preparation (player who just hit) defend mechanisms."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from nash_skills.defend_mechanisms import (
    DEFAULT_DEFEND_MECHANISM, DEFEND_MECHANISMS, set_defend_choices, validate_defend_mechanism,
)


class RecordingEnv:
    def __init__(self):
        self.calls = []

    def set_defend_reactive_positions(self, pos1, pos2):
        self.calls.append(("reactive", pos1, pos2))

    def set_defend_preparation_positions(self, pos1, pos2, calibrated=False):
        self.calls.append(("preparation_cal" if calibrated else "preparation", pos1, pos2))


def test_all_mechanisms_are_available_and_reactive_stays_the_default():
    assert set(DEFEND_MECHANISMS) == {"reactive", "preparation", "preparation_cal"}
    assert DEFAULT_DEFEND_MECHANISM == "reactive"


@pytest.mark.parametrize("mechanism", ["reactive", "preparation", "preparation_cal"])
def test_routes_to_exactly_the_chosen_mechanism(mechanism):
    env = RecordingEnv()
    set_defend_choices(env, mechanism, "left", "right")
    assert env.calls == [(mechanism, "left", "right")]


def test_unknown_mechanism_raises_and_calls_nothing():
    env = RecordingEnv()
    with pytest.raises(ValueError):
        set_defend_choices(env, "hard_override", "left", "right")
    assert env.calls == []


def test_validate_returns_the_name_or_raises():
    assert validate_defend_mechanism("preparation") == "preparation"
    with pytest.raises(ValueError):
        validate_defend_mechanism("static_pose")
