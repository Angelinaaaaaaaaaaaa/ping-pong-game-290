"""Tests for nash_skills.defend_positions.

TDD: written before implementation. Pure Python, no MuJoCo -- covers the
defend-position vocabulary for the attacker/defender extension (Design B:
every player picks an attack skill AND a defend position each rally; the
environment applies whichever half is relevant based on who the ball is
currently heading toward).

Values and names come directly from the spec given for this feature:
    left   -> (-0.5, +0.35)
    right  -> (-0.5, -0.35)
    center -> (-0.5,  0.0)
"relative to the player's own qpos" -- matching the convention already used
by the gantry x-clamp in mujoco_env_comp.py, which applies the *same*
numeric range to both player 1 (qpos[0:2]) and player 2 (qpos[9:11]) with no
sign-flip or reflection. This is different from skills.py's world_target_xy,
which *does* mirror racket-aim targets for player 2 -- defend position is a
gantry-qpos concept like the clamp, not a racket-aim concept like a skill
target, so no mirroring is applied here. Flagged clearly since it's the one
real assumption in this module -- easy to revisit if wrong.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from nash_skills.defend_positions import (
    DEFEND_POSITION_NAMES,
    N_DEFEND_POSITIONS,
    get_defend_position,
    defend_position_index,
    defend_position_from_index,
    world_defend_qpos,
)


class TestDefendPositionNames:
    def test_three_positions(self):
        assert N_DEFEND_POSITIONS == 3
        assert len(DEFEND_POSITION_NAMES) == 3

    def test_names_are_left_center_right(self):
        assert set(DEFEND_POSITION_NAMES) == {"left", "center", "right"}


class TestGetDefendPosition:
    def test_left(self):
        assert get_defend_position("left") == (-0.5, 0.35)

    def test_right(self):
        assert get_defend_position("right") == (-0.5, -0.35)

    def test_center(self):
        assert get_defend_position("center") == (-0.5, 0.0)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown defend position"):
            get_defend_position("up")


class TestDefendPositionIndex:
    def test_index_and_back(self):
        for name in DEFEND_POSITION_NAMES:
            idx = defend_position_index(name)
            assert defend_position_from_index(idx) == name

    def test_distinct_indices(self):
        indices = [defend_position_index(n) for n in DEFEND_POSITION_NAMES]
        assert len(set(indices)) == N_DEFEND_POSITIONS


class TestWorldDefendQpos:
    def test_player1_uses_raw_values_unmirrored(self):
        # Matches the gantry x-clamp precedent: qpos[0:2] for player 1 takes
        # the spec values directly, no transform.
        assert world_defend_qpos(1, "left") == (-0.5, 0.35)
        assert world_defend_qpos(1, "right") == (-0.5, -0.35)
        assert world_defend_qpos(1, "center") == (-0.5, 0.0)

    def test_player2_also_uses_raw_values_unmirrored(self):
        # Key assumption under test: unlike skills.world_target_xy (which
        # mirrors player 2's racket-aim target), defend position does NOT
        # mirror -- same numeric qpos target for both players, matching how
        # the gantry clamp range itself is identical for qpos[0:2] and
        # qpos[9:11] in mujoco_env_comp.py.
        assert world_defend_qpos(2, "left") == (-0.5, 0.35)
        assert world_defend_qpos(2, "right") == (-0.5, -0.35)

    def test_invalid_player_raises(self):
        with pytest.raises(ValueError, match="player must be 1 or 2"):
            world_defend_qpos(3, "left")
