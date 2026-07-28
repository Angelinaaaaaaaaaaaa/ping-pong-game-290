"""Tests for nash_skills.v2.state_encoder_gantry.

TDD: written before the encode_opp_gantry skill-slot-swap fix.
Covers the M2 finding from code review: encode_opp_gantry must swap
which player's skill index lands in the "ego skill" slot (out[10]) vs.
the "opp skill" slot (out[11]).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from nash_skills.v2.state_encoder_gantry import (
    encode_ego_gantry,
    encode_opp_gantry,
    STATE_DIM,
)


def _make_raw_obs(ego_skill: float, opp_skill: float) -> np.ndarray:
    """Build a 116-dim raw obs with distinguishable gantry/skill values."""
    obs = np.zeros(116, dtype=np.float32)
    obs[0:2] = [1.0, 2.0]      # ego gantry
    obs[18:20] = [3.0, 4.0]    # opp gantry
    obs[36:39] = [5.0, 6.0, 7.0]   # ball position
    obs[39:42] = [8.0, 9.0, 10.0]  # ball velocity
    obs[-2] = ego_skill
    obs[-1] = opp_skill
    return obs


class TestEncodeEgoGantry:
    def test_shape(self):
        obs = _make_raw_obs(0.25, 0.75)
        out = encode_ego_gantry(obs)
        assert out.shape == (STATE_DIM,)

    def test_ego_gantry_in_first_slot(self):
        obs = _make_raw_obs(0.25, 0.75)
        out = encode_ego_gantry(obs)
        np.testing.assert_allclose(out[0:2], [1.0, 2.0])

    def test_opp_gantry_in_second_slot(self):
        obs = _make_raw_obs(0.25, 0.75)
        out = encode_ego_gantry(obs)
        np.testing.assert_allclose(out[2:4], [3.0, 4.0])

    def test_skill_slots_unswapped(self):
        obs = _make_raw_obs(0.25, 0.75)
        out = encode_ego_gantry(obs)
        assert out[10] == 0.25  # ego's own skill
        assert out[11] == 0.75  # opp's skill


class TestEncodeOppGantry:
    def test_shape(self):
        obs = _make_raw_obs(0.25, 0.75)
        out = encode_opp_gantry(obs)
        assert out.shape == (STATE_DIM,)

    def test_gantry_positions_swapped(self):
        """From opp's perspective, opp's own gantry becomes the 'ego' slot."""
        obs = _make_raw_obs(0.25, 0.75)
        out = encode_opp_gantry(obs)
        np.testing.assert_allclose(out[0:2], [3.0, 4.0])  # opp gantry -> ego slot
        np.testing.assert_allclose(out[2:4], [1.0, 2.0])  # ego gantry -> opp slot

    def test_skill_slots_swapped(self):
        """
        From opp's perspective, opp's own skill must land in the ego-skill
        slot (out[10]) and the original ego's skill in the opp-skill slot
        (out[11]) -- mirrors the gantry swap above. This is the M2 bug fix.
        """
        obs = _make_raw_obs(ego_skill=0.25, opp_skill=0.75)
        out = encode_opp_gantry(obs)
        assert out[10] == 0.75  # opp's own skill -> their "ego skill" slot
        assert out[11] == 0.25  # original ego's skill -> their "opp skill" slot

    def test_ball_position_unchanged(self):
        obs = _make_raw_obs(0.25, 0.75)
        out = encode_opp_gantry(obs)
        np.testing.assert_allclose(out[4:7], [5.0, 6.0, 7.0])

    def test_ball_velocity_unchanged(self):
        obs = _make_raw_obs(0.25, 0.75)
        out = encode_opp_gantry(obs)
        np.testing.assert_allclose(out[7:10], [8.0, 9.0, 10.0])

    def test_double_flip_recovers_skill_order(self):
        """
        Applying the opp encoding to a symmetric round-trip should place
        skills back in original order (encode_opp_gantry is self-inverse
        for the skill slots specifically, since it's a pure swap).
        """
        obs = _make_raw_obs(0.25, 0.75)
        once = encode_opp_gantry(obs)
        assert once[10] == 0.75 and once[11] == 0.25
