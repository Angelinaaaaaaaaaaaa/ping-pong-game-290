"""Tests for nash_skills.v2.state_encoder_one_hot.

TDD: written before implementation.

This is a NEW, separate encoder for the main Nash-Phi pipeline -- it does
NOT modify nash_skills/v2/state_encoder.py (STATE_DIM there stays 76, and
every existing checkpoint trained on it keeps loading exactly as before).
It replaces that encoder's 2 scalar skill-identity dims (_SKILL_EGO=74,
_SKILL_OPP=75 -- skill_index/(N_SKILLS-1), an arbitrary ordinal scalar
flagged earlier as a likely contributor to the Phi-vs-win-rate
correlation problem) with a 5-dim one-hot block per player, keeping every
other dim byte-for-byte identical to the legacy encoder.

STATE_DIM_ONE_HOT = 74 + 5 + 5 = 84.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from nash_skills.v2.state_encoder import encode_ego as encode_ego_legacy, encode_opp as encode_opp_legacy, STATE_DIM
from nash_skills.v2.state_encoder_one_hot import (
    encode_ego,
    encode_opp,
    skill_one_hot,
    STATE_DIM_ONE_HOT,
)
from nash_skills.skills import SKILL_NAMES, N_SKILLS, skill_index


def _make_obs_info(ego_skill: str, opp_skill: str):
    """Build a minimal, internally-consistent (obs, info) pair covering
    every field nash_skills.v2.state_encoder.encode_ego/encode_opp read,
    with distinguishable values so dim provenance can be checked."""
    obs = np.arange(116, dtype=np.float32)  # distinct value per raw index
    obs[-2] = skill_index(ego_skill) / (N_SKILLS - 1)
    obs[-1] = skill_index(opp_skill) / (N_SKILLS - 1)
    info = {
        "diff_pos": np.array([100.0, 101.0, 102.0], dtype=np.float32),
        "diff_quat": np.array([103.0, 104.0, 105.0, 106.0], dtype=np.float32),
        "target": np.array([107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0], dtype=np.float32),
        "diff_pos_opp": np.array([200.0, 201.0, 202.0], dtype=np.float32),
        "diff_quat_opp": np.array([203.0, 204.0, 205.0, 206.0], dtype=np.float32),
        "target_opp": np.array([207.0, 208.0, 209.0, 210.0, 211.0, 212.0, 213.0], dtype=np.float32),
    }
    return obs, info


class TestShape:
    def test_state_dim_one_hot_is_84(self):
        assert STATE_DIM_ONE_HOT == 84

    def test_encode_ego_shape(self):
        obs, info = _make_obs_info("left", "right")
        assert encode_ego(obs, info).shape == (84,)

    def test_encode_opp_shape(self):
        obs, info = _make_obs_info("left", "right")
        assert encode_opp(obs, info).shape == (84,)


class TestLegacyPrefixUnchanged:
    """Dims [0:74] must be byte-for-byte identical to the legacy 76-dim
    encoder's own output for the same (obs, info) -- reused by slicing
    the legacy encoder's output directly, not reimplemented."""

    def test_encode_ego_prefix_matches_legacy(self):
        obs, info = _make_obs_info("left_short", "center_safe")
        legacy = encode_ego_legacy(obs, info)
        onehot = encode_ego(obs, info)
        np.testing.assert_array_equal(onehot[:74], legacy[:74])

    def test_encode_opp_prefix_matches_legacy(self):
        obs, info = _make_obs_info("left_short", "center_safe")
        legacy = encode_opp_legacy(obs, info)
        onehot = encode_opp(obs, info)
        np.testing.assert_array_equal(onehot[:74], legacy[:74])

    def test_legacy_module_itself_is_untouched(self):
        # Sanity: importing the one-hot module must not have mutated the
        # shared legacy module's own STATE_DIM.
        assert STATE_DIM == 76


class TestSkillOneHot:
    def test_one_hot_shape(self):
        assert skill_one_hot("left").shape == (N_SKILLS,)

    def test_one_hot_is_a_single_hot_vector(self):
        vec = skill_one_hot("center_safe")
        assert vec.sum() == 1.0
        assert vec[skill_index("center_safe")] == 1.0

    def test_every_skill_gets_a_distinct_one_hot(self):
        vectors = [tuple(skill_one_hot(name)) for name in SKILL_NAMES]
        assert len(set(vectors)) == len(SKILL_NAMES)


class TestOneHotSuffixPlacement:
    def test_ego_encode_places_ego_skill_one_hot_at_74_79(self):
        obs, info = _make_obs_info(ego_skill="right", opp_skill="left")
        out = encode_ego(obs, info)
        np.testing.assert_array_equal(out[74:79], skill_one_hot("right"))

    def test_ego_encode_places_opp_skill_one_hot_at_79_84(self):
        obs, info = _make_obs_info(ego_skill="right", opp_skill="left")
        out = encode_ego(obs, info)
        np.testing.assert_array_equal(out[79:84], skill_one_hot("left"))

    def test_opp_encode_keeps_same_ego_opp_slot_order_as_ego_encode(self):
        # Matches the legacy convention: both encode_ego and encode_opp
        # keep ego-skill and opp-skill in the SAME fixed slot positions,
        # regardless of whose perspective the rest of the state is from.
        obs, info = _make_obs_info(ego_skill="right", opp_skill="left")
        out = encode_opp(obs, info)
        np.testing.assert_array_equal(out[74:79], skill_one_hot("right"))
        np.testing.assert_array_equal(out[79:84], skill_one_hot("left"))

    def test_round_trips_every_skill_pair_correctly(self):
        for ego_skill in SKILL_NAMES:
            for opp_skill in SKILL_NAMES:
                obs, info = _make_obs_info(ego_skill, opp_skill)
                out = encode_ego(obs, info)
                assert out[74:79].argmax() == skill_index(ego_skill)
                assert out[79:84].argmax() == skill_index(opp_skill)
                assert out[74:79].sum() == 1.0
                assert out[79:84].sum() == 1.0
