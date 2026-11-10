"""State encoder version attack_defend_joint_v3: v2 plus the ball's spin and the rally status
(the paper's omega_t and ell_t), appended after the choice slots."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills.skills import N_SKILLS, skill_index
from nash_skills.v2.state_encoder_attack_defend import (
    ENCODER_V2, ENCODER_V3, SPIN_SCALE, drop_choice_slots, encode_ego, encode_opp, state_dim,
)

SPIN = np.array([120.0, -40.0, 7.0])


def _obs():
    obs = np.zeros(116, dtype=np.float32)
    obs[0:36] = np.arange(1, 37)
    obs[36:39], obs[39:42] = [0.5, 0.1, 1.0], [-3.0, 0.2, 0.5]
    obs[-2], obs[-1] = skill_index("left") / (N_SKILLS - 1), skill_index("right") / (N_SKILLS - 1)
    return obs


def _info(last_hitter="ego", bounces=0, chance=False, with_status=True):
    info = {"target": np.arange(7.0), "diff_pos": np.arange(3.0), "diff_quat": np.arange(4.0),
            "target_opp": np.arange(7.0), "diff_pos_opp": np.arange(3.0), "diff_quat_opp": np.arange(4.0)}
    if with_status:
        info["ball_angvel"] = SPIN.copy()
        info["rally_status"] = {"last_hitter": last_hitter, "bounces_since_hit": bounces, "receiver_had_chance": chance}
    return info


def test_dimensions_by_version():
    assert state_dim(ENCODER_V2) == 90 and state_dim(ENCODER_V3) == 97
    with pytest.raises(ValueError):
        state_dim("attack_defend_joint_v9")


def test_v3_keeps_the_v2_layout_and_appends_spin_and_status():
    v2 = encode_ego(_obs(), _info(), "left", "center")
    v3 = encode_ego(_obs(), _info(bounces=2, chance=True), "left", "center", encoder=ENCODER_V3)
    assert v2.shape == (90,) and v3.shape == (97,)
    np.testing.assert_array_equal(v3[:90], v2)
    np.testing.assert_allclose(v3[90:93], SPIN * SPIN_SCALE)
    assert v3[93:95].tolist() == [1.0, 0.0] and v3[95] == 2.0 and v3[96] == 1.0


def test_last_hitter_one_hot_and_player_two_view():
    assert encode_ego(_obs(), _info("opp"), "left", "left", encoder=ENCODER_V3)[93:95].tolist() == [0.0, 1.0]
    assert encode_ego(_obs(), _info(None), "left", "left", encoder=ENCODER_V3)[93:95].tolist() == [0.0, 0.0]
    a = encode_ego(_obs(), _info(), "left", "right", encoder=ENCODER_V3)
    b = encode_opp(_obs(), _info(), "left", "right", encoder=ENCODER_V3)
    assert np.flatnonzero(a != b).tolist() == [72, 73]  # only the acting-player slot differs


def test_v3_needs_the_wrapper_info_keys():
    with pytest.raises(KeyError, match="env_wrapper"):
        encode_ego(_obs(), _info(with_status=False), "left", "left", encoder=ENCODER_V3)


def test_drop_choice_slots_keeps_everything_but_74_to_90():
    v3 = encode_ego(_obs(), _info(), "left", "center", encoder=ENCODER_V3)
    out = drop_choice_slots(v3)
    assert out.shape == (81,)
    np.testing.assert_array_equal(out, np.concatenate([v3[:74], v3[90:]]))
    assert drop_choice_slots(encode_ego(_obs(), _info(), "left", "center")).shape == (74,)
