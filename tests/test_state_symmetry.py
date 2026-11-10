"""Player-swap symmetry of the Design B joint state (nash_skills.v2.state_symmetry) and its use
as training-data augmentation in the alpha-potential trainer."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills.v2.state_symmetry import augment_rallies_with_swap, swap_players


def _state():
    s = np.arange(90, dtype=np.float32) / 100.0
    s[36:42] = [1.2, 0.3, 0.9, 4.0, -1.5, 0.5]     # ball pos / vel (world)
    s[70:72] = [1.0, 0.0]                           # player 1 attacks
    s[72:74] = [1.0, 0.0]                           # acting player 1
    return s


def test_swapping_twice_is_the_identity():
    s = np.random.default_rng(0).normal(size=90).astype(np.float32)
    np.testing.assert_allclose(swap_players(swap_players(s)), s, atol=1e-6)


def test_blocks_swap_and_the_ball_rotates_about_the_table_centre():
    s = _state()
    t = swap_players(s)
    np.testing.assert_array_equal(t[0:18], s[18:36])     # robot 1 <- robot 2 (qpos, qvel)
    np.testing.assert_array_equal(t[18:36], s[0:18])
    np.testing.assert_allclose(t[36:42], [1.8, -0.3, 0.9, -4.0, 1.5, 0.5], atol=1e-6)
    np.testing.assert_array_equal(t[42:56], s[56:70])    # racket targets / errors
    np.testing.assert_array_equal(t[56:70], s[42:56])
    np.testing.assert_array_equal(t[70:72], [0.0, 1.0])  # player 2 attacks after the swap
    np.testing.assert_array_equal(t[72:74], s[72:74])    # acting slot unchanged
    np.testing.assert_array_equal(t[74:79], s[79:84])    # attack choices / theta
    np.testing.assert_array_equal(t[84:87], s[87:90])    # preparation choices / theta


def test_input_is_not_modified():
    s = _state()
    before = s.copy()
    swap_players(s)
    np.testing.assert_array_equal(s, before)


def test_augmentation_adds_a_swapped_copy_of_each_rally():
    rally = {"states": [_state(), _state()], "winner": 1,
             "ego_attack_choices": ["left", "right"], "opp_attack_choices": ["center_safe", "left"],
             "ego_defend_choices": ["center", "left"], "opp_defend_choices": ["right", "right"],
             "ego_theta": [[1, 0, 0, 0, 0], [0, 1, 0]], "opp_theta": [[0, 0, 1, 0, 0], [0, 0, 1]]}
    out = augment_rallies_with_swap([rally])
    assert len(out) == 2 and out[0] is rally
    swapped = out[1]
    assert swapped["winner"] == 2
    assert swapped["ego_attack_choices"] == ["center_safe", "left"]
    assert swapped["opp_defend_choices"] == ["center", "left"]
    assert swapped["ego_theta"] == rally["opp_theta"]
    np.testing.assert_allclose(swapped["states"][0], swap_players(rally["states"][0]))
    assert rally["winner"] == 1  # original untouched


def test_trainer_augments_training_rallies_only(tmp_path):
    from nash_skills.v2 import train_q_model_5skill_v3_attack_defend as tq
    from tests.test_train_q_model_5skill_v3_attack_defend import TestOutDir
    path = tmp_path / "r.pkl"
    TestOutDir()._write_rallies(path, n_rallies=10)
    plain = tq.train(str(path), n_epochs=2, lr=1e-3, out_dir=str(tmp_path / "a"))
    aug = tq.train(str(path), n_epochs=2, lr=1e-3, out_dir=str(tmp_path / "b"), augment_swap=True)
    assert aug["n_train_states"] == 2 * plain["n_train_states"]
    assert aug["n_holdout_states"] == plain["n_holdout_states"]
    assert json.load(open(tmp_path / "b" / "heldout_attack_defend.json"))["augment_swap"] is True


def test_v3_swap_rotates_spin_and_exchanges_the_last_hitter():
    from nash_skills.v2.state_symmetry import swap_players
    s = np.arange(97, dtype=np.float32)
    s[93:95] = [1.0, 0.0]
    out = swap_players(s)
    assert out[90:93].tolist() == [-90.0, -91.0, 92.0]
    assert out[93:95].tolist() == [0.0, 1.0] and out[95] == s[95] and out[96] == s[96]
    np.testing.assert_allclose(swap_players(out), s)
