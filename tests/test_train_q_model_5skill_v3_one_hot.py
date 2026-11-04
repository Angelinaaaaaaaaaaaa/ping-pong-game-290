"""Tests for nash_skills.v2.train_q_model_5skill_v3_one_hot.apply_skill_pair.

TDD: written before implementation.

The legacy trainer's apply_skill_pair (train_q_model_5skill_v3.py) writes
a single scalar to x[:, -2] / x[:, -1] -- correct for the 76-dim scalar
skill encoding, but fundamentally wrong for the 84-dim one-hot encoding
(nash_skills.v2.state_encoder_one_hot), where skill identity is a 5-dim
one-hot block at [74:79] (ego) / [79:84] (opp), not a single scalar.
This is exactly the kind of silent, numerically-valid-looking-but-wrong
bug that would corrupt the potential model's counterfactual-constraint
training (apply_skill_pair is used to build "same state, different
skill" pairs for every (i, j, i2, j2) constraint) without raising any
error -- so this is tested directly, not just exercised indirectly
through a full training run.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from nash_skills.v2.train_q_model_5skill_v3_one_hot import apply_skill_pair
from nash_skills.v2.state_encoder_one_hot import STATE_DIM_ONE_HOT
from nash_skills.skills import N_SKILLS


def _base_batch(n=3):
    # Distinct, nonzero values everywhere so "did it touch the wrong
    # dims" is easy to catch -- deliberately NOT a valid one-hot block
    # to begin with, so the test proves apply_skill_pair fully overwrites
    # it rather than merely nudging it.
    return torch.full((n, STATE_DIM_ONE_HOT), 7.0, dtype=torch.float32)


class TestOneHotBlocksSetCorrectly:
    def test_ego_one_hot_block_is_set(self):
        x = _base_batch()
        out = apply_skill_pair(x, ego_skill=2, opp_skill=0)
        expected = torch.zeros(N_SKILLS)
        expected[2] = 1.0
        for row in out[:, 74:79]:
            torch.testing.assert_close(row, expected)

    def test_opp_one_hot_block_is_set(self):
        x = _base_batch()
        out = apply_skill_pair(x, ego_skill=0, opp_skill=3)
        expected = torch.zeros(N_SKILLS)
        expected[3] = 1.0
        for row in out[:, 79:84]:
            torch.testing.assert_close(row, expected)

    def test_every_skill_index_pair_round_trips(self):
        x = _base_batch(n=1)
        for i in range(N_SKILLS):
            for j in range(N_SKILLS):
                out = apply_skill_pair(x, ego_skill=i, opp_skill=j)
                assert out[0, 74:79].argmax().item() == i
                assert out[0, 79:84].argmax().item() == j
                assert out[0, 74:79].sum().item() == 1.0
                assert out[0, 79:84].sum().item() == 1.0


class TestPrefixUntouched:
    def test_dims_before_74_are_unchanged(self):
        x = _base_batch()
        out = apply_skill_pair(x, ego_skill=1, opp_skill=1)
        torch.testing.assert_close(out[:, :74], x[:, :74])

    def test_original_tensor_is_not_mutated_in_place(self):
        x = _base_batch()
        x_before = x.clone()
        apply_skill_pair(x, ego_skill=4, opp_skill=4)
        torch.testing.assert_close(x, x_before)


class TestShape:
    def test_output_shape_matches_input(self):
        x = _base_batch(n=5)
        out = apply_skill_pair(x, ego_skill=0, opp_skill=0)
        assert out.shape == x.shape == (5, STATE_DIM_ONE_HOT)
