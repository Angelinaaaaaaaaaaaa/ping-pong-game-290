"""Tests for nash_skills.v2.joint_action.

TDD: written before implementation.

Shared joint-(attack,defend)-action index/one-hot-write helpers for Design
B, used by BOTH the Phi/Q trainer (train_q_model_5skill_v3_attack_defend.py)
and the decision-rule module (eval_matchup_attack_defend.py) -- a single
shared implementation so the two can never silently diverge on the
joint-index convention or the one-hot-write logic, the exact failure mode
already caught twice this session (eval_matchup.py's _write_skill_pair
bug; the legacy-vs-one-hot apply_skill_pair split).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from nash_skills.v2.joint_action import (
    N_JOINT_ACTIONS,
    joint_action_index,
    joint_action_from_index,
    write_joint_action_pair,
    write_joint_pair_by_index,
)
from nash_skills.v2.state_encoder_attack_defend import STATE_DIM_ATTACK_DEFEND
from nash_skills.skills import N_SKILLS
from nash_skills.defend_positions import N_DEFEND_POSITIONS


class TestConstants:
    def test_n_joint_actions_is_15(self):
        assert N_JOINT_ACTIONS == 15 == N_SKILLS * N_DEFEND_POSITIONS


class TestIndexRoundTrip:
    def test_every_attack_defend_pair_round_trips_through_flat_index(self):
        seen = set()
        for attack_idx in range(N_SKILLS):
            for defend_idx in range(N_DEFEND_POSITIONS):
                flat = joint_action_index(attack_idx, defend_idx)
                assert 0 <= flat < N_JOINT_ACTIONS
                assert flat not in seen  # every pair maps to a distinct index
                seen.add(flat)
                assert joint_action_from_index(flat) == (attack_idx, defend_idx)
        assert seen == set(range(N_JOINT_ACTIONS))

    def test_index_zero_is_first_attack_first_defend(self):
        assert joint_action_index(0, 0) == 0
        assert joint_action_from_index(0) == (0, 0)

    def test_index_max_is_last_attack_last_defend(self):
        assert joint_action_index(N_SKILLS - 1, N_DEFEND_POSITIONS - 1) == N_JOINT_ACTIONS - 1
        assert joint_action_from_index(N_JOINT_ACTIONS - 1) == (N_SKILLS - 1, N_DEFEND_POSITIONS - 1)


def _base_batch(n=3):
    # Distinct, nonzero, deliberately-not-one-hot values everywhere, so
    # "did it touch the wrong dims" or "did it leave stale values" is
    # easy to catch.
    return torch.full((n, STATE_DIM_ATTACK_DEFEND), 7.0, dtype=torch.float32)


class TestWriteJointActionPairBlocksSetCorrectly:
    def test_ego_attack_block_is_set(self):
        x = _base_batch()
        out = write_joint_action_pair(x, ego_attack=2, ego_defend=0, opp_attack=0, opp_defend=0)
        expected = torch.zeros(N_SKILLS)
        expected[2] = 1.0
        for row in out[:, 74:79]:
            torch.testing.assert_close(row, expected)

    def test_opp_attack_block_is_set(self):
        x = _base_batch()
        out = write_joint_action_pair(x, ego_attack=0, ego_defend=0, opp_attack=3, opp_defend=0)
        expected = torch.zeros(N_SKILLS)
        expected[3] = 1.0
        for row in out[:, 79:84]:
            torch.testing.assert_close(row, expected)

    def test_ego_defend_block_is_set(self):
        x = _base_batch()
        out = write_joint_action_pair(x, ego_attack=0, ego_defend=2, opp_attack=0, opp_defend=0)
        expected = torch.zeros(N_DEFEND_POSITIONS)
        expected[2] = 1.0
        for row in out[:, 84:87]:
            torch.testing.assert_close(row, expected)

    def test_opp_defend_block_is_set(self):
        x = _base_batch()
        out = write_joint_action_pair(x, ego_attack=0, ego_defend=0, opp_attack=0, opp_defend=1)
        expected = torch.zeros(N_DEFEND_POSITIONS)
        expected[1] = 1.0
        for row in out[:, 87:90]:
            torch.testing.assert_close(row, expected)

    def test_every_combination_round_trips(self):
        x = _base_batch(n=1)
        for ea in range(N_SKILLS):
            for ed in range(N_DEFEND_POSITIONS):
                for oa in range(N_SKILLS):
                    for od in range(N_DEFEND_POSITIONS):
                        out = write_joint_action_pair(x, ea, ed, oa, od)
                        assert out[0, 74:79].argmax().item() == ea
                        assert out[0, 79:84].argmax().item() == oa
                        assert out[0, 84:87].argmax().item() == ed
                        assert out[0, 87:90].argmax().item() == od
                        assert out[0, 74:79].sum().item() == 1.0
                        assert out[0, 79:84].sum().item() == 1.0
                        assert out[0, 84:87].sum().item() == 1.0
                        assert out[0, 87:90].sum().item() == 1.0


class TestWriteJointActionPairPrefixAndMutation:
    def test_dims_before_74_are_unchanged(self):
        x = _base_batch()
        out = write_joint_action_pair(x, 1, 1, 1, 1)
        torch.testing.assert_close(out[:, :74], x[:, :74])

    def test_original_tensor_is_not_mutated_in_place(self):
        x = _base_batch()
        x_before = x.clone()
        write_joint_action_pair(x, 4, 2, 4, 2)
        torch.testing.assert_close(x, x_before)

    def test_output_shape_matches_input(self):
        x = _base_batch(n=5)
        out = write_joint_action_pair(x, 0, 0, 0, 0)
        assert out.shape == x.shape == (5, STATE_DIM_ATTACK_DEFEND)

    def test_accepts_unbatched_single_row_tensor(self):
        x = torch.full((STATE_DIM_ATTACK_DEFEND,), 7.0, dtype=torch.float32)
        out = write_joint_action_pair(x, 2, 1, 3, 0)
        assert out.shape == (STATE_DIM_ATTACK_DEFEND,)
        assert out[74:79].argmax().item() == 2
        assert out[84:87].argmax().item() == 1


class TestWriteJointPairByIndex:
    """
    The flat-joint-index wrapper both the Phi/Q trainer and the
    decision-rule module need (a 15x15 table's row/column is a flat
    index, not an already-decomposed (attack,defend) pair) -- defined
    once here so neither module reimplements the decomposition and
    risks drifting from the other.
    """

    def test_decomposes_ego_joint_index_correctly(self):
        x = _base_batch()
        ego_joint = joint_action_index(2, 1)
        out = write_joint_pair_by_index(x, ego_joint, opp_joint=0)
        assert out[0, 74:79].argmax().item() == 2
        assert out[0, 84:87].argmax().item() == 1

    def test_decomposes_opp_joint_index_correctly(self):
        x = _base_batch()
        opp_joint = joint_action_index(3, 2)
        out = write_joint_pair_by_index(x, ego_joint=0, opp_joint=opp_joint)
        assert out[0, 79:84].argmax().item() == 3
        assert out[0, 87:90].argmax().item() == 2

    def test_every_joint_index_pair_round_trips(self):
        x = _base_batch(n=1)
        for ego_joint in range(N_JOINT_ACTIONS):
            for opp_joint in range(N_JOINT_ACTIONS):
                out = write_joint_pair_by_index(x, ego_joint, opp_joint)
                assert out[0, 74:79].sum().item() == 1.0
                assert out[0, 79:84].sum().item() == 1.0
                assert out[0, 84:87].sum().item() == 1.0
                assert out[0, 87:90].sum().item() == 1.0

    def test_matches_write_joint_action_pair_given_decomposed_indices(self):
        x = _base_batch()
        ego_joint = joint_action_index(4, 2)
        opp_joint = joint_action_index(1, 0)
        by_index = write_joint_pair_by_index(x, ego_joint, opp_joint)
        direct = write_joint_action_pair(x, ego_attack=4, ego_defend=2, opp_attack=1, opp_defend=0)
        torch.testing.assert_close(by_index, direct)
