"""
Unit tests for nash_skills/brute_force_compare.py.

Tests cover:
  - joint_potential: scoring all 25 (s1, s2) pairs for a state
  - brute_force_joint: finding the argmax (s1, s2) pair
  - nash_p_pick: best-response approximation (iterated argmax)
  - compare_on_states: agreement rate computation
  - disagreement_tally: which skills differ most often

Run: python -m pytest tests/test_brute_force_compare.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np


def _import_bfc():
    try:
        import nash_skills.brute_force_compare as m
        return m
    except ModuleNotFoundError:
        return None


# =========================================================================== #
# 1. joint_potential                                                           #
# =========================================================================== #

class TestJointPotential(unittest.TestCase):

    def setUp(self):
        m = _import_bfc()
        if m is None:
            self.skipTest("nash_skills/brute_force_compare.py not yet created")
        self.m = m

    def _make_model(self, fixed_val=0.5):
        """Stub potential model: always returns fixed_val."""
        import torch
        import torch.nn as nn

        class ConstModel(nn.Module):
            def forward(self, x):
                return torch.full((x.shape[0], 1), fixed_val)

        return ConstModel()

    def test_returns_25_scores(self):
        m = self.m
        state = np.zeros(116, dtype=np.float32)
        model = self._make_model(0.5)
        scores = m.joint_potential(state, model)
        self.assertEqual(scores.shape, (5, 5))

    def test_constant_model_all_same(self):
        m = self.m
        state = np.zeros(116, dtype=np.float32)
        model = self._make_model(0.7)
        scores = m.joint_potential(state, model)
        # All 25 cells should be the same value
        self.assertTrue(np.allclose(scores, scores[0, 0]))

    def test_skill_indices_written_to_obs(self):
        """Model that reads obs[-2] and obs[-1] should see normalised indices."""
        import torch
        import torch.nn as nn
        from nash_skills.skills import N_SKILLS

        class ReadObsModel(nn.Module):
            def forward(self, x):
                # Return sum of last two obs dims so we can verify they vary
                return (x[:, -2] + x[:, -1]).unsqueeze(1)

        m = self.m
        state = np.zeros(116, dtype=np.float32)
        scores = m.joint_potential(state, ReadObsModel())
        # Corner (0,0) → obs[-2]=0/4=0, obs[-1]=0/4=0 → sum=0
        self.assertAlmostEqual(float(scores[0, 0]), 0.0, places=4)
        # Corner (4,4) → obs[-2]=1.0, obs[-1]=1.0 → sum=2.0
        self.assertAlmostEqual(float(scores[4, 4]), 2.0, places=4)


# =========================================================================== #
# 2. brute_force_joint                                                         #
# =========================================================================== #

class TestBruteForceJoint(unittest.TestCase):

    def setUp(self):
        m = _import_bfc()
        if m is None:
            self.skipTest("nash_skills/brute_force_compare.py not yet created")
        self.m = m

    def _make_peaked_model(self, peak_s1, peak_s2, peak_val=10.0):
        """Model that returns peak_val only for (peak_s1, peak_s2), else 0."""
        import torch
        import torch.nn as nn
        from nash_skills.skills import N_SKILLS

        class PeakModel(nn.Module):
            def forward(self, x):
                s1_idx = (x[:, -2] * (N_SKILLS - 1)).round().long()
                s2_idx = (x[:, -1] * (N_SKILLS - 1)).round().long()
                out = torch.zeros(x.shape[0], 1)
                mask = (s1_idx == peak_s1) & (s2_idx == peak_s2)
                out[mask] = peak_val
                return out

        return PeakModel()

    def test_finds_global_maximum(self):
        m = self.m
        state = np.zeros(116, dtype=np.float32)
        model = self._make_peaked_model(peak_s1=2, peak_s2=3)
        s1, s2 = m.brute_force_joint(state, model)
        self.assertEqual(s1, 2)
        self.assertEqual(s2, 3)

    def test_returns_tuple_of_ints(self):
        m = self.m
        state = np.zeros(116, dtype=np.float32)
        import torch, torch.nn as nn

        class ConstModel(nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], 1)

        s1, s2 = m.brute_force_joint(state, ConstModel())
        self.assertIsInstance(s1, int)
        self.assertIsInstance(s2, int)


# =========================================================================== #
# 3. nash_p_pick                                                               #
# =========================================================================== #

class TestNashPPick(unittest.TestCase):

    def setUp(self):
        m = _import_bfc()
        if m is None:
            self.skipTest("nash_skills/brute_force_compare.py not yet created")
        self.m = m

    def test_returns_tuple_of_ints(self):
        import torch, torch.nn as nn

        class ConstModel(nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], 1)

        m = self.m
        state = np.zeros(116, dtype=np.float32)
        s1, s2 = m.nash_p_pick(state, ConstModel())
        self.assertIsInstance(s1, int)
        self.assertIsInstance(s2, int)

    def test_valid_skill_range(self):
        from nash_skills.skills import N_SKILLS
        import torch, torch.nn as nn

        class RandModel(nn.Module):
            def forward(self, x):
                return torch.randn(x.shape[0], 1)

        m = self.m
        state = np.zeros(116, dtype=np.float32)
        for _ in range(10):
            s1, s2 = m.nash_p_pick(state, RandModel())
            self.assertGreaterEqual(s1, 0)
            self.assertLess(s1, N_SKILLS)
            self.assertGreaterEqual(s2, 0)
            self.assertLess(s2, N_SKILLS)


# =========================================================================== #
# 4. compare_on_states                                                         #
# =========================================================================== #

class TestCompareOnStates(unittest.TestCase):

    def setUp(self):
        m = _import_bfc()
        if m is None:
            self.skipTest("nash_skills/brute_force_compare.py not yet created")
        self.m = m

    def _make_const_model(self, val=0.0):
        import torch, torch.nn as nn

        class ConstModel(nn.Module):
            def forward(self, x):
                return torch.full((x.shape[0], 1), val)

        return ConstModel()

    def test_returns_dict_with_required_keys(self):
        m = self.m
        states = [np.zeros(116, dtype=np.float32) for _ in range(5)]
        model = self._make_const_model()
        result = m.compare_on_states(states, model)
        self.assertIsInstance(result, dict)
        self.assertIn('agreement_rate', result)
        self.assertIn('n_states', result)
        self.assertIn('n_agree', result)
        self.assertIn('disagreements', result)

    def test_constant_model_full_agreement(self):
        """When model is constant, brute-force and nash-p both pick (0,0)."""
        m = self.m
        states = [np.zeros(116, dtype=np.float32) for _ in range(10)]
        model = self._make_const_model(0.0)
        result = m.compare_on_states(states, model)
        # Both methods tie-break the same way (argmax of all-equal → index 0)
        self.assertAlmostEqual(result['agreement_rate'], 1.0)

    def test_n_states_matches_input(self):
        m = self.m
        states = [np.zeros(116, dtype=np.float32) for _ in range(7)]
        model = self._make_const_model()
        result = m.compare_on_states(states, model)
        self.assertEqual(result['n_states'], 7)

    def test_agreement_rate_in_01(self):
        import torch, torch.nn as nn

        class RandModel(nn.Module):
            def forward(self, x):
                return torch.randn(x.shape[0], 1)

        m = self.m
        np.random.seed(42)
        states = [np.random.randn(116).astype(np.float32) for _ in range(20)]
        result = m.compare_on_states(states, RandModel())
        self.assertGreaterEqual(result['agreement_rate'], 0.0)
        self.assertLessEqual(result['agreement_rate'], 1.0)


# =========================================================================== #
# 5. disagreement_tally                                                        #
# =========================================================================== #

class TestDisagreementTally(unittest.TestCase):

    def setUp(self):
        m = _import_bfc()
        if m is None:
            self.skipTest("nash_skills/brute_force_compare.py not yet created")
        self.m = m

    def test_empty_disagreements_returns_empty(self):
        m = self.m
        tally = m.disagreement_tally([])
        self.assertIsInstance(tally, dict)

    def test_counts_ego_skill_mismatches(self):
        m = self.m
        disagreements = [
            {'bf_s1': 0, 'bf_s2': 1, 'np_s1': 2, 'np_s2': 1},
            {'bf_s1': 0, 'bf_s2': 2, 'np_s1': 2, 'np_s2': 2},
        ]
        tally = m.disagreement_tally(disagreements)
        # ego (s1) disagreement: bf=0, np=2 — both times
        self.assertIn('ego_bf_skill', tally)
        self.assertIn('ego_np_skill', tally)

    def test_most_common_ego_mismatch(self):
        from nash_skills.skills import SKILL_NAMES
        m = self.m
        disagreements = [
            {'bf_s1': 0, 'bf_s2': 0, 'np_s1': 2, 'np_s2': 0},
            {'bf_s1': 0, 'bf_s2': 0, 'np_s1': 2, 'np_s2': 0},
            {'bf_s1': 1, 'bf_s2': 0, 'np_s1': 2, 'np_s2': 0},
        ]
        tally = m.disagreement_tally(disagreements)
        # bf picked skill 0 most often for ego
        self.assertEqual(tally['ego_bf_skill'], SKILL_NAMES[0])
        # np picked skill 2 most often for ego
        self.assertEqual(tally['ego_np_skill'], SKILL_NAMES[2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
