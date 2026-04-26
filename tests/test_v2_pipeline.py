"""
Unit tests for the redesigned high-level Nash pipeline (v2).

Covers:
  1. State encoding  — nash_skills/v2/state_encoder.py
  2. Labeling logic  — nash_skills/v2/labeling.py
  3. Data balance    — sanity checks on rally dicts
  4. Nash-p action selection — regression for best_idx=0 left-collapse fix

TDD note: tests are written BEFORE the implementation files exist.
Run:
    MUJOCO_GL=cgl venv/bin/python -m pytest tests/test_v2_pipeline.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import unittest
import numpy as np


# =========================================================================== #
# Helpers — lazy imports so tests can be skipped cleanly if module missing    #
# =========================================================================== #

def _import_state_encoder():
    try:
        import nash_skills.v2.state_encoder as m
        return m
    except ModuleNotFoundError:
        return None


def _import_labeling():
    try:
        import nash_skills.v2.labeling as m
        return m
    except ModuleNotFoundError:
        return None


# =========================================================================== #
# 1. State encoder                                                              #
# =========================================================================== #

class TestStateEncoderDimension(unittest.TestCase):
    """encode_ego / encode_opp must return a 1-D float32 array of STATE_DIM."""

    def setUp(self):
        m = _import_state_encoder()
        if m is None:
            self.skipTest("nash_skills/v2/state_encoder.py not yet created")
        self.enc = m
        # Build a plausible 116-dim raw obs (all zeros is fine for shape tests)
        self.obs = np.zeros(116, dtype=np.float32)
        self.info = {
            "diff_pos":      np.zeros(3,  dtype=np.float32),
            "diff_quat":     np.zeros(4,  dtype=np.float32),
            "target":        np.zeros(7,  dtype=np.float32),
            "diff_pos_opp":  np.zeros(3,  dtype=np.float32),
            "diff_quat_opp": np.zeros(4,  dtype=np.float32),
            "target_opp":    np.zeros(7,  dtype=np.float32),
        }

    def test_encode_ego_returns_1d_float32(self):
        out = self.enc.encode_ego(self.obs, self.info)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.dtype, np.float32)

    def test_encode_opp_returns_1d_float32(self):
        out = self.enc.encode_opp(self.obs, self.info)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.dtype, np.float32)

    def test_ego_and_opp_same_length(self):
        ego = self.enc.encode_ego(self.obs, self.info)
        opp = self.enc.encode_opp(self.obs, self.info)
        self.assertEqual(len(ego), len(opp))

    def test_state_dim_constant_matches_output(self):
        ego = self.enc.encode_ego(self.obs, self.info)
        self.assertEqual(len(ego), self.enc.STATE_DIM)

    def test_state_dim_constant_exported(self):
        self.assertIsInstance(self.enc.STATE_DIM, int)
        self.assertGreater(self.enc.STATE_DIM, 0)


class TestStateEncoderContents(unittest.TestCase):
    """Verify that ball position and skill indices survive round-trip."""

    def setUp(self):
        m = _import_state_encoder()
        if m is None:
            self.skipTest("nash_skills/v2/state_encoder.py not yet created")
        self.enc = m

    def _make_obs(self, ball_pos=(1.5, 0.1, 0.7), skill_ego=0.25, skill_opp=0.75):
        obs = np.zeros(116, dtype=np.float32)
        obs[36:39] = ball_pos          # ball position
        obs[39:42] = [-2.0, 0.0, 1.0] # ball velocity
        obs[-2] = skill_ego
        obs[-1] = skill_opp
        return obs

    def _make_info(self):
        return {
            "diff_pos":      np.array([0.1, -0.05, 0.2], dtype=np.float32),
            "diff_quat":     np.zeros(4, dtype=np.float32),
            "target":        np.ones(7, dtype=np.float32) * 0.5,
            "diff_pos_opp":  np.zeros(3, dtype=np.float32),
            "diff_quat_opp": np.zeros(4, dtype=np.float32),
            "target_opp":    np.zeros(7, dtype=np.float32),
        }

    def test_ego_encodes_ball_position(self):
        """Ball position must appear verbatim somewhere in the ego encoding."""
        obs = self._make_obs(ball_pos=(1.23, 0.45, 0.67))
        enc = self.enc.encode_ego(obs, self._make_info())
        # At least one of the three ball components must be present unchanged
        found = any(
            np.any(np.abs(enc - v) < 1e-4)
            for v in [1.23, 0.45, 0.67]
        )
        self.assertTrue(found, "Ball position components not found in ego encoding")

    def test_skill_indices_present_in_ego(self):
        """Skill encoding (ego=0.25) must appear in the ego state vector."""
        obs = self._make_obs(skill_ego=0.25, skill_opp=0.75)
        enc = self.enc.encode_ego(obs, self._make_info())
        self.assertTrue(
            np.any(np.abs(enc - 0.25) < 1e-4),
            "Ego skill index 0.25 not found in encoding"
        )

    def test_different_skills_produce_different_encodings(self):
        obs_left  = self._make_obs(skill_ego=0.0)
        obs_right = self._make_obs(skill_ego=1.0)
        info = self._make_info()
        enc_left  = self.enc.encode_ego(obs_left,  info)
        enc_right = self.enc.encode_ego(obs_right, info)
        self.assertFalse(
            np.allclose(enc_left, enc_right),
            "Different ego skills produced identical state encodings"
        )

    def test_gantry_position_in_ego(self):
        """qpos[0] and qpos[1] are gantry positions — must be in ego encoding."""
        obs = np.zeros(116, dtype=np.float32)
        obs[0] = -0.8   # gantry x
        obs[1] =  0.5   # gantry y
        info = {k: np.zeros(s, dtype=np.float32)
                for k, s in [("diff_pos", 3), ("diff_quat", 4),
                              ("target", 7), ("diff_pos_opp", 3),
                              ("diff_quat_opp", 4), ("target_opp", 7)]}
        enc = self.enc.encode_ego(obs, info)
        found_x = np.any(np.abs(enc - (-0.8)) < 1e-4)
        found_y = np.any(np.abs(enc -  0.5)  < 1e-4)
        self.assertTrue(found_x or found_y,
                        "Gantry position (qpos[0]/qpos[1]) not found in ego encoding")


# =========================================================================== #
# 2. Labeling — discounted returns                                              #
# =========================================================================== #

class TestLabelingDiscountedReturn(unittest.TestCase):
    """
    compute_returns(rally, gamma, winner) should assign discounted returns.

    Convention:
      winner = 1  → ego wins this rally   → last crossing reward = +1
      winner = 2  → opp wins this rally   → last crossing reward = -1
      winner = 0  → truncated (no clear winner) → terminal reward = 0

    Returns are computed backward: G[t] = r[t] + gamma * G[t+1]
    where r[t] = 0 for all but the terminal step.
    """

    def setUp(self):
        m = _import_labeling()
        if m is None:
            self.skipTest("nash_skills/v2/labeling.py not yet created")
        self.label = m

    def _make_rally(self, n_crossings):
        """Fake rally with n_crossings dummy states (each state is a 116-dim zero vec)."""
        return [np.zeros(116, dtype=np.float32) for _ in range(n_crossings)]

    def test_returns_length_matches_rally(self):
        rally = self._make_rally(5)
        g1, g2 = self.label.compute_returns(rally, gamma=0.9, winner=1)
        self.assertEqual(len(g1), 5)
        self.assertEqual(len(g2), 5)

    def test_ego_wins_terminal_is_plus_one(self):
        rally = self._make_rally(4)
        g1, g2 = self.label.compute_returns(rally, gamma=0.9, winner=1)
        self.assertAlmostEqual(g1[-1], 1.0, places=5,
                               msg="ego wins → G1[last] should be +1")

    def test_opp_wins_terminal_is_minus_one(self):
        rally = self._make_rally(4)
        g1, g2 = self.label.compute_returns(rally, gamma=0.9, winner=2)
        self.assertAlmostEqual(g1[-1], -1.0, places=5,
                               msg="opp wins → G1[last] should be -1")

    def test_truncated_terminal_is_zero(self):
        rally = self._make_rally(3)
        g1, g2 = self.label.compute_returns(rally, gamma=0.9, winner=0)
        self.assertAlmostEqual(g1[-1], 0.0, places=5,
                               msg="truncated → terminal G1 should be 0")

    def test_discounting_applied_backward(self):
        """For a 3-crossing rally where ego wins, G1 should be [gamma^2, gamma, 1]."""
        rally = self._make_rally(3)
        gamma = 0.9
        g1, _ = self.label.compute_returns(rally, gamma=gamma, winner=1)
        expected = [gamma**2, gamma**1, 1.0]
        for t, (got, exp) in enumerate(zip(g1, expected)):
            self.assertAlmostEqual(
                got, exp, places=5,
                msg=f"G1[{t}]: expected {exp:.4f}, got {got:.4f}"
            )

    def test_opp_returns_are_negated_ego_returns(self):
        """Player2 returns = -player1 returns (zero-sum game)."""
        rally = self._make_rally(5)
        g1, g2 = self.label.compute_returns(rally, gamma=0.9, winner=1)
        for t, (v1, v2) in enumerate(zip(g1, g2)):
            self.assertAlmostEqual(
                v1 + v2, 0.0, places=5,
                msg=f"Zero-sum violated at t={t}: G1={v1:.4f}, G2={v2:.4f}"
            )

    def test_single_crossing_rally(self):
        """A rally with exactly one crossing still works."""
        rally = self._make_rally(1)
        g1, g2 = self.label.compute_returns(rally, gamma=0.9, winner=1)
        self.assertEqual(len(g1), 1)
        self.assertAlmostEqual(g1[0], 1.0, places=5)

    def test_returns_are_bounded(self):
        """All returns must be in [-1, 1] since terminal reward is ±1."""
        rally = self._make_rally(20)
        g1, g2 = self.label.compute_returns(rally, gamma=0.9, winner=2)
        for v in g1:
            self.assertGreaterEqual(v, -1.0 - 1e-6)
            self.assertLessEqual(v,    1.0 + 1e-6)


# =========================================================================== #
# 3. Winner detection from rally terminal state                                 #
# =========================================================================== #

class TestWinnerDetection(unittest.TestCase):
    """
    detect_winner(rally, done) must correctly infer which player won.

    The convention used in KukaTennisEnv:
      done=True when ball exits past a racket by >0.3m
      The LAST state in the rally stores ball_vel[0]:
        ball_vel[0] > 0  → ball moving toward opp side → opp missed → ego wins
        ball_vel[0] < 0  → ball moving toward ego side → ego missed → opp wins
    """

    def setUp(self):
        m = _import_labeling()
        if m is None:
            self.skipTest("nash_skills/v2/labeling.py not yet created")
        self.label = m

    def _make_state(self, ball_vel_x):
        s = np.zeros(116, dtype=np.float32)
        s[39] = ball_vel_x   # ball_vel[0]
        return s

    def test_ball_moving_right_ego_wins(self):
        """ball_vel[0] > 0 → ball going to opp side → opp missed → ego wins (1)."""
        rally = [self._make_state(3.0)]
        winner = self.label.detect_winner(rally, done=True)
        self.assertEqual(winner, 1)

    def test_ball_moving_left_opp_wins(self):
        """ball_vel[0] < 0 → ball going to ego side → ego missed → opp wins (2)."""
        rally = [self._make_state(-3.0)]
        winner = self.label.detect_winner(rally, done=True)
        self.assertEqual(winner, 2)

    def test_truncated_returns_zero(self):
        """done=False (truncated episode) → no winner → return 0."""
        rally = [self._make_state(2.0)]
        winner = self.label.detect_winner(rally, done=False)
        self.assertEqual(winner, 0)

    def test_empty_rally_truncated(self):
        """Empty rally (no crossings recorded) → no winner."""
        winner = self.label.detect_winner([], done=True)
        self.assertEqual(winner, 0)


# =========================================================================== #
# 4. Data balance sanity checks                                                 #
# =========================================================================== #

class TestDataBalanceSummary(unittest.TestCase):
    """
    summarise_balance(rallies) must return per-pair counts and flag imbalance.
    """

    def setUp(self):
        m = _import_labeling()
        if m is None:
            self.skipTest("nash_skills/v2/labeling.py not yet created")
        self.label = m

    def _make_rallies(self, pairs_and_counts):
        """
        pairs_and_counts: list of ((skill1, skill2), count) tuples.
        Returns a list of rally dicts.
        """
        rallies = []
        for (s1, s2), count in pairs_and_counts:
            for _ in range(count):
                rallies.append({
                    "skill1": s1,
                    "skill2": s2,
                    "states": [np.zeros(116, dtype=np.float32)],
                    "winner": 1,
                })
        return rallies

    def test_summary_returns_dict(self):
        rallies = self._make_rallies([(("left", "right"), 5)])
        summary = self.label.summarise_balance(rallies)
        self.assertIsInstance(summary, dict)

    def test_summary_counts_correct(self):
        rallies = self._make_rallies([
            (("left",  "left"),  10),
            (("left",  "right"),  5),
            (("right", "left"),   8),
        ])
        summary = self.label.summarise_balance(rallies)
        self.assertEqual(summary[("left",  "left")],  10)
        self.assertEqual(summary[("left",  "right")],  5)
        self.assertEqual(summary[("right", "left")],   8)

    def test_summary_missing_pair_is_zero(self):
        rallies = self._make_rallies([(("left", "right"), 3)])
        summary = self.label.summarise_balance(rallies)
        # A pair not in the data should return 0 or be absent
        count = summary.get(("right", "left"), 0)
        self.assertEqual(count, 0)

    def test_imbalance_ratio_detected(self):
        """max_count / min_count > 10 → imbalanced flag."""
        rallies = self._make_rallies([
            (("left",  "left"),  100),
            (("right", "right"),   1),
        ])
        is_balanced, ratio = self.label.check_balance(rallies, threshold=10.0)
        self.assertFalse(is_balanced)
        self.assertGreater(ratio, 10.0)

    def test_balanced_dataset_passes(self):
        rallies = self._make_rallies([
            (("left",  "left"),  10),
            (("left",  "right"), 12),
            (("right", "left"),   9),
            (("right", "right"), 11),
        ])
        is_balanced, ratio = self.label.check_balance(rallies, threshold=10.0)
        self.assertTrue(is_balanced)
        self.assertLessEqual(ratio, 10.0)


# =========================================================================== #
# 5. Regression: nash-p argmax never defaults to index 0 when last is best    #
# =========================================================================== #

class TestNashPNoLeftBiasV2(unittest.TestCase):
    """
    The v2 eval pipeline must also use collect-all-then-argmax, not sequential
    scan with best_idx=0 initialization.

    Verified by injecting a mock model that prefers the last skill.
    """

    def setUp(self):
        try:
            import torch
            import torch.nn as nn
            from nash_skills.skills import N_SKILLS
            from nash_skills.eval_matchup import make_picker
        except ImportError:
            self.skipTest("torch, nash_skills.skills, or eval_matchup not available")
        self.torch = torch
        self.nn = nn
        self.N_SKILLS = N_SKILLS
        self.make_picker = make_picker

    def _make_last_preferred_model(self):
        torch = self.torch
        nn = self.nn

        class LastPreferred(nn.Module):
            def forward(self, x):
                # Returns 1.0 when ego skill index (obs[-2]) == 1.0, else 0.0
                return (x[:, -2] == 1.0).float().unsqueeze(1)

        return LastPreferred()

    def test_picks_last_not_first_when_last_best(self):
        model = self._make_last_preferred_model()
        pick = self.make_picker("nash-p", model)
        obs = self.torch.zeros(116).numpy()
        chosen = pick(1, obs, 0)   # player=1, other_skill=0 (left)
        self.assertEqual(
            chosen, self.N_SKILLS - 1,
            msg=f"Expected last skill ({self.N_SKILLS-1}=right), got {chosen}. "
                f"Indicates best_idx=0 left-collapse bug."
        )


# =========================================================================== #
# 6. compute_returns: gamma=0.9 default is honoured                            #
# =========================================================================== #

class TestLabelingDefaultGamma(unittest.TestCase):

    def setUp(self):
        m = _import_labeling()
        if m is None:
            self.skipTest("nash_skills/v2/labeling.py not yet created")
        self.label = m

    def test_default_gamma_is_0_9(self):
        """GAMMA constant in labeling module must be 0.9."""
        self.assertAlmostEqual(self.label.GAMMA, 0.9, places=5)

    def test_compute_returns_uses_gamma_arg(self):
        """gamma=0.5 should give G[0] = 0.5**2 = 0.25 for 3-step rally."""
        rally = [np.zeros(116, dtype=np.float32) for _ in range(3)]
        g1, _ = self.label.compute_returns(rally, gamma=0.5, winner=1)
        self.assertAlmostEqual(g1[0], 0.25, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
