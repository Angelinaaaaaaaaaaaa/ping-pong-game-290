"""
Unit tests for the 2-skill v2 pipeline.

Covers:
  1. collect_data_v2.py — rally dict format, winner field, balance helpers
  2. train_q_model_v2.py — build_dataset_2skill uses discounted returns
  3. Regression: 2-skill state uses {-1, +1} encoding (not {0, 1})
  4. train_q_model_v2.py — potential LR is 0.001, not 0.1

TDD note: tests written before implementation. Run:
    MUJOCO_GL=cgl venv/bin/python -m pytest tests/test_2skill_v2.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np


# =========================================================================== #
# Lazy imports                                                                 #
# =========================================================================== #

def _import_trainer():
    try:
        import train_q_model_v2 as m
        return m
    except ModuleNotFoundError:
        return None


# =========================================================================== #
# 1. build_dataset_2skill uses discounted returns, not sparse {-1,0,+1}       #
# =========================================================================== #

class TestBuildDataset2Skill(unittest.TestCase):
    """build_dataset_2skill(rallies, gamma) must return discounted returns."""

    def setUp(self):
        m = _import_trainer()
        if m is None:
            self.skipTest("train_q_model_v2.py not yet created")
        self.build = m.build_dataset_2skill

    def _make_rally(self, n, winner, skill1=-1.0, skill2=1.0):
        """Make a rally dict with n dummy 116-dim states."""
        states = []
        for _ in range(n):
            s = np.zeros(116, dtype=np.float32)
            s[-2] = skill1
            s[-1] = skill2
            states.append(s)
        return {
            "skill1": "left" if skill1 < 0 else "right",
            "skill2": "left" if skill2 < 0 else "right",
            "states": states,
            "winner": winner,
        }

    def test_returns_three_tensors(self):
        import torch
        rallies = [self._make_rally(3, winner=1)]
        X, Y1, Y2 = self.build(rallies, gamma=0.9)
        self.assertIsInstance(X,  torch.Tensor)
        self.assertIsInstance(Y1, torch.Tensor)
        self.assertIsInstance(Y2, torch.Tensor)

    def test_dataset_length(self):
        rallies = [
            self._make_rally(3, winner=1),
            self._make_rally(2, winner=2),
        ]
        X, Y1, Y2 = self.build(rallies, gamma=0.9)
        self.assertEqual(X.shape[0], 5)   # 3 + 2 states

    def test_ego_wins_terminal_is_plus_one(self):
        rallies = [self._make_rally(4, winner=1)]
        X, Y1, Y2 = self.build(rallies, gamma=0.9)
        self.assertAlmostEqual(Y1[-1].item(), 1.0, places=4)

    def test_opp_wins_terminal_is_minus_one(self):
        rallies = [self._make_rally(4, winner=2)]
        X, Y1, Y2 = self.build(rallies, gamma=0.9)
        self.assertAlmostEqual(Y1[-1].item(), -1.0, places=4)

    def test_discounting_backward(self):
        """For 3-step ego-win rally, Y1 should be [0.81, 0.9, 1.0]."""
        rallies = [self._make_rally(3, winner=1)]
        X, Y1, Y2 = self.build(rallies, gamma=0.9)
        expected = [0.81, 0.9, 1.0]
        for t, exp in enumerate(expected):
            self.assertAlmostEqual(Y1[t].item(), exp, places=4,
                                   msg=f"Y1[{t}] expected {exp}, got {Y1[t].item()}")

    def test_zero_sum_property(self):
        """Y1[t] + Y2[t] == 0 for all t."""
        rallies = [self._make_rally(5, winner=1)]
        X, Y1, Y2 = self.build(rallies, gamma=0.9)
        for t in range(len(Y1)):
            self.assertAlmostEqual(
                Y1[t].item() + Y2[t].item(), 0.0, places=5,
                msg=f"Zero-sum violated at t={t}"
            )

    def test_truncated_all_zero(self):
        """winner=0 (truncated) → all returns are 0."""
        rallies = [self._make_rally(4, winner=0)]
        X, Y1, Y2 = self.build(rallies, gamma=0.9)
        for v in Y1:
            self.assertAlmostEqual(v.item(), 0.0, places=5)

    def test_nonzero_fraction_much_higher_than_sparse(self):
        """
        v2 discounted returns: every state in a completed rally is nonzero.
        Old sparse scheme: only 2 out of L states were nonzero.
        Check: for a 10-state won rally, all 10 Y1 values are nonzero.
        """
        rallies = [self._make_rally(10, winner=1)]
        X, Y1, Y2 = self.build(rallies, gamma=0.9)
        nonzero = sum(1 for v in Y1 if abs(v.item()) > 1e-6)
        self.assertEqual(nonzero, 10,
                         msg=f"Only {nonzero}/10 states have nonzero return. "
                             f"Should be 10 with discounted returns.")


# =========================================================================== #
# 2. Skill encoding regression — 2-skill uses {-1, +1} not {0, 1}            #
# =========================================================================== #

class TestTrainQModel2SkillEncoding(unittest.TestCase):
    """
    Static inspection: train_q_model_v2.py must use {-1, +1} for skill dims,
    not the old {0, 1} that caused sign-flipped potential targets.
    """

    def _read_source(self):
        path = os.path.join(os.path.dirname(__file__), "..", "train_q_model_v2.py")
        if not os.path.exists(path):
            self.skipTest("train_q_model_v2.py not yet created")
        with open(path) as f:
            return f.read()

    def test_x01_ego_uses_minus_one(self):
        """X01[:,-2] must be -1.0 (left ego) not 0.0."""
        import re
        src = self._read_source()
        m = re.search(r'X01\[:,-2\]\s*=\s*([-\d.]+)', src)
        self.assertIsNotNone(m, "X01[:,-2] assignment not found")
        self.assertAlmostEqual(float(m.group(1)), -1.0, places=4)

    def test_x10_opp_uses_minus_one(self):
        """X10[:,-1] must be -1.0 (left opp) not 0.0."""
        import re
        src = self._read_source()
        m = re.search(r'X10\[:,-1\]\s*=\s*([-\d.]+)', src)
        self.assertIsNotNone(m, "X10[:,-1] assignment not found")
        self.assertAlmostEqual(float(m.group(1)), -1.0, places=4)

    def test_x00_both_minus_one(self):
        """X00[:,-2] and X00[:,-1] must both be -1.0."""
        import re
        src = self._read_source()
        m2 = re.search(r'X00\[:,-2\]\s*=\s*([-\d.]+)', src)
        m1 = re.search(r'X00\[:,-1\]\s*=\s*([-\d.]+)', src)
        self.assertIsNotNone(m2)
        self.assertIsNotNone(m1)
        self.assertAlmostEqual(float(m2.group(1)), -1.0, places=4)
        self.assertAlmostEqual(float(m1.group(1)), -1.0, places=4)

    def test_potential_lr_is_0_001(self):
        """optimizer_p must use lr=0.001, not lr=0.1."""
        import re
        src = self._read_source()
        m = re.search(r'optimizer_p\s*=.*?lr\s*=\s*([\d.]+)', src)
        self.assertIsNotNone(m, "optimizer_p lr not found")
        self.assertAlmostEqual(float(m.group(1)), 0.001, places=4)


# =========================================================================== #
# 3. Output model files use _v2 suffix                                         #
# =========================================================================== #

class TestTrainQModel2SkillOutputPaths(unittest.TestCase):
    """
    train_q_model_v2.py must save to model1_v2.pth, model2_v2.pth, model_p_v2.pth.
    This prevents silently overwriting the (possibly-corrected) old .pth files.
    """

    def _read_source(self):
        path = os.path.join(os.path.dirname(__file__), "..", "train_q_model_v2.py")
        if not os.path.exists(path):
            self.skipTest("train_q_model_v2.py not yet created")
        with open(path) as f:
            return f.read()

    def test_saves_model1_v2(self):
        src = self._read_source()
        self.assertIn("model1_v2.pth", src,
                      "Expected output path 'model1_v2.pth' not found")

    def test_saves_model2_v2(self):
        src = self._read_source()
        self.assertIn("model2_v2.pth", src,
                      "Expected output path 'model2_v2.pth' not found")

    def test_saves_model_p_v2(self):
        src = self._read_source()
        self.assertIn("model_p_v2.pth", src,
                      "Expected output path 'model_p_v2.pth' not found")

    def test_does_not_overwrite_model1_pth(self):
        """Must NOT save directly to 'models/model1.pth' (would clobber old weights)."""
        import re
        src = self._read_source()
        # Find torch.save calls — none should target model1.pth without _v2
        saves = re.findall(r'torch\.save\(.*?"([^"]+)"', src)
        for path in saves:
            self.assertFalse(
                path.endswith("model1.pth") or path.endswith("model2.pth")
                or path.endswith("model_p.pth"),
                msg=f"Found torch.save to old path '{path}'. Use _v2 suffix."
            )


# =========================================================================== #
# 4. collect_data_v2 rally dict has 'winner' field                             #
# =========================================================================== #

class TestCollectData2SkillRallyFormat(unittest.TestCase):
    """
    The rally dict format for 2-skill v2 must include 'winner'.
    We test the format by importing detect_winner from labeling and checking
    that it correctly identifies winners from dummy terminal states.
    """

    def setUp(self):
        try:
            from nash_skills.v2.labeling import detect_winner
            self.detect_winner = detect_winner
        except ImportError:
            self.skipTest("nash_skills.v2.labeling not available")

    def _make_state(self, ball_vel_x):
        s = np.zeros(116, dtype=np.float32)
        s[39] = ball_vel_x
        return s

    def test_required_keys_present(self):
        """Rally dict must have skill1, skill2, states, winner keys."""
        rally = {
            "skill1": "left",
            "skill2": "right",
            "states": [self._make_state(2.0)],
            "winner": self.detect_winner([self._make_state(2.0)], done=True),
        }
        for key in ("skill1", "skill2", "states", "winner"):
            self.assertIn(key, rally, f"Missing key '{key}' in rally dict")

    def test_winner_field_is_int(self):
        rally_states = [self._make_state(-2.0)]
        winner = self.detect_winner(rally_states, done=True)
        self.assertIsInstance(winner, int)

    def test_winner_2_when_opp_wins(self):
        """ball_vel_x < 0 → ego missed → opp wins (2)."""
        winner = self.detect_winner([self._make_state(-3.0)], done=True)
        self.assertEqual(winner, 2)

    def test_winner_1_when_ego_wins(self):
        """ball_vel_x > 0 → opp missed → ego wins (1)."""
        winner = self.detect_winner([self._make_state(3.0)], done=True)
        self.assertEqual(winner, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
