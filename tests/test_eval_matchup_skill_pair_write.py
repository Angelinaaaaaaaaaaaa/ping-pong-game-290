"""
Unit tests for nash_skills/eval_matchup.py's _write_skill_pair.

The bug this fixes: every nash-p-*/ibr/ibr-q decision rule builds
counterfactual states by writing `x[-2] = ego_s/(N-1)` and
`x[-1] = opp_s/(N-1)` directly -- correct for the legacy 76-dim encoded
state and the raw 116-dim observation (both end with a scalar skill
index at those two positions), but wrong for the one-hot 84-dim encoded
state (nash_skills.v2.state_encoder_one_hot), where [-2]/[-1] are just
2 of a player's 5-dim one-hot block. Writing a fraction into 2 of 5
one-hot slots while leaving the other 3 untouched produces a state that
is neither valid one-hot nor valid scalar -- corrupting every
counterfactual query nash-p-hard/br/minimax/adaptive/ibr/ibr-q rely on
for a one-hot model.

_write_skill_pair centralizes this write behind one function that
auto-detects which convention applies from the tensor's own last
dimension, defaulting to the ORIGINAL scalar behavior for anything that
isn't exactly 84-dim (safe for the existing 76-dim and 116-dim cases,
which must not change) and only switching to the one-hot-correct
zero-then-set-one-index behavior for 84.

Run: python -m pytest tests/test_eval_matchup_skill_pair_write.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest


def _import_eval():
    try:
        import nash_skills.eval_matchup as m
        return m
    except ModuleNotFoundError:
        return None


class TestWriteSkillPairLegacyScalar(unittest.TestCase):
    """76-dim and 116-dim: must preserve the ORIGINAL scalar behavior
    exactly -- these are the dimensions every existing test and every
    existing trained checkpoint already depends on."""

    def setUp(self):
        m = _import_eval()
        if m is None:
            self.skipTest("nash_skills/eval_matchup.py not yet created")
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        self.torch = torch
        self.write = m._write_skill_pair

    def test_76_dim_writes_scalar_at_last_two_positions(self):
        torch = self.torch
        x = torch.zeros(76)
        out = self.write(x, ego_skill_idx=3, opp_skill_idx=1)
        self.assertAlmostEqual(out[-2].item(), 3 / 4)
        self.assertAlmostEqual(out[-1].item(), 1 / 4)

    def test_116_dim_raw_obs_writes_scalar_at_last_two_positions(self):
        torch = self.torch
        x = torch.zeros(116)
        out = self.write(x, ego_skill_idx=0, opp_skill_idx=4)
        self.assertAlmostEqual(out[-2].item(), 0.0)
        self.assertAlmostEqual(out[-1].item(), 1.0)

    def test_76_dim_leaves_every_other_position_untouched(self):
        torch = self.torch
        x = torch.arange(76, dtype=torch.float32)
        out = self.write(x, ego_skill_idx=2, opp_skill_idx=2)
        self.torch.testing.assert_close(out[:-2], x[:-2])

    def test_does_not_mutate_the_input_tensor(self):
        torch = self.torch
        x = torch.zeros(76)
        self.write(x, ego_skill_idx=4, opp_skill_idx=4)
        self.assertTrue(torch.all(x == 0.0))

    def test_batched_76_dim_writes_per_row(self):
        torch = self.torch
        x = torch.zeros(3, 76)
        out = self.write(x, ego_skill_idx=4, opp_skill_idx=0)
        for row in range(3):
            self.assertAlmostEqual(out[row, -2].item(), 1.0)
            self.assertAlmostEqual(out[row, -1].item(), 0.0)


class TestWriteSkillPairOneHot(unittest.TestCase):
    """84-dim: must zero each player's full 5-dim one-hot block and set
    exactly one index -- the actual bug fix."""

    def setUp(self):
        m = _import_eval()
        if m is None:
            self.skipTest("nash_skills/eval_matchup.py not yet created")
        try:
            import torch
            from nash_skills.v2.state_encoder_one_hot import STATE_DIM_ONE_HOT
        except ImportError:
            self.skipTest("torch not available")
        self.torch = torch
        self.write = m._write_skill_pair
        self.dim = STATE_DIM_ONE_HOT

    def test_84_dim_sets_a_clean_one_hot_block_for_each_player(self):
        torch = self.torch
        x = torch.full((self.dim,), 7.0)  # deliberately not-already-one-hot
        out = self.write(x, ego_skill_idx=2, opp_skill_idx=4)
        ego_block = out[-10:-5]
        opp_block = out[-5:]
        expected_ego = torch.zeros(5)
        expected_ego[2] = 1.0
        expected_opp = torch.zeros(5)
        expected_opp[4] = 1.0
        torch.testing.assert_close(ego_block, expected_ego)
        torch.testing.assert_close(opp_block, expected_opp)

    def test_84_dim_every_skill_pair_round_trips(self):
        torch = self.torch
        for ego_s in range(5):
            for opp_s in range(5):
                x = torch.zeros(self.dim)
                out = self.write(x, ego_skill_idx=ego_s, opp_skill_idx=opp_s)
                self.assertEqual(int(out[-10:-5].argmax()), ego_s)
                self.assertEqual(int(out[-5:].argmax()), opp_s)
                self.assertAlmostEqual(out[-10:-5].sum().item(), 1.0)
                self.assertAlmostEqual(out[-5:].sum().item(), 1.0)

    def test_84_dim_leaves_dims_before_74_untouched(self):
        torch = self.torch
        x = torch.arange(self.dim, dtype=torch.float32)
        out = self.write(x, ego_skill_idx=1, opp_skill_idx=1)
        torch.testing.assert_close(out[:74], x[:74])

    def test_84_dim_does_not_mutate_the_input_tensor(self):
        torch = self.torch
        x = torch.full((self.dim,), 7.0)
        self.write(x, ego_skill_idx=0, opp_skill_idx=0)
        self.assertTrue(torch.all(x == 7.0))

    def test_batched_84_dim_writes_per_row(self):
        torch = self.torch
        x = torch.zeros(2, self.dim)
        out = self.write(x, ego_skill_idx=3, opp_skill_idx=1)
        for row in range(2):
            self.assertEqual(int(out[row, -10:-5].argmax()), 3)
            self.assertEqual(int(out[row, -5:].argmax()), 1)


class TestWriteSkillPairUnknownDimension(unittest.TestCase):
    def setUp(self):
        m = _import_eval()
        if m is None:
            self.skipTest("nash_skills/eval_matchup.py not yet created")
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        self.torch = torch
        self.write = m._write_skill_pair

    def test_unexpected_dimension_falls_back_to_scalar_not_crash(self):
        # Defaulting to scalar for "anything but 84" is the deliberately
        # safe choice -- it must never raise just because some other
        # caller passes an unfamiliar length, matching the ORIGINAL
        # code's unconditional behavior before this fix existed.
        torch = self.torch
        x = torch.zeros(12)
        out = self.write(x, ego_skill_idx=1, opp_skill_idx=1)
        self.assertAlmostEqual(out[-2].item(), 1 / 4)
        self.assertAlmostEqual(out[-1].item(), 1 / 4)


if __name__ == "__main__":
    unittest.main()
