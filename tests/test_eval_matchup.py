"""
Unit tests for nash_skills/eval_matchup.py.

Tests cover:
  - MatchupResult dataclass correctness
  - win-rate computation
  - rally-length computation
  - CSV output format
  - summary table generation
  - strategy validation

Run: python -m pytest tests/test_eval_matchup.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import io
import unittest
import tempfile


# =========================================================================== #
# Helper: import the module under test (skip if not yet created)              #
# =========================================================================== #

def _import_eval():
    try:
        import nash_skills.eval_matchup as m
        return m
    except ModuleNotFoundError:
        return None


# =========================================================================== #
# 1. MatchupResult dataclass                                                   #
# =========================================================================== #

class TestMatchupResult(unittest.TestCase):

    def setUp(self):
        m = _import_eval()
        if m is None:
            self.skipTest("nash_skills/eval_matchup.py not yet created")
        self.MatchupResult = m.MatchupResult

    def test_win_rate_zero_episodes(self):
        r = self.MatchupResult(
            strategy1='nash-p', strategy2='random',
            episodes=0, ego_wins=0, opp_wins=0,
            ego_contacts=0, opp_contacts=0,
            ego_successes=0, opp_successes=0,
            rally_lengths=[],
        )
        self.assertIsNone(r.win_rate)

    def test_win_rate_all_wins(self):
        r = self.MatchupResult(
            strategy1='nash-p', strategy2='random',
            episodes=10, ego_wins=10, opp_wins=0,
            ego_contacts=20, opp_contacts=15,
            ego_successes=18, opp_successes=12,
            rally_lengths=[3, 4, 5],
        )
        self.assertAlmostEqual(r.win_rate, 1.0)

    def test_win_rate_half(self):
        r = self.MatchupResult(
            strategy1='nash-p', strategy2='random',
            episodes=10, ego_wins=5, opp_wins=5,
            ego_contacts=20, opp_contacts=20,
            ego_successes=15, opp_successes=15,
            rally_lengths=[2, 3, 2],
        )
        self.assertAlmostEqual(r.win_rate, 0.5)

    def test_avg_rally_length_empty(self):
        r = self.MatchupResult(
            strategy1='nash-p', strategy2='left',
            episodes=0, ego_wins=0, opp_wins=0,
            ego_contacts=0, opp_contacts=0,
            ego_successes=0, opp_successes=0,
            rally_lengths=[],
        )
        self.assertIsNone(r.avg_rally_length)

    def test_avg_rally_length_computed(self):
        r = self.MatchupResult(
            strategy1='nash-p', strategy2='left',
            episodes=3, ego_wins=2, opp_wins=1,
            ego_contacts=6, opp_contacts=4,
            ego_successes=5, opp_successes=3,
            rally_lengths=[2, 4, 6],
        )
        self.assertAlmostEqual(r.avg_rally_length, 4.0)

    def test_ego_success_rate_zero_contacts(self):
        r = self.MatchupResult(
            strategy1='nash-p', strategy2='left',
            episodes=5, ego_wins=0, opp_wins=5,
            ego_contacts=0, opp_contacts=10,
            ego_successes=0, opp_successes=8,
            rally_lengths=[1],
        )
        self.assertIsNone(r.ego_success_rate)

    def test_ego_success_rate_computed(self):
        r = self.MatchupResult(
            strategy1='nash-p', strategy2='left',
            episodes=5, ego_wins=3, opp_wins=2,
            ego_contacts=10, opp_contacts=8,
            ego_successes=8, opp_successes=6,
            rally_lengths=[2, 3],
        )
        self.assertAlmostEqual(r.ego_success_rate, 0.8)


# =========================================================================== #
# 2. CSV output                                                                #
# =========================================================================== #

class TestCsvOutput(unittest.TestCase):

    def setUp(self):
        m = _import_eval()
        if m is None:
            self.skipTest("nash_skills/eval_matchup.py not yet created")
        self.MatchupResult = m.MatchupResult
        self.save_csv = m.save_csv

    def _make_result(self, s1='nash-p', s2='random', wins=5, total=10):
        return self.MatchupResult(
            strategy1=s1, strategy2=s2,
            episodes=total, ego_wins=wins, opp_wins=total - wins,
            ego_contacts=20, opp_contacts=15,
            ego_successes=18, opp_successes=12,
            rally_lengths=[2, 3, 4, 5],
        )

    def test_csv_has_header(self):
        results = [self._make_result()]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        self.save_csv(results, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            self.assertIn('strategy1',      reader.fieldnames)
            self.assertIn('strategy2',      reader.fieldnames)
            self.assertIn('win_rate',       reader.fieldnames)
            self.assertIn('ego_contacts',   reader.fieldnames)
            self.assertIn('avg_rally_length', reader.fieldnames)
        os.unlink(path)

    def test_csv_row_count(self):
        results = [self._make_result('nash-p', 'random'),
                   self._make_result('nash-p', 'left'),
                   self._make_result('nash-p', 'right')]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        self.save_csv(results, path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)
        os.unlink(path)

    def test_csv_win_rate_value(self):
        results = [self._make_result(wins=7, total=10)]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        self.save_csv(results, path)
        with open(path) as f:
            row = list(csv.DictReader(f))[0]
        self.assertAlmostEqual(float(row['win_rate']), 0.7, places=3)
        os.unlink(path)


# =========================================================================== #
# 3. Summary table                                                             #
# =========================================================================== #

class TestSummaryTable(unittest.TestCase):

    def setUp(self):
        m = _import_eval()
        if m is None:
            self.skipTest("nash_skills/eval_matchup.py not yet created")
        self.MatchupResult = m.MatchupResult
        self.print_summary = m.print_summary

    def _make_result(self, s2, wins, total):
        return self.MatchupResult(
            strategy1='nash-p', strategy2=s2,
            episodes=total, ego_wins=wins, opp_wins=total - wins,
            ego_contacts=20, opp_contacts=15,
            ego_successes=18, opp_successes=12,
            rally_lengths=[3, 4, 5],
        )

    def test_summary_prints_all_matchups(self):
        results = [
            self._make_result('random', 6, 10),
            self._make_result('left',   7, 10),
            self._make_result('right',  5, 10),
        ]
        buf = io.StringIO()
        self.print_summary(results, file=buf)
        output = buf.getvalue()
        self.assertIn('random', output)
        self.assertIn('left',   output)
        self.assertIn('right',  output)

    def test_summary_contains_win_rate(self):
        results = [self._make_result('random', 8, 10)]
        buf = io.StringIO()
        self.print_summary(results, file=buf)
        output = buf.getvalue()
        # 80% win rate should appear
        self.assertIn('80', output)

    def test_summary_has_header_line(self):
        results = [self._make_result('random', 5, 10)]
        buf = io.StringIO()
        self.print_summary(results, file=buf)
        output = buf.getvalue()
        # Should have a separator or header with "strategy" or "win"
        self.assertTrue(
            'strategy' in output.lower() or 'win' in output.lower(),
            f"Summary table header not found in: {output[:200]}"
        )


# =========================================================================== #
# 4. Strategy validation                                                        #
# =========================================================================== #

# =========================================================================== #
# 5. Skill usage tracking                                                      #
# =========================================================================== #

class TestSkillUsage(unittest.TestCase):

    def setUp(self):
        m = _import_eval()
        if m is None:
            self.skipTest("nash_skills/eval_matchup.py not yet created")
        self.MatchupResult = m.MatchupResult

    def _make_result(self, skill_usage=None):
        return self.MatchupResult(
            strategy1='nash-p', strategy2='random',
            episodes=10, ego_wins=6, opp_wins=4,
            ego_contacts=20, opp_contacts=15,
            ego_successes=18, opp_successes=12,
            rally_lengths=[3, 4, 5],
            skill_usage=skill_usage or {},
        )

    def test_skill_usage_field_exists(self):
        r = self._make_result({'left': 5, 'right': 3})
        self.assertIsNotNone(r.skill_usage)

    def test_skill_usage_empty_default(self):
        r = self._make_result({})
        self.assertEqual(r.skill_usage, {})

    def test_skill_usage_most_used(self):
        m = _import_eval()
        if m is None:
            self.skipTest("nash_skills/eval_matchup.py not yet created")
        r = self._make_result({'left': 10, 'right': 3, 'center_safe': 1})
        most = m.most_used_skill(r)
        self.assertEqual(most, 'left')

    def test_skill_usage_most_used_empty(self):
        m = _import_eval()
        if m is None:
            self.skipTest("nash_skills/eval_matchup.py not yet created")
        r = self._make_result({})
        most = m.most_used_skill(r)
        self.assertIsNone(most)

    def test_skill_usage_dominance(self):
        """dominant_skill_fraction = max_count / total_picks."""
        m = _import_eval()
        if m is None:
            self.skipTest("nash_skills/eval_matchup.py not yet created")
        r = self._make_result({'left': 8, 'right': 2})
        frac = m.dominant_skill_fraction(r)
        self.assertAlmostEqual(frac, 0.8)

    def test_skill_usage_dominance_empty(self):
        m = _import_eval()
        if m is None:
            self.skipTest("nash_skills/eval_matchup.py not yet created")
        r = self._make_result({})
        frac = m.dominant_skill_fraction(r)
        self.assertIsNone(frac)


# =========================================================================== #
# 6. Analysis functions                                                         #
# =========================================================================== #

class TestAnalysis(unittest.TestCase):

    def setUp(self):
        m = _import_eval()
        if m is None:
            self.skipTest("nash_skills/eval_matchup.py not yet created")
        self.MatchupResult = m.MatchupResult
        self.analyse_results = m.analyse_results

    def _make(self, s2, wins, total, rally_lengths=None, skill_usage=None):
        return self.MatchupResult(
            strategy1='nash-p', strategy2=s2,
            episodes=total, ego_wins=wins, opp_wins=total - wins,
            ego_contacts=max(1, wins), opp_contacts=max(1, total - wins),
            ego_successes=wins, opp_successes=total - wins,
            rally_lengths=rally_lengths or [5] * total,
            skill_usage=skill_usage or {},
        )

    def test_analyse_returns_dict(self):
        results = [self._make('random', 5, 10)]
        analysis = self.analyse_results(results)
        self.assertIsInstance(analysis, dict)

    def test_analyse_has_center_safe_flag(self):
        results = [
            self._make('center_safe', 0, 0, rally_lengths=[200, 300]),
            self._make('left', 5, 10, rally_lengths=[20, 30]),
        ]
        analysis = self.analyse_results(results)
        self.assertIn('center_safe_long_rallies', analysis)

    def test_centre_safe_long_rally_detected(self):
        """center_safe matchup with very long rallies should flag the issue."""
        results = [
            self._make('center_safe', 0, 2, rally_lengths=[150, 200]),
            self._make('left',        5, 10, rally_lengths=[15, 20]),
        ]
        analysis = self.analyse_results(results)
        self.assertTrue(analysis['center_safe_long_rallies'])

    def test_centre_safe_normal_rally_no_flag(self):
        results = [
            self._make('center_safe', 5, 10, rally_lengths=[10, 12]),
            self._make('left',        5, 10, rally_lengths=[15, 20]),
        ]
        analysis = self.analyse_results(results)
        self.assertFalse(analysis['center_safe_long_rallies'])

    def test_analyse_recommendation_key_exists(self):
        results = [self._make('random', 5, 10)]
        analysis = self.analyse_results(results)
        self.assertIn('recommendation', analysis)

    def test_recommendation_is_valid_string(self):
        m = _import_eval()
        results = [
            self._make('random',     5, 10),
            self._make('left',       7, 10),
            self._make('right',      8, 10),
            self._make('left_short', 3, 10),
            self._make('right_short',6, 10),
            self._make('center_safe',5, 10),
        ]
        analysis = self.analyse_results(results)
        valid = {'keep_all_5', 'drop_center_safe', 'drop_left_short',
                 'reduce_to_4', 'accept_as_final'}
        self.assertIn(analysis['recommendation'], valid)


# =========================================================================== #
# 7. Left-collapse regression tests — make_picker argmax tie-breaking          #
# =========================================================================== #

class TestMakePickerNoLeftBias(unittest.TestCase):
    """
    Regression tests for the best_idx=0 initialization bug that caused the
    5-skill nash-p agent to always pick LEFT when the potential surface is flat.

    Strategy: give make_picker a model_p that returns CONSTANT output for all
    inputs.  With the old sequential-scan code, best_idx never changes from 0
    (left) because no later skill ever beats the first iteration.  With the
    correct collect-all-then-argmax code, np.argmax on a constant array returns
    index 0 (still), but critically:
      (a) If we tilt the constant model to prefer the LAST skill, the picker
          must return that skill — not always left.
      (b) The picker must return a valid skill index (0 to N_SKILLS-1).

    We test (a) by using a model that adds 1.0 to the value when the ego
    normalized skill index equals (N_SKILLS-1)/(N_SKILLS-1) = 1.0 (i.e. RIGHT,
    the last skill).  The correct argmax code picks RIGHT; the old code picks
    LEFT.
    """

    def setUp(self):
        m = _import_eval()
        if m is None:
            self.skipTest("nash_skills/eval_matchup.py not yet created")
        self.make_picker = m.make_picker
        try:
            import torch
            import torch.nn as nn
            from nash_skills.skills import N_SKILLS
        except ImportError:
            self.skipTest("torch or nash_skills not available")
        self.torch = torch
        self.nn = nn
        self.N_SKILLS = N_SKILLS

    def _make_last_skill_preferred_model(self):
        """
        Returns a trivial nn.Module whose output is:
            +1.0  if  x[-2] == 1.0   (right = last skill, normalized idx = 1.0)
            0.0   otherwise

        This model is flat for all skills except the last one.
        The correct argmax should always pick the last skill.
        """
        torch = self.torch
        nn = self.nn

        class LastSkillPreferred(nn.Module):
            def forward(self, x):
                # x shape: (batch, 116)
                # Return 1.0 where x[:,-2] == 1.0, else 0.0
                out = (x[:, -2] == 1.0).float().unsqueeze(1)
                return out

        return LastSkillPreferred()

    def _make_flat_model(self):
        """Returns a model that always outputs 0.0 regardless of input."""
        torch = self.torch
        nn = self.nn

        class FlatModel(nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], 1)

        return FlatModel()

    def test_picker_selects_last_skill_when_last_is_best(self):
        """
        When the model clearly prefers the last skill (normalized idx 1.0),
        the picker must return N_SKILLS-1 (RIGHT), not 0 (LEFT).

        This test FAILS with the old best_idx=0 sequential code because
        the first skill (idx=0) never gets a val > -inf at step 0, so
        best_idx stays 0 forever even though idx 4 is better.
        """
        model = self._make_last_skill_preferred_model()
        pick = self.make_picker("nash-p", model)
        obs = self.torch.zeros(116).numpy()
        # player=1, other_skill=0 (left)
        chosen = pick(1, obs, 0)
        self.assertEqual(
            chosen, self.N_SKILLS - 1,
            msg=f"Expected last skill (idx {self.N_SKILLS-1}=right), "
                f"got {chosen}. This indicates the best_idx=0 left-bias bug."
        )

    def test_picker_returns_valid_skill_index_flat_model(self):
        """
        Even for a completely flat model, the result must be a valid skill index.
        """
        model = self._make_flat_model()
        pick = self.make_picker("nash-p", model)
        obs = self.torch.zeros(116).numpy()
        chosen = pick(1, obs, 2)
        self.assertGreaterEqual(chosen, 0)
        self.assertLess(chosen, self.N_SKILLS)

    def test_picker_player2_also_correct(self):
        """
        Same test but for player=2 (opponent slot uses obs[-1] not obs[-2]).
        The model prefers obs[-1]==1.0 for this check.
        """
        torch = self.torch
        nn = self.nn
        N_SKILLS = self.N_SKILLS

        class LastSkillPreferredOpp(nn.Module):
            def forward(self, x):
                return (x[:, -1] == 1.0).float().unsqueeze(1)

        model = LastSkillPreferredOpp()
        pick = self.make_picker("nash-p", model)
        obs = torch.zeros(116).numpy()
        chosen = pick(2, obs, 0)
        self.assertEqual(
            chosen, N_SKILLS - 1,
            msg=f"Expected last skill for player2, got {chosen}."
        )


# =========================================================================== #
# 8. train_q_model.py encoding + LR regression tests                           #
# =========================================================================== #

class TestTrainQModelEncodingAndLR(unittest.TestCase):
    """
    Static code-inspection tests that verify train_q_model.py uses the correct
    {-1, +1} skill encoding and lr=0.001 for potential training.

    These tests do NOT execute any training — they read the source file and
    check for the correct literal values.  This catches regressions where
    someone accidentally reverts the fix.

    Root cause of left-collapse:
      - Old code: X01[:,-2] = 0.  (wrong — treated midpoint as "left")
      - Old code: lr=0.1           (100x too large → dying ReLU)
      - Fix:      X01[:,-2] = -1.  (correct — matches side_target for left)
      - Fix:      lr=0.001         (same as Q training LR)
    """

    def _read_train_q_model(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "train_q_model.py")
        with open(path, "r") as f:
            return f.read()

    def test_potential_lr_is_not_0_1(self):
        """
        lr=0.1 for the potential optimizer caused dying ReLU.
        The fix sets lr=0.001.  This test catches any regression back to 0.1.
        """
        src = self._read_train_q_model()
        # Must NOT find lr=0.1 on the optimizer_p line
        import re
        # Find the optimizer_p = ... line and check its lr
        match = re.search(r'optimizer_p\s*=.*?lr\s*=\s*([\d.]+)', src)
        self.assertIsNotNone(match, "Could not find optimizer_p lr in train_q_model.py")
        lr_val = float(match.group(1))
        self.assertNotAlmostEqual(
            lr_val, 0.1, places=4,
            msg=f"optimizer_p lr={lr_val} — should not be 0.1 (dying ReLU). "
                f"Fix: set lr=0.001"
        )

    def test_potential_lr_is_0_001(self):
        """lr=0.001 matches the Q-model LR and prevents dying ReLU."""
        src = self._read_train_q_model()
        import re
        match = re.search(r'optimizer_p\s*=.*?lr\s*=\s*([\d.]+)', src)
        self.assertIsNotNone(match)
        lr_val = float(match.group(1))
        self.assertAlmostEqual(
            lr_val, 0.001, places=4,
            msg=f"optimizer_p lr={lr_val} — expected 0.001"
        )

    def test_x01_ego_is_minus_one(self):
        """
        X01 represents (left ego, right opp).
        obs[-2] must be -1.0 to match env side_target for left.
        Old code used 0.0 which flipped the sign of the potential target.
        """
        src = self._read_train_q_model()
        # Check that X01[:,-2] is set to -1. (not 0.)
        import re
        match = re.search(r'X01\[:,-2\]\s*=\s*([-\d.]+)', src)
        self.assertIsNotNone(match, "Could not find X01[:,-2] assignment")
        val = float(match.group(1))
        self.assertAlmostEqual(
            val, -1.0, places=4,
            msg=f"X01[:,-2]={val} — must be -1.0 (left side_target). Old bug: was 0.0"
        )

    def test_x10_opp_is_minus_one(self):
        """
        X10 represents (right ego, left opp).
        obs[-1] must be -1.0 to match env side_target for left.
        """
        src = self._read_train_q_model()
        import re
        match = re.search(r'X10\[:,-1\]\s*=\s*([-\d.]+)', src)
        self.assertIsNotNone(match, "Could not find X10[:,-1] assignment")
        val = float(match.group(1))
        self.assertAlmostEqual(
            val, -1.0, places=4,
            msg=f"X10[:,-1]={val} — must be -1.0 (left side_target). Old bug: was 0.0"
        )

    def test_x00_ego_is_minus_one(self):
        """X00 = (left ego, left opp): both obs[-2] and obs[-1] must be -1.0."""
        src = self._read_train_q_model()
        import re
        match = re.search(r'X00\[:,-2\]\s*=\s*([-\d.]+)', src)
        self.assertIsNotNone(match, "Could not find X00[:,-2] assignment")
        val = float(match.group(1))
        self.assertAlmostEqual(val, -1.0, places=4,
                               msg=f"X00[:,-2]={val} — must be -1.0")

    def test_x00_opp_is_minus_one(self):
        src = self._read_train_q_model()
        import re
        match = re.search(r'X00\[:,-1\]\s*=\s*([-\d.]+)', src)
        self.assertIsNotNone(match, "Could not find X00[:,-1] assignment")
        val = float(match.group(1))
        self.assertAlmostEqual(val, -1.0, places=4,
                               msg=f"X00[:,-1]={val} — must be -1.0")


class TestStrategyValidation(unittest.TestCase):

    def setUp(self):
        m = _import_eval()
        if m is None:
            self.skipTest("nash_skills/eval_matchup.py not yet created")
        self.VALID_STRATEGIES = m.VALID_STRATEGIES

    def test_nash_p_is_valid(self):
        self.assertIn('nash-p', self.VALID_STRATEGIES)

    def test_random_is_valid(self):
        self.assertIn('random', self.VALID_STRATEGIES)

    def test_all_skill_names_are_valid(self):
        from nash_skills.skills import SKILL_NAMES
        for name in SKILL_NAMES:
            self.assertIn(name, self.VALID_STRATEGIES,
                          f"Skill name '{name}' should be a valid strategy")

    def test_smash_is_not_valid(self):
        self.assertNotIn('smash', self.VALID_STRATEGIES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
