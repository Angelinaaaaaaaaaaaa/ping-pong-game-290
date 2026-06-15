"""
Unit tests for nash_skills/eval_matchup_2skill.py.

The 2-skill baseline evaluator runs the same headless matchup loop as
eval_matchup.py but uses only the original 2 skills (left / right) and
the original potential model (models/model_p.pth).

Tests cover:
  - VALID_STRATEGIES_2SKILL constant
  - MatchupResult2Skill dataclass (reuses or wraps MatchupResult)
  - make_picker_2skill: left, right, random, nash-p-2skill strategies
  - save_csv_2skill: writes correct CSV columns
  - print_summary_2skill: contains matchup names and win rates

Run: python -m pytest tests/test_eval_matchup_2skill.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv, io, json, tempfile, unittest
import numpy as np


def _import():
    try:
        import nash_skills.eval_matchup_2skill as m
        return m
    except ModuleNotFoundError:
        return None


# =========================================================================== #
# 1. VALID_STRATEGIES_2SKILL                                                  #
# =========================================================================== #

class TestValidStrategies(unittest.TestCase):

    def setUp(self):
        m = _import()
        if m is None:
            self.skipTest("nash_skills/eval_matchup_2skill.py not yet created")
        self.m = m

    def test_left_is_valid(self):
        self.assertIn('left', self.m.VALID_STRATEGIES_2SKILL)

    def test_right_is_valid(self):
        self.assertIn('right', self.m.VALID_STRATEGIES_2SKILL)

    def test_random_is_valid(self):
        self.assertIn('random', self.m.VALID_STRATEGIES_2SKILL)

    def test_nash_p_2skill_is_valid(self):
        self.assertIn('nash-p-2skill', self.m.VALID_STRATEGIES_2SKILL)

    def test_only_2_skills(self):
        # Should not include 5-skill-only skills
        self.assertNotIn('left_short',  self.m.VALID_STRATEGIES_2SKILL)
        self.assertNotIn('right_short', self.m.VALID_STRATEGIES_2SKILL)
        self.assertNotIn('center_safe', self.m.VALID_STRATEGIES_2SKILL)


# =========================================================================== #
# 2. make_picker_2skill                                                       #
# =========================================================================== #

class TestMakePicker2Skill(unittest.TestCase):

    def setUp(self):
        m = _import()
        if m is None:
            self.skipTest("nash_skills/eval_matchup_2skill.py not yet created")
        self.m = m

    def _dummy_model(self):
        import torch, torch.nn as nn
        class ConstModel(nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], 1)
        return ConstModel()

    def test_left_always_returns_0(self):
        m = self.m
        pick = m.make_picker_2skill('left', self._dummy_model())
        obs = np.zeros(116, dtype=np.float32)
        for _ in range(10):
            self.assertEqual(pick(1, obs, 1), 0)   # left = index 0

    def test_right_always_returns_1(self):
        m = self.m
        pick = m.make_picker_2skill('right', self._dummy_model())
        obs = np.zeros(116, dtype=np.float32)
        for _ in range(10):
            self.assertEqual(pick(1, obs, 0), 1)   # right = index 1

    def test_random_returns_0_or_1(self):
        m = self.m
        pick = m.make_picker_2skill('random', self._dummy_model())
        obs = np.zeros(116, dtype=np.float32)
        results = {pick(1, obs, 0) for _ in range(30)}
        self.assertTrue(results.issubset({0, 1}))

    def test_nash_p_2skill_returns_0_or_1(self):
        m = self.m
        pick = m.make_picker_2skill('nash-p-2skill', self._dummy_model())
        obs = np.zeros(116, dtype=np.float32)
        result = pick(1, obs, 0)
        self.assertIn(result, (0, 1))

    def test_unknown_strategy_raises(self):
        m = self.m
        with self.assertRaises(ValueError):
            m.make_picker_2skill('smash', self._dummy_model())

    def test_nash_p_uses_plus_minus_1_encoding_not_0_1(self):
        """
        The original model_p was trained with obs[-2], obs[-1] in {-1.0, +1.0}.
        make_picker_2skill must write -1.0/+1.0 (left/right) into those dims,
        NOT 0.0/1.0.  We verify this by using a model that reads obs[-2] and
        checking the values it actually receives.
        """
        import torch, torch.nn as nn

        seen_values = []

        class RecordingModel(nn.Module):
            def forward(self, x):
                # record the last two dims of every row in the batch
                for row in x:
                    seen_values.append((float(row[-2]), float(row[-1])))
                return torch.zeros(x.shape[0], 1)

        m = self.m
        pick = m.make_picker_2skill('nash-p-2skill', RecordingModel())
        obs = np.zeros(116, dtype=np.float32)
        pick(1, obs, 0)   # ego picks; other_idx=0 (left)

        # All values fed into the model must be ±1.0, never 0.0 or 1.0-only
        all_dims = [v for pair in seen_values for v in pair]
        self.assertTrue(
            all(v in (-1.0, 1.0) for v in all_dims),
            f"Expected only ±1.0 in obs[-2:], got: {set(all_dims)}"
        )


# =========================================================================== #
# 3. save_csv_2skill                                                          #
# =========================================================================== #

class TestSaveCsv2Skill(unittest.TestCase):

    def setUp(self):
        m = _import()
        if m is None:
            self.skipTest("nash_skills/eval_matchup_2skill.py not yet created")
        self.m = m

    def _make_result(self, s2='random', wins=5, total=10):
        from nash_skills.eval_matchup import MatchupResult
        return MatchupResult(
            strategy1='nash-p-2skill', strategy2=s2,
            episodes=total, truncated_episodes=0,
            ego_wins=wins, opp_wins=total-wins,
            ego_contacts=15, opp_contacts=12,
            ego_successes=12, opp_successes=9,
            rally_lengths=[2, 3, 4],
            episode_steps=[10, 12, 11],
            skill_usage={'left': 2, 'right': 1},
            total_steps=100,
        )

    def test_csv_has_required_columns(self):
        m = self.m
        results = [self._make_result('random'), self._make_result('left')]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        m.save_csv_2skill(results, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            for col in ('strategy1', 'strategy2', 'episodes',
                        'ego_wins', 'opp_wins', 'win_rate',
                        'ego_contacts', 'avg_rally_length'):
                self.assertIn(col, reader.fieldnames)
        os.unlink(path)

    def test_csv_row_count(self):
        m = self.m
        results = [self._make_result('random'), self._make_result('left'),
                   self._make_result('right')]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        m.save_csv_2skill(results, path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)
        os.unlink(path)


# =========================================================================== #
# 4. print_summary_2skill                                                     #
# =========================================================================== #

class TestPrintSummary2Skill(unittest.TestCase):

    def setUp(self):
        m = _import()
        if m is None:
            self.skipTest("nash_skills/eval_matchup_2skill.py not yet created")
        self.m = m

    def _make_result(self, s2='random', wins=6, total=10):
        from nash_skills.eval_matchup import MatchupResult
        return MatchupResult(
            strategy1='nash-p-2skill', strategy2=s2,
            episodes=total, truncated_episodes=0,
            ego_wins=wins, opp_wins=total-wins,
            ego_contacts=15, opp_contacts=12,
            ego_successes=12, opp_successes=9,
            rally_lengths=[2, 3, 4],
            episode_steps=[10, 12, 11],
            skill_usage={'left': 2, 'right': 1},
            total_steps=100,
        )

    def test_summary_contains_opponents(self):
        m = self.m
        results = [self._make_result('random'), self._make_result('left'),
                   self._make_result('right')]
        buf = io.StringIO()
        m.print_summary_2skill(results, file=buf)
        output = buf.getvalue()
        self.assertIn('random', output)
        self.assertIn('left',   output)
        self.assertIn('right',  output)

    def test_summary_contains_win_rate(self):
        m = self.m
        results = [self._make_result('random', wins=8, total=10)]
        buf = io.StringIO()
        m.print_summary_2skill(results, file=buf)
        output = buf.getvalue()
        self.assertIn('80', output)

    def test_summary_has_header(self):
        m = self.m
        results = [self._make_result('random')]
        buf = io.StringIO()
        m.print_summary_2skill(results, file=buf)
        output = buf.getvalue()
        lower = output.lower()
        self.assertTrue('strategy' in lower or 'win' in lower)

    def test_summary_title_says_2skill_not_5skill(self):
        """
        print_summary_2skill must NOT show '5-SKILL' in its title line.
        It should say '2-SKILL' (or 'BASELINE') to distinguish from the
        5-skill evaluator's output.
        """
        m = self.m
        results = [self._make_result('random')]
        buf = io.StringIO()
        m.print_summary_2skill(results, file=buf)
        output = buf.getvalue()
        self.assertNotIn('5-SKILL', output,
                         "print_summary_2skill must not print '5-SKILL' in title")
        self.assertTrue(
            '2-SKILL' in output or 'BASELINE' in output or '2-skill' in output.lower(),
            f"Expected '2-SKILL' or 'BASELINE' in title, got:\n{output[:300]}"
        )


# =========================================================================== #
# 5. _parse_contact_lines  (RED – not yet exported)                           #
# =========================================================================== #

class TestParseContactLines(unittest.TestCase):

    def setUp(self):
        m = _import()
        if m is None:
            self.skipTest("nash_skills/eval_matchup_2skill.py not yet created")
        if not hasattr(m, '_parse_contact_lines'):
            self.skipTest("_parse_contact_lines not yet public")
        self.m = m

    def test_parse_contact_lines_catches_ego_print(self):
        """
        A line exactly matching the env's ego print should increment ego_contacts.
        Format: 'Returned successfully by ego <x> <y>'
        """
        m = self.m
        lines = ["Returned successfully by ego 1.876 0.198"]
        e_c, o_c, _e_s, _o_s = m._parse_contact_lines(lines)
        self.assertEqual(e_c, 1, "ego_contacts should be 1 for one ego line")
        self.assertEqual(o_c, 0, "opp_contacts should be 0")

    def test_parse_contact_lines_catches_opp_print(self):
        """
        A line exactly matching the env's opp print should increment opp_contacts.
        Format: 'Returned successfully by opp <x> <y>'
        """
        m = self.m
        lines = ["Returned successfully by opp 1.019 -0.249"]
        e_c, o_c, _e_s, _o_s = m._parse_contact_lines(lines)
        self.assertEqual(o_c, 1, "opp_contacts should be 1 for one opp line")
        self.assertEqual(e_c, 0, "ego_contacts should be 0")

    def test_parse_contact_lines_clarification_note(self):
        """
        ego_contacts counts only SUCCESSFUL (in-bounds) returns, NOT all
        racket-ball hits. A hit where the ball lands off-table does NOT emit
        a 'Returned successfully' print, so it is invisible to this parser.
        This test documents that behaviour explicitly.
        """
        m = self.m
        # No 'Returned successfully' lines → contacts = 0 even if the ball hit
        lines = [
            "racket_hit detected at (1.2, 0.1)",   # hypothetical raw hit line (not real)
            "ball bounced off racket",              # not matching 'Returned successfully'
        ]
        e_c, o_c, _e_s, _o_s = m._parse_contact_lines(lines)
        self.assertEqual(e_c, 0, "Non-standard lines must not be counted as contacts")
        self.assertEqual(o_c, 0)


# =========================================================================== #
# 6. Guaranteed episode count  (RED – loop bug not yet fixed)                 #
# =========================================================================== #

class TestGuaranteedEpisodeCount(unittest.TestCase):
    """
    run_matchup_2skill must complete EXACTLY n_episodes episodes even when
    individual episodes are long (pathological case: env never sends done=True,
    so every episode hits the per-episode step cap and gets force-reset).

    Currently the loop has an extra `total_steps < max_total_steps` guard that
    can exit early.  This test should be RED until that guard is removed.
    """

    def setUp(self):
        m = _import()
        if m is None:
            self.skipTest("nash_skills/eval_matchup_2skill.py not yet created")
        self.m = m

    def test_guaranteed_episode_count(self):
        import types, torch, torch.nn as nn
        from unittest.mock import MagicMock, patch

        # --- tiny model_p that always returns 0 ---
        class ZeroModel(nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], 1)

        # --- fake env: done=False always (forces truncation each episode) ---
        fake_obs = np.zeros(116, dtype=np.float32)
        fake_obs[36] = 1.6   # ball just past TABLE_SHIFT so crossing detected
        # _build_obs1 / _build_obs2 need these info keys
        fake_info = {
            "diff_pos":      np.zeros(3,  dtype=np.float32),
            "diff_quat":     np.zeros(4,  dtype=np.float32),
            "target":        np.zeros(7,  dtype=np.float32),
            "diff_pos_opp":  np.zeros(3,  dtype=np.float32),
            "diff_quat_opp": np.zeros(4,  dtype=np.float32),
            "target_opp":    np.zeros(7,  dtype=np.float32),
        }

        fake_env_instance = MagicMock()
        fake_env_instance.reset.return_value = (fake_obs.copy(), fake_info)
        # step always returns done=False
        fake_env_instance.step.return_value = (fake_obs.copy(), 0.0, False, False, fake_info)

        # --- fake _TwoSkillEnv that wraps the fake env ---
        m = self.m

        class FakeTwoSkillEnv:
            def __init__(self, proc_id=1): pass
            def set_skills(self, _i1, _i2): pass
            def reset(self, seed=None):
                return fake_obs.copy(), fake_info
            def step(self, _action):
                return fake_obs.copy(), 0.0, False, False, fake_info
            def close(self): pass

        # --- fake PPO ---
        fake_ppo = MagicMock()
        fake_ppo.predict.return_value = (np.zeros(9), None)

        # Patch _TwoSkillEnv inside the module and disable stdout capture overhead
        with patch.object(m, '_TwoSkillEnv', FakeTwoSkillEnv), \
             patch.object(m, '_capture_env_step',
                          lambda env, action: (
                              (fake_obs.copy(), 0.0, False, False, fake_info), []
                          )):
            result = m.run_matchup_2skill(
                strategy1='left',
                strategy2='right',
                ppo=fake_ppo,
                model_p=ZeroModel(),
                n_episodes=10,
                max_steps_per_episode=5,   # tiny cap → many forced resets
                warmup_steps=0,
                max_total_steps=None,      # must NOT cap total steps
            )

        self.assertEqual(
            result.episodes, 10,
            f"Expected exactly 10 completed episodes, got {result.episodes}"
        )


# =========================================================================== #
# 7. MatchupResult.win_rate_clean and done_episodes                           #
# =========================================================================== #

class TestWinRateClean(unittest.TestCase):
    """
    win_rate (ego_wins / episodes) is misleading when most episodes are
    truncated (no winner decided).  We need:

      done_episodes   = episodes - truncated_episodes
      win_rate_clean  = ego_wins / done_episodes   (None if done_episodes == 0)

    Currently MatchupResult has no win_rate_clean property → tests are RED.
    """

    def setUp(self):
        m = _import()
        if m is None:
            self.skipTest("nash_skills/eval_matchup_2skill.py not yet created")
        self.m = m

    def _make_result(self, ego_wins, trunc, total=60):
        from nash_skills.eval_matchup import MatchupResult
        done = total - trunc
        return MatchupResult(
            strategy1='nash-p-2skill', strategy2='random',
            episodes=total, truncated_episodes=trunc,
            ego_wins=ego_wins, opp_wins=done - ego_wins,
            ego_contacts=0, opp_contacts=0,
            ego_successes=0, opp_successes=0,
            rally_lengths=[5] * total,
            episode_steps=[300] * total,
            skill_usage={'left': 1, 'right': 1},
            total_steps=total * 300,
        )

    def test_done_episodes_property_exists(self):
        """MatchupResult must have a done_episodes property."""
        from nash_skills.eval_matchup import MatchupResult
        r = self._make_result(ego_wins=3, trunc=55)
        self.assertTrue(
            hasattr(r, 'done_episodes'),
            "MatchupResult has no 'done_episodes' property"
        )

    def test_done_episodes_value(self):
        r = self._make_result(ego_wins=3, trunc=55)
        # 60 total - 55 truncated = 5 done
        self.assertEqual(r.done_episodes, 5)

    def test_win_rate_clean_property_exists(self):
        from nash_skills.eval_matchup import MatchupResult
        r = self._make_result(ego_wins=3, trunc=55)
        self.assertTrue(
            hasattr(r, 'win_rate_clean'),
            "MatchupResult has no 'win_rate_clean' property"
        )

    def test_win_rate_clean_value(self):
        r = self._make_result(ego_wins=3, trunc=55)
        # 3 wins out of 5 done episodes = 0.6
        self.assertAlmostEqual(r.win_rate_clean, 0.6)

    def test_win_rate_clean_none_when_all_truncated(self):
        r = self._make_result(ego_wins=0, trunc=60)
        self.assertIsNone(r.win_rate_clean)

    def test_old_win_rate_still_works(self):
        """win_rate (over all episodes) must still exist and equal ego_wins/episodes."""
        r = self._make_result(ego_wins=3, trunc=55)
        self.assertAlmostEqual(r.win_rate, 3 / 60)


# =========================================================================== #
# 8. _capture_env_step stdout capture                                         #
# =========================================================================== #

class TestCaptureEnvStep(unittest.TestCase):
    """
    Verify that _capture_env_step really captures Python print() calls that
    happen inside env.step().  This is the exact code path that should be
    capturing 'Returned successfully by ego ...' lines.

    If this test passes, the capture mechanism works and ego_contacts=0 must
    mean the print condition is never triggered (ball lands out-of-bounds).
    If this test fails, the capture mechanism is broken.
    """

    def setUp(self):
        m = _import()
        if m is None:
            self.skipTest("nash_skills/eval_matchup_2skill.py not yet created")
        if not hasattr(m, '_capture_env_step'):
            self.skipTest("_capture_env_step not yet public")
        self.m = m

    def test_capture_env_step_captures_python_print(self):
        """
        A fake env whose step() calls Python print() must have that output
        captured in the returned lines list.
        """
        m = self.m
        fake_obs = np.zeros(116, dtype=np.float32)
        fake_info = {}

        class PrintingEnv:
            def step(self, _action):
                print("Returned successfully by ego 1.876 0.198")
                return fake_obs.copy(), 0.0, False, False, fake_info

        env = PrintingEnv()
        result, lines = m._capture_env_step(env, np.zeros(18))

        self.assertGreater(
            len(lines), 0,
            "_capture_env_step returned no lines — print() was not captured"
        )
        self.assertTrue(
            any("ego" in l for l in lines),
            f"Expected an 'ego' line but got: {lines}"
        )

    def test_capture_env_step_ego_increments_contact_count(self):
        """
        End-to-end: fake env prints ego line → _parse_contact_lines sees ego_contacts=1.
        """
        m = self.m
        fake_obs = np.zeros(116, dtype=np.float32)
        fake_info = {}

        class PrintingEnv:
            def step(self, _action):
                print("Returned successfully by ego 1.876 0.198")
                return fake_obs.copy(), 0.0, False, False, fake_info

        env = PrintingEnv()
        _result, lines = m._capture_env_step(env, np.zeros(18))
        e_c, o_c, _e_s, _o_s = m._parse_contact_lines(lines)

        self.assertEqual(e_c, 1, f"Expected ego_contacts=1, got {e_c}. Lines: {lines}")
        self.assertEqual(o_c, 0)


# =========================================================================== #
# 9. Self-contained: no cross-import from eval_matchup                        #
# =========================================================================== #

class TestSelfContained(unittest.TestCase):
    """
    eval_matchup_2skill.py must define MatchupResult, save_csv, print_summary,
    _build_obs1, and _build_obs2 locally — it must NOT rely on importing them
    from nash_skills.eval_matchup at runtime.

    We verify this by checking that each name is present in the module's own
    __dict__ (i.e., defined or assigned there) rather than only inherited via
    a cross-module import that the user wants removed.
    """

    def setUp(self):
        m = _import()
        if m is None:
            self.skipTest("nash_skills/eval_matchup_2skill.py not yet created")
        self.m = m

    def _defined_in_module(self, name):
        """True iff `name` lives in eval_matchup_2skill's own namespace."""
        obj = getattr(self.m, name, None)
        if obj is None:
            return False
        module = getattr(obj, '__module__', None)
        if module is None:
            # dataclass or other object — check it's actually in the dict
            return name in self.m.__dict__
        return 'eval_matchup_2skill' in module

    def test_MatchupResult_defined_locally(self):
        self.assertIn(
            'MatchupResult', self.m.__dict__,
            "MatchupResult must be defined in eval_matchup_2skill, not just imported"
        )

    def test_save_csv_defined_locally(self):
        obj = self.m.__dict__.get('save_csv')
        self.assertIsNotNone(obj, "save_csv must be defined in eval_matchup_2skill")
        # Must not be the exact same object as in eval_matchup
        try:
            import nash_skills.eval_matchup as em
            self.assertIsNot(
                obj, em.save_csv,
                "save_csv in eval_matchup_2skill must be a local definition, "
                "not the imported object from eval_matchup"
            )
        except ImportError:
            pass   # if eval_matchup doesn't exist, the local def is fine

    def test_print_summary_defined_locally(self):
        obj = self.m.__dict__.get('print_summary')
        self.assertIsNotNone(obj, "print_summary must be defined in eval_matchup_2skill")
        try:
            import nash_skills.eval_matchup as em
            self.assertIsNot(
                obj, em.print_summary,
                "print_summary in eval_matchup_2skill must be a local definition"
            )
        except ImportError:
            pass

    def test_build_obs1_defined_locally(self):
        obj = self.m.__dict__.get('_build_obs1')
        self.assertIsNotNone(obj, "_build_obs1 must be defined in eval_matchup_2skill")
        try:
            import nash_skills.eval_matchup as em
            self.assertIsNot(obj, em._build_obs1,
                             "_build_obs1 must be a local copy, not the imported object")
        except ImportError:
            pass

    def test_build_obs2_defined_locally(self):
        obj = self.m.__dict__.get('_build_obs2')
        self.assertIsNotNone(obj, "_build_obs2 must be defined in eval_matchup_2skill")
        try:
            import nash_skills.eval_matchup as em
            self.assertIsNot(obj, em._build_obs2,
                             "_build_obs2 must be a local copy, not the imported object")
        except ImportError:
            pass

    def test_no_runtime_import_of_eval_matchup(self):
        """
        The module's source must not contain the cross-import block.
        We check the source file directly.
        """
        import inspect
        src = inspect.getsource(self.m)
        self.assertNotIn(
            'from nash_skills.eval_matchup import',
            src,
            "eval_matchup_2skill.py still contains 'from nash_skills.eval_matchup import'"
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
