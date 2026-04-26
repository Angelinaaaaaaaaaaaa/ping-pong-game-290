"""
Unit tests for skill_eval/final_report.py.

Tests cover:
  - load_matchup_json: loads and validates structure
  - load_brute_force_json: loads and validates structure
  - format_matchup_table: produces a table string with all matchups
  - format_brute_force_summary: produces a string with agreement rate
  - save_final_csv: writes clean CSV with required columns
  - save_final_json: writes clean JSON with results + analysis + brute_force

Run: python -m pytest tests/test_final_report.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import json
import tempfile
import unittest


def _import_fr():
    try:
        import skill_eval.final_report as m
        return m
    except ModuleNotFoundError:
        return None


# ─────────────────────────────────────────────────────────────────────────── #
# Fixtures                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def _make_matchup_entry(s2='random', episodes=20, ego_wins=12, opp_wins=8):
    return {
        'strategy1': 'nash-p',
        'strategy2': s2,
        'episodes':  episodes,
        'ego_wins':  ego_wins,
        'opp_wins':  opp_wins,
        'ego_contacts':  30,
        'opp_contacts':  25,
        'ego_successes': 25,
        'opp_successes': 20,
        'rally_lengths': [3, 4, 5, 6],
        'skill_usage': {'left': 50, 'left_short': 10, 'center_safe': 8,
                        'right_short': 7, 'right': 15},
        'win_rate':        ego_wins / episodes,
        'avg_rally_length': 4.5,
        'ego_success_rate': 25 / 30,
        'opp_success_rate': 20 / 25,
        'most_used_skill': 'left',
        'dominant_fraction': 0.56,
    }


def _make_matchup_json(tmp_path):
    data = {
        'results': [
            _make_matchup_entry('random',      20, 12, 8),
            _make_matchup_entry('left',        25, 6,  19),
            _make_matchup_entry('right',       10, 8,  2),
            _make_matchup_entry('left_short',  15, 11, 4),
            _make_matchup_entry('right_short', 9,  7,  2),
            _make_matchup_entry('center_safe', 10, 8,  2),
            _make_matchup_entry('nash-p-2skill', 22, 15, 7),
        ],
        'analysis': {
            'center_safe_long_rallies': False,
            'left_short_win_rate': 11/15,
            'recommendation': 'keep_all_5',
        },
    }
    path = os.path.join(tmp_path, 'matchup_results.json')
    with open(path, 'w') as f:
        json.dump(data, f)
    return path, data


def _make_brute_force_json(tmp_path):
    data = {
        'n_states': 500,
        'n_agree': 416,
        'agreement_rate': 0.832,
        'n_disagreements': 84,
        'tally': {
            'ego_bf_skill': 'right',
            'ego_np_skill': 'left',
            'opp_bf_skill': 'right',
            'opp_np_skill': 'right',
            'ego_bf_counts': {'left': 19, 'right': 20, 'left_short': 18,
                              'right_short': 17, 'center_safe': 10},
            'ego_np_counts': {'left': 23, 'center_safe': 18, 'right_short': 16,
                              'left_short': 15, 'right': 12},
        },
        'verdict': 'GOOD — approximation mostly agrees; acceptable for a course project.',
        'sample_joint_surface': [[0.1] * 5] * 5,
    }
    path = os.path.join(tmp_path, 'brute_force_compare.json')
    with open(path, 'w') as f:
        json.dump(data, f)
    return path, data


# ─────────────────────────────────────────────────────────────────────────── #
# 1. load_matchup_json                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

class TestLoadMatchupJson(unittest.TestCase):

    def setUp(self):
        m = _import_fr()
        if m is None:
            self.skipTest("skill_eval/final_report.py not yet created")
        self.m = m
        self.tmp = tempfile.mkdtemp()

    def test_loads_results_list(self):
        path, _ = _make_matchup_json(self.tmp)
        results, analysis = self.m.load_matchup_json(path)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 7)

    def test_loads_analysis_dict(self):
        path, _ = _make_matchup_json(self.tmp)
        _, analysis = self.m.load_matchup_json(path)
        self.assertIsInstance(analysis, dict)
        self.assertIn('recommendation', analysis)

    def test_each_result_has_required_keys(self):
        path, _ = _make_matchup_json(self.tmp)
        results, _ = self.m.load_matchup_json(path)
        required = {'strategy1', 'strategy2', 'episodes', 'ego_wins',
                    'opp_wins', 'win_rate', 'avg_rally_length', 'ego_contacts'}
        for r in results:
            for k in required:
                self.assertIn(k, r, f"Missing key '{k}' in result")


# ─────────────────────────────────────────────────────────────────────────── #
# 2. load_brute_force_json                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

class TestLoadBruteForceJson(unittest.TestCase):

    def setUp(self):
        m = _import_fr()
        if m is None:
            self.skipTest("skill_eval/final_report.py not yet created")
        self.m = m
        self.tmp = tempfile.mkdtemp()

    def test_loads_dict(self):
        path, _ = _make_brute_force_json(self.tmp)
        bf = self.m.load_brute_force_json(path)
        self.assertIsInstance(bf, dict)

    def test_has_agreement_rate(self):
        path, _ = _make_brute_force_json(self.tmp)
        bf = self.m.load_brute_force_json(path)
        self.assertIn('agreement_rate', bf)
        self.assertAlmostEqual(bf['agreement_rate'], 0.832)

    def test_has_verdict(self):
        path, _ = _make_brute_force_json(self.tmp)
        bf = self.m.load_brute_force_json(path)
        self.assertIn('verdict', bf)


# ─────────────────────────────────────────────────────────────────────────── #
# 3. format_matchup_table                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

class TestFormatMatchupTable(unittest.TestCase):

    def setUp(self):
        m = _import_fr()
        if m is None:
            self.skipTest("skill_eval/final_report.py not yet created")
        self.m = m
        self.tmp = tempfile.mkdtemp()

    def test_returns_string(self):
        path, data = _make_matchup_json(self.tmp)
        results, analysis = self.m.load_matchup_json(path)
        table = self.m.format_matchup_table(results, analysis)
        self.assertIsInstance(table, str)

    def test_contains_all_opponents(self):
        path, data = _make_matchup_json(self.tmp)
        results, analysis = self.m.load_matchup_json(path)
        table = self.m.format_matchup_table(results, analysis)
        for opp in ('random', 'left', 'right', 'left_short', 'right_short',
                    'center_safe', 'nash-p-2skill'):
            self.assertIn(opp, table, f"Opponent '{opp}' missing from table")

    def test_contains_win_rate_column(self):
        path, data = _make_matchup_json(self.tmp)
        results, analysis = self.m.load_matchup_json(path)
        table = self.m.format_matchup_table(results, analysis)
        self.assertTrue(
            'win' in table.lower() or '%' in table,
            "Win rate not visible in table"
        )

    def test_contains_recommendation(self):
        path, data = _make_matchup_json(self.tmp)
        results, analysis = self.m.load_matchup_json(path)
        table = self.m.format_matchup_table(results, analysis)
        self.assertIn('keep_all_5', table)


# ─────────────────────────────────────────────────────────────────────────── #
# 4. format_brute_force_summary                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class TestFormatBruteForceSummary(unittest.TestCase):

    def setUp(self):
        m = _import_fr()
        if m is None:
            self.skipTest("skill_eval/final_report.py not yet created")
        self.m = m
        self.tmp = tempfile.mkdtemp()

    def test_returns_string(self):
        path, _ = _make_brute_force_json(self.tmp)
        bf = self.m.load_brute_force_json(path)
        summary = self.m.format_brute_force_summary(bf)
        self.assertIsInstance(summary, str)

    def test_contains_agreement_rate(self):
        path, _ = _make_brute_force_json(self.tmp)
        bf = self.m.load_brute_force_json(path)
        summary = self.m.format_brute_force_summary(bf)
        self.assertIn('83', summary)   # 83.2%

    def test_contains_flat_surface_explanation(self):
        path, _ = _make_brute_force_json(self.tmp)
        bf = self.m.load_brute_force_json(path)
        summary = self.m.format_brute_force_summary(bf)
        # Should mention flat/tie/near-tie explanation
        lower = summary.lower()
        self.assertTrue(
            'flat' in lower or 'tie' in lower or 'near' in lower,
            f"No flat-surface explanation found in: {summary[:300]}"
        )

    def test_contains_disagreement_rate(self):
        path, _ = _make_brute_force_json(self.tmp)
        bf = self.m.load_brute_force_json(path)
        summary = self.m.format_brute_force_summary(bf)
        # 16.8% disagreement rate should appear (84/500)
        self.assertTrue('16' in summary or 'disagree' in summary.lower())


# ─────────────────────────────────────────────────────────────────────────── #
# 5. save_final_csv                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

class TestSaveFinalCsv(unittest.TestCase):

    def setUp(self):
        m = _import_fr()
        if m is None:
            self.skipTest("skill_eval/final_report.py not yet created")
        self.m = m
        self.tmp = tempfile.mkdtemp()

    def test_writes_csv_with_header(self):
        path, data = _make_matchup_json(self.tmp)
        results, _ = self.m.load_matchup_json(path)
        out = os.path.join(self.tmp, 'final.csv')
        self.m.save_final_csv(results, out)
        with open(out) as f:
            reader = csv.DictReader(f)
            self.assertIn('strategy2',      reader.fieldnames)
            self.assertIn('win_rate',       reader.fieldnames)
            self.assertIn('episodes',       reader.fieldnames)
            self.assertIn('ego_wins',       reader.fieldnames)
            self.assertIn('opp_wins',       reader.fieldnames)
            self.assertIn('ego_contacts',   reader.fieldnames)
            self.assertIn('avg_rally_length', reader.fieldnames)

    def test_csv_row_count(self):
        path, data = _make_matchup_json(self.tmp)
        results, _ = self.m.load_matchup_json(path)
        out = os.path.join(self.tmp, 'final.csv')
        self.m.save_final_csv(results, out)
        with open(out) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 7)


# ─────────────────────────────────────────────────────────────────────────── #
# 6. save_final_json                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

class TestSaveFinalJson(unittest.TestCase):

    def setUp(self):
        m = _import_fr()
        if m is None:
            self.skipTest("skill_eval/final_report.py not yet created")
        self.m = m
        self.tmp = tempfile.mkdtemp()

    def test_writes_json_with_top_level_keys(self):
        mpath, mdata = _make_matchup_json(self.tmp)
        bpath, bdata = _make_brute_force_json(self.tmp)
        results, analysis = self.m.load_matchup_json(mpath)
        bf = self.m.load_brute_force_json(bpath)
        out = os.path.join(self.tmp, 'final.json')
        self.m.save_final_json(results, analysis, bf, out)
        with open(out) as f:
            out_data = json.load(f)
        self.assertIn('results',      out_data)
        self.assertIn('analysis',     out_data)
        self.assertIn('brute_force',  out_data)
        self.assertIn('generated_at', out_data)

    def test_results_count_preserved(self):
        mpath, _ = _make_matchup_json(self.tmp)
        bpath, _ = _make_brute_force_json(self.tmp)
        results, analysis = self.m.load_matchup_json(mpath)
        bf = self.m.load_brute_force_json(bpath)
        out = os.path.join(self.tmp, 'final.json')
        self.m.save_final_json(results, analysis, bf, out)
        with open(out) as f:
            out_data = json.load(f)
        self.assertEqual(len(out_data['results']), 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
