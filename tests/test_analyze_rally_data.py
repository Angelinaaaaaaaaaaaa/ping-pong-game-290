"""
Unit tests for nash_skills/analyze_rally_data.py.

Tests cover:
  - count_rallies: total rally count and per-pair breakdown
  - count_states: total and per-pair state counts
  - usable_rallies: filtering out near-empty (<=MIN_STATES) rallies
  - imbalance_ratio: max/min usable counts across all 25 pairs
  - pair_recommendation: per-pair status (ok / weak / critical)
  - overall_recommendation: one of train_now / collect_more /
                             topup_weak_pairs / drop_weak_skills

Run: python -m pytest tests/test_analyze_rally_data.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import unittest


def _import():
    try:
        import nash_skills.analyze_rally_data as m
        return m
    except ModuleNotFoundError:
        return None


def _make_rallies(pairs_counts):
    """
    Build a synthetic rally list.
    pairs_counts: dict {(s1, s2): (n_rallies, states_per_rally)}
    """
    rallies = []
    for (s1, s2), (n_rallies, n_states) in pairs_counts.items():
        for _ in range(n_rallies):
            rallies.append({
                'skill1': s1,
                'skill2': s2,
                'states': [None] * n_states,
            })
    return rallies


# =========================================================================== #
# 1. count_rallies                                                             #
# =========================================================================== #

class TestCountRallies(unittest.TestCase):

    def setUp(self):
        m = _import()
        if m is None:
            self.skipTest("nash_skills/analyze_rally_data.py not yet created")
        self.m = m

    def test_total_rally_count(self):
        rallies = _make_rallies({('left','left'): (10, 5), ('left','right'): (8, 3)})
        result = self.m.count_rallies(rallies)
        self.assertEqual(result['total'], 18)

    def test_per_pair_count(self):
        rallies = _make_rallies({('left','left'): (10, 5), ('right','right'): (3, 4)})
        result = self.m.count_rallies(rallies)
        self.assertEqual(result['per_pair'][('left','left')], 10)
        self.assertEqual(result['per_pair'][('right','right')], 3)

    def test_missing_pair_is_zero(self):
        rallies = _make_rallies({('left','left'): (5, 5)})
        result = self.m.count_rallies(rallies)
        self.assertEqual(result['per_pair'].get(('right','right'), 0), 0)


# =========================================================================== #
# 2. count_states                                                              #
# =========================================================================== #

class TestCountStates(unittest.TestCase):

    def setUp(self):
        m = _import()
        if m is None:
            self.skipTest("nash_skills/analyze_rally_data.py not yet created")
        self.m = m

    def test_total_state_count(self):
        rallies = _make_rallies({('left','left'): (2, 10), ('left','right'): (3, 5)})
        result = self.m.count_states(rallies)
        self.assertEqual(result['total'], 35)

    def test_per_pair_state_count(self):
        rallies = _make_rallies({('left','left'): (4, 6), ('right','right'): (2, 3)})
        result = self.m.count_states(rallies)
        self.assertEqual(result['per_pair'][('left','left')], 24)
        self.assertEqual(result['per_pair'][('right','right')], 6)


# =========================================================================== #
# 3. usable_rallies                                                            #
# =========================================================================== #

class TestUsableRallies(unittest.TestCase):

    def setUp(self):
        m = _import()
        if m is None:
            self.skipTest("nash_skills/analyze_rally_data.py not yet created")
        self.m = m

    def test_filters_below_min_states(self):
        rallies = [
            {'skill1': 'left', 'skill2': 'left', 'states': [None] * 1},
            {'skill1': 'left', 'skill2': 'left', 'states': [None] * 2},
            {'skill1': 'left', 'skill2': 'left', 'states': [None] * 5},
            {'skill1': 'left', 'skill2': 'left', 'states': [None] * 10},
        ]
        result = self.m.usable_rallies(rallies, min_states=3)
        # Only the 5 and 10 state rallies are usable
        self.assertEqual(result['total_usable'], 2)

    def test_usable_per_pair(self):
        rallies = [
            {'skill1': 'left', 'skill2': 'left', 'states': [None] * 5},
            {'skill1': 'left', 'skill2': 'left', 'states': [None] * 1},
            {'skill1': 'right', 'skill2': 'right', 'states': [None] * 8},
        ]
        result = self.m.usable_rallies(rallies, min_states=3)
        self.assertEqual(result['per_pair'][('left','left')], 1)
        self.assertEqual(result['per_pair'][('right','right')], 1)

    def test_default_min_states_is_3(self):
        """Default threshold should be 3 (arm settling artifacts at <=2)."""
        m = self.m
        import inspect
        sig = inspect.signature(m.usable_rallies)
        self.assertEqual(sig.parameters['min_states'].default, 3)


# =========================================================================== #
# 4. imbalance_ratio                                                           #
# =========================================================================== #

class TestImbalanceRatio(unittest.TestCase):

    def setUp(self):
        m = _import()
        if m is None:
            self.skipTest("nash_skills/analyze_rally_data.py not yet created")
        self.m = m

    def test_balanced_dataset(self):
        per_pair = {('left','left'): 10, ('left','right'): 10,
                    ('right','left'): 10, ('right','right'): 10}
        ratio = self.m.imbalance_ratio(per_pair)
        self.assertAlmostEqual(ratio, 1.0)

    def test_imbalanced_dataset(self):
        per_pair = {('left','left'): 100, ('right','right'): 5,
                    ('left','right'): 10, ('right','left'): 10}
        ratio = self.m.imbalance_ratio(per_pair)
        self.assertAlmostEqual(ratio, 20.0)

    def test_zero_count_treated_as_one_for_division(self):
        """Missing pair (0 count) should not cause ZeroDivisionError."""
        # Only one pair present; max/min among present values = 1.0
        # The point is no ZeroDivisionError is raised.
        per_pair = {('left','left'): 50}
        ratio = self.m.imbalance_ratio(per_pair)
        self.assertGreaterEqual(ratio, 1.0)


# =========================================================================== #
# 5. pair_recommendation                                                       #
# =========================================================================== #

class TestPairRecommendation(unittest.TestCase):

    def setUp(self):
        m = _import()
        if m is None:
            self.skipTest("nash_skills/analyze_rally_data.py not yet created")
        self.m = m

    def test_ok_when_enough(self):
        status = self.m.pair_recommendation(usable=10)
        self.assertEqual(status, 'ok')

    def test_weak_when_few(self):
        status = self.m.pair_recommendation(usable=4)
        self.assertEqual(status, 'weak')

    def test_critical_when_very_few(self):
        status = self.m.pair_recommendation(usable=2)
        self.assertEqual(status, 'critical')

    def test_critical_when_zero(self):
        status = self.m.pair_recommendation(usable=0)
        self.assertEqual(status, 'critical')


# =========================================================================== #
# 6. overall_recommendation                                                    #
# =========================================================================== #

class TestOverallRecommendation(unittest.TestCase):

    def setUp(self):
        m = _import()
        if m is None:
            self.skipTest("nash_skills/analyze_rally_data.py not yet created")
        self.m = m

    _VALID = {'train_now', 'collect_more', 'topup_weak_pairs', 'drop_weak_skills'}

    def test_returns_valid_string(self):
        rec = self.m.overall_recommendation(
            total_usable=200, n_critical=0, n_weak=2, imbalance=3.0
        )
        self.assertIn(rec, self._VALID)

    def test_critical_pairs_trigger_not_train_now(self):
        rec = self.m.overall_recommendation(
            total_usable=100, n_critical=5, n_weak=3, imbalance=10.0
        )
        self.assertNotEqual(rec, 'train_now')

    def test_good_data_recommends_train_now(self):
        rec = self.m.overall_recommendation(
            total_usable=300, n_critical=0, n_weak=0, imbalance=2.0
        )
        self.assertEqual(rec, 'train_now')

    def test_high_imbalance_no_critical_recommends_topup(self):
        rec = self.m.overall_recommendation(
            total_usable=200, n_critical=0, n_weak=5, imbalance=20.0
        )
        self.assertIn(rec, {'topup_weak_pairs', 'collect_more'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
