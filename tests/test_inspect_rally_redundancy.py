"""Tests for nash_skills.v2.inspect_rally_redundancy.

TDD: written before implementation. Pure Python, no MuJoCo -- reports how
many training states each rally contributes, to check whether a handful of
long rallies dominate the dataset (correlated/non-iid samples, meeting note
item 7). Mirrors the style of inspect_truncated_rallies.py.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from nash_skills.v2.inspect_rally_redundancy import (
    gini_coefficient,
    concentration_stats,
    top_k_rallies_table,
    _validate_rally_schema,
)


def _rally(skill1, skill2, n_states, winner=1):
    return {
        "skill1": skill1, "skill2": skill2, "winner": winner,
        "states": [object()] * n_states,
    }


class TestGiniCoefficient:
    def test_uniform_distribution_is_zero(self):
        assert gini_coefficient([5, 5, 5, 5]) == pytest.approx(0.0, abs=1e-9)

    def test_empty_is_zero(self):
        assert gini_coefficient([]) == 0.0

    def test_all_zero_is_zero(self):
        assert gini_coefficient([0, 0, 0]) == 0.0

    def test_skewed_distribution_is_high(self):
        # One rally holds almost all the states -> high inequality
        g_skewed = gini_coefficient([0, 0, 0, 10])
        g_uniform = gini_coefficient([2, 3, 3, 2])
        assert g_skewed > g_uniform

    def test_bounded_between_0_and_1(self):
        g = gini_coefficient([1, 2, 3, 100, 0, 7])
        assert 0.0 <= g <= 1.0


class TestConcentrationStats:
    def test_totals(self):
        rallies = [_rally("left", "right", 5) for _ in range(10)]
        stats = concentration_stats(rallies)
        assert stats["total_rallies"] == 10
        assert stats["total_states"] == 50

    def test_uniform_top_frac_share_matches_frac(self):
        rallies = [_rally("left", "right", 5) for _ in range(100)]
        stats = concentration_stats(rallies, top_fracs=(0.10,))
        share = stats["top_fracs"][0.10]["share_of_total"]
        assert share == pytest.approx(0.10, abs=1e-6)

    def test_skewed_top_frac_share_is_large(self):
        rallies = [_rally("left", "right", 1) for _ in range(99)]
        rallies.append(_rally("left", "right", 1000))
        stats = concentration_stats(rallies, top_fracs=(0.01,))
        share = stats["top_fracs"][0.01]["share_of_total"]
        assert share > 0.85

    def test_max_states_single_rally(self):
        rallies = [_rally("left", "right", 3), _rally("left", "right", 40)]
        stats = concentration_stats(rallies)
        assert stats["max_states_single_rally"] == 40

    def test_gini_included(self):
        rallies = [_rally("left", "right", 5) for _ in range(10)]
        stats = concentration_stats(rallies)
        assert "gini" in stats
        assert stats["gini"] == pytest.approx(0.0, abs=1e-9)

    def test_empty_rallies(self):
        stats = concentration_stats([])
        assert stats["total_rallies"] == 0
        assert stats["total_states"] == 0
        assert stats["gini"] == 0.0


class TestTopKRalliesTable:
    def test_sorted_descending_by_states(self):
        rallies = [
            _rally("left", "right", 3),
            _rally("center_safe", "center_safe", 50),
            _rally("right", "left", 10),
        ]
        table = top_k_rallies_table(rallies, k=3)
        assert [row["n_states"] for row in table] == [50, 10, 3]

    def test_respects_k(self):
        rallies = [_rally("left", "right", n) for n in range(20)]
        table = top_k_rallies_table(rallies, k=5)
        assert len(table) == 5

    def test_row_contains_skill_and_winner_info(self):
        rallies = [_rally("left", "right", 12, winner=2)]
        table = top_k_rallies_table(rallies, k=1)
        row = table[0]
        assert row["skill1"] == "left"
        assert row["skill2"] == "right"
        assert row["winner"] == 2
        assert row["n_states"] == 12

    def test_empty_rallies_returns_empty(self):
        assert top_k_rallies_table([], k=10) == []


class TestValidateRallySchema:
    def test_accepts_dict_entries(self):
        _validate_rally_schema([_rally("left", "right", 3)], "some_file.pkl")

    def test_rejects_non_dict_first_entry(self):
        with pytest.raises(ValueError, match="rally-dict schema"):
            _validate_rally_schema([[1, 2, 3]], "rallies.pkl")

    def test_empty_list_does_not_raise(self):
        _validate_rally_schema([], "empty.pkl")
