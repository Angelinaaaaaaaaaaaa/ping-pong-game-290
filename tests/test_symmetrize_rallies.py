"""Tests for nash_skills.v2.symmetrize_rallies — winner=None handling.

TDD: written before the None-winner-count fix (L2 finding from code review).
_count_winners must not let a None winner silently create a new dict key
that distorts the printed percentages; it must be treated as truncated.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nash_skills.v2.symmetrize_rallies import count_winners


class TestCountWinners:
    def test_counts_ego_and_opp_wins(self):
        rallies = [{"winner": 1}, {"winner": 2}, {"winner": 1}]
        counts = count_winners(rallies)
        assert counts[1] == 2
        assert counts[2] == 1

    def test_zero_winner_counted_as_truncated(self):
        rallies = [{"winner": 0}, {"winner": 1}]
        counts = count_winners(rallies)
        assert counts[0] == 1
        assert counts[1] == 1

    def test_none_winner_counted_as_truncated_not_new_key(self):
        rallies = [{"winner": None}, {"winner": 1}, {"winner": None}]
        counts = count_winners(rallies)
        assert counts[0] == 2  # both None winners folded into truncated bucket
        assert counts[1] == 1
        assert None not in counts

    def test_mixed_zero_and_none_both_count_as_truncated(self):
        rallies = [{"winner": 0}, {"winner": None}, {"winner": 1}, {"winner": 2}]
        counts = count_winners(rallies)
        assert counts[0] == 2
        assert counts[1] == 1
        assert counts[2] == 1

    def test_total_equals_input_length(self):
        rallies = [{"winner": 1}, {"winner": None}, {"winner": 2}, {"winner": 0}]
        counts = count_winners(rallies)
        assert sum(counts.values()) == len(rallies)

    def test_missing_winner_key_treated_as_truncated(self):
        rallies = [{}, {"winner": 1}]
        counts = count_winners(rallies)
        assert counts[0] == 1
        assert counts[1] == 1

    def test_empty_list_returns_zero_counts(self):
        counts = count_winners([])
        assert counts[0] == 0
        assert counts[1] == 0
        assert counts[2] == 0
