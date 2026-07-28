"""Tests for nash_skills.v2.inspect_truncated_rallies.

TDD: written before implementation. Pure Python, no MuJoCo, no model
loading -- covers per-skill-pair truncation stats used to diagnose whether
imbalance comes from particular skill pairs.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nash_skills.v2.inspect_truncated_rallies import (
    build_pair_stats,
    summarize_overall,
)


def _rally(skill1, skill2, winner, n_states):
    return {
        "skill1": skill1,
        "skill2": skill2,
        "winner": winner,
        "states": list(range(n_states)),
    }


class TestSummarizeOverall:
    def test_counts_decided_and_truncated(self):
        rallies = [
            _rally("left", "right", 1, 5),
            _rally("left", "right", 2, 6),
            _rally("left", "right", 0, 8),
            _rally("left", "right", None, 9),
        ]
        s = summarize_overall(rallies)
        assert s["total"] == 4
        assert s["decided"] == 2
        assert s["truncated"] == 2

    def test_truncation_rate(self):
        rallies = [_rally("left", "right", 1, 5)] * 3 + [_rally("left", "right", 0, 5)] * 1
        s = summarize_overall(rallies)
        assert abs(s["truncation_rate"] - 0.25) < 1e-9

    def test_empty_list(self):
        s = summarize_overall([])
        assert s["total"] == 0
        assert s["truncation_rate"] is None


class TestBuildPairStats:
    def test_groups_by_skill_pair(self):
        rallies = [
            _rally("left", "right", 1, 5),
            _rally("left", "right", 2, 6),
            _rally("center_safe", "center_safe", 0, 10),
        ]
        stats = build_pair_stats(rallies)
        pairs = {(r["skill1"], r["skill2"]) for r in stats}
        assert ("left", "right") in pairs
        assert ("center_safe", "center_safe") in pairs

    def test_per_pair_truncation_rate(self):
        rallies = (
            [_rally("left", "right", 1, 5)] * 2
            + [_rally("left", "right", 0, 20)] * 2
        )
        stats = build_pair_stats(rallies)
        row = next(r for r in stats if r["skill1"] == "left" and r["skill2"] == "right")
        assert row["total"] == 4
        assert row["truncated"] == 2
        assert abs(row["truncation_rate"] - 0.5) < 1e-9

    def test_avg_rally_length_uses_states_len(self):
        rallies = [_rally("left", "left", 1, 4), _rally("left", "left", 2, 6)]
        stats = build_pair_stats(rallies)
        row = stats[0]
        assert abs(row["avg_rally_length"] - 5.0) < 1e-9

    def test_win_rate_over_decided_only(self):
        rallies = (
            [_rally("left", "left", 1, 5)] * 3
            + [_rally("left", "left", 2, 5)] * 1
            + [_rally("left", "left", 0, 5)] * 10  # truncated, excluded from win_rate
        )
        stats = build_pair_stats(rallies)
        row = stats[0]
        assert abs(row["win_rate"] - 0.75) < 1e-9

    def test_win_rate_none_when_no_decided(self):
        rallies = [_rally("left", "left", 0, 5)] * 3
        stats = build_pair_stats(rallies)
        row = stats[0]
        assert row["win_rate"] is None

    def test_none_winner_treated_as_truncated(self):
        rallies = [_rally("left", "left", None, 5), _rally("left", "left", 1, 5)]
        stats = build_pair_stats(rallies)
        row = stats[0]
        assert row["truncated"] == 1
        assert row["total"] == 2

    def test_sorted_by_truncation_rate_descending(self):
        rallies = (
            [_rally("left", "left", 1, 5)] * 10          # 0% truncation
            + [_rally("right", "right", 0, 5)] * 8         # 100% truncation
        )
        stats = build_pair_stats(rallies)
        assert stats[0]["skill1"] == "right"  # highest truncation first
        assert stats[0]["truncation_rate"] == 1.0

    def test_empty_rallies_returns_empty_list(self):
        assert build_pair_stats([]) == []
