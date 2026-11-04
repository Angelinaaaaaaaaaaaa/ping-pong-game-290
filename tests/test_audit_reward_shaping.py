"""Tests for nash_skills.v2.audit_reward_shaping.

TDD: written before implementation. Pure Python, no MuJoCo -- quantifies how
much of self-play's total reward magnitude comes from per-crossing shaping
vs. the terminal win/loss signal (meeting items 6 and 13), using the
SHAPING_COEF=0.05 constant mirrored from selfplay_5skill.py
(origin/feature-selfplay) and the fact that
net_crossings == len(rally['states']) (see nash_skills/v2/collect_data.py,
where `net_crossings = len(curr_states)`).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from nash_skills.v2.audit_reward_shaping import (
    shaped_reward,
    compute_rally_shaping_stats,
    SHAPING_COEF,
    TERMINAL_MAGNITUDE,
    _validate_rally_schema,
)


def _rally(winner, n_states):
    return {
        "skill1": "left", "skill2": "right", "winner": winner,
        "states": [object()] * n_states,
    }


class TestShapedReward:
    def test_zero_crossings_gives_zero(self):
        assert shaped_reward(0) == 0.0

    def test_scales_linearly_with_crossings(self):
        assert shaped_reward(10) == pytest.approx(10 * SHAPING_COEF)

    def test_custom_coef(self):
        assert shaped_reward(4, shaping_coef=0.1) == pytest.approx(0.4)


class TestComputeRallyShapingStats:
    def test_truncated_rally_shaping_equals_full_reward(self):
        # Truncated (winner not in (1,2)) -> ego_terminal=0, so shaped IS the
        # entire reward magnitude for both players (per selfplay_5skill.py:
        # ego_reward = shaped + ego_terminal, ego_terminal=0 when not done).
        rallies = [_rally(0, 20)]
        stats = compute_rally_shaping_stats(rallies)
        assert stats["truncated_count"] == 1
        assert stats["decided_count"] == 0
        assert stats["truncated_shaped_rewards"][0] == pytest.approx(20 * SHAPING_COEF)

    def test_decided_rally_ratio_computed_against_terminal_magnitude(self):
        rallies = [_rally(1, 20)]  # decided, 20 crossings
        stats = compute_rally_shaping_stats(rallies)
        assert stats["decided_count"] == 1
        expected_ratio = (20 * SHAPING_COEF) / TERMINAL_MAGNITUDE
        assert stats["decided_shaping_ratios"][0] == pytest.approx(expected_ratio)

    def test_fraction_shaping_dominates_decided(self):
        # 20 crossings -> shaped=1.0 == terminal magnitude -> ratio=1.0 -> dominates (>=1.0)
        # 5 crossings  -> shaped=0.25 -> does not dominate
        rallies = [_rally(1, 20), _rally(2, 5)]
        stats = compute_rally_shaping_stats(rallies)
        assert stats["frac_decided_shaping_dominates"] == pytest.approx(0.5)

    def test_none_and_missing_winner_treated_as_truncated(self):
        rallies = [_rally(None, 3), {"skill1": "left", "skill2": "right", "states": [object()] * 3}]
        stats = compute_rally_shaping_stats(rallies)
        assert stats["truncated_count"] == 2
        assert stats["decided_count"] == 0

    def test_empty_rallies(self):
        stats = compute_rally_shaping_stats([])
        assert stats["truncated_count"] == 0
        assert stats["decided_count"] == 0
        assert stats["frac_decided_shaping_dominates"] is None
        assert stats["decided_ratio_mean"] is None
        assert stats["truncated_shaped_mean"] is None

    def test_summary_percentiles_present(self):
        rallies = [_rally(1, n) for n in (1, 5, 10, 20, 40)]
        stats = compute_rally_shaping_stats(rallies)
        assert "decided_ratio_mean" in stats
        assert "decided_ratio_median" in stats
        assert "decided_ratio_p90" in stats
        assert "decided_ratio_p99" in stats

    def test_custom_shaping_coef_changes_ratio(self):
        rallies = [_rally(1, 10)]
        stats = compute_rally_shaping_stats(rallies, shaping_coef=0.1)
        assert stats["decided_shaping_ratios"][0] == pytest.approx(1.0)


class TestValidateRallySchema:
    def test_accepts_dict_entries(self):
        _validate_rally_schema([_rally(1, 3)], "some_file.pkl")

    def test_rejects_non_dict_first_entry(self):
        with pytest.raises(ValueError, match="rally-dict schema"):
            _validate_rally_schema([[1, 2, 3]], "rallies.pkl")

    def test_empty_list_does_not_raise(self):
        _validate_rally_schema([], "empty.pkl")
