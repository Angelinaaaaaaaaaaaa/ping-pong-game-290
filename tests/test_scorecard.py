"""Tests for nash_skills.v2.scorecard.

TDD: written before implementation. Pure Python -- a shared evaluation
scorecard (win rate, truncation rate, rally-length stats, skill entropy,
dominant-skill fraction, per-skill usage) that both eval_matchup.py (main)
and eval_selfplay_5skill.py (origin/feature-selfplay) can call, so every
future eval run reports the same complete picture (meeting note item 19).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from nash_skills.v2.scorecard import (
    compute_scorecard,
    skill_entropy,
    skill_entropy_normalized,
    dominant_skill_info,
    format_scorecard,
)


class TestSkillEntropy:
    def test_uniform_usage_is_max_entropy(self):
        usage = {"a": 10, "b": 10, "c": 10, "d": 10}
        assert skill_entropy_normalized(usage) == pytest.approx(1.0, abs=1e-6)

    def test_single_skill_is_zero_entropy(self):
        usage = {"a": 10, "b": 0, "c": 0}
        assert skill_entropy_normalized(usage) == pytest.approx(0.0, abs=1e-6)

    def test_empty_usage_returns_none(self):
        assert skill_entropy({}) is None
        assert skill_entropy_normalized({}) is None

    def test_single_key_normalized_returns_none(self):
        # log2(1) == 0 -- normalization is undefined with only one possible skill
        assert skill_entropy_normalized({"a": 5}) is None


class TestDominantSkillInfo:
    def test_finds_dominant_skill_and_fraction(self):
        skill, frac = dominant_skill_info({"left": 8, "right": 2})
        assert skill == "left"
        assert frac == pytest.approx(0.8)

    def test_empty_usage_returns_none_none(self):
        skill, frac = dominant_skill_info({})
        assert skill is None
        assert frac is None

    def test_all_zero_usage_returns_none_none(self):
        skill, frac = dominant_skill_info({"a": 0, "b": 0})
        assert skill is None
        assert frac is None


class TestComputeScorecard:
    def test_win_rate_decided_only(self):
        sc = compute_scorecard(wins=3, losses=1, truncated=6,
                                rally_lengths=[5] * 10, skill_usage={"left": 10})
        assert sc["win_rate"] == pytest.approx(0.75)

    def test_win_rate_none_when_no_decided(self):
        sc = compute_scorecard(wins=0, losses=0, truncated=5,
                                rally_lengths=[1] * 5, skill_usage={})
        assert sc["win_rate"] is None

    def test_truncation_rate(self):
        sc = compute_scorecard(wins=6, losses=2, truncated=2,
                                rally_lengths=[1] * 10, skill_usage={"left": 10})
        assert sc["truncation_rate"] == pytest.approx(0.2)

    def test_avg_and_median_rally_length(self):
        sc = compute_scorecard(wins=1, losses=0, truncated=0,
                                rally_lengths=[2, 4, 6], skill_usage={"left": 1})
        assert sc["avg_rally_length"] == pytest.approx(4.0)
        assert sc["median_rally_length"] == pytest.approx(4.0)

    def test_no_rally_lengths_gives_none(self):
        sc = compute_scorecard(wins=0, losses=0, truncated=0,
                                rally_lengths=[], skill_usage={})
        assert sc["avg_rally_length"] is None
        assert sc["median_rally_length"] is None

    def test_dominant_skill_and_fraction(self):
        sc = compute_scorecard(wins=1, losses=0, truncated=0, rally_lengths=[1],
                                skill_usage={"left": 8, "right": 2})
        assert sc["dominant_skill"] == "left"
        assert sc["dominant_skill_fraction"] == pytest.approx(0.8)

    def test_skill_entropy_uniform_is_near_max(self):
        sc = compute_scorecard(wins=1, losses=0, truncated=0, rally_lengths=[1],
                                skill_usage={"a": 10, "b": 10, "c": 10, "d": 10, "e": 10})
        assert sc["skill_entropy_normalized"] == pytest.approx(1.0, abs=1e-6)

    def test_skill_entropy_single_skill_is_zero(self):
        sc = compute_scorecard(wins=1, losses=0, truncated=0, rally_lengths=[1],
                                skill_usage={"a": 10, "b": 0, "c": 0})
        assert sc["skill_entropy_normalized"] == pytest.approx(0.0, abs=1e-6)

    def test_empty_skill_usage_handled(self):
        sc = compute_scorecard(wins=0, losses=0, truncated=0, rally_lengths=[], skill_usage={})
        assert sc["dominant_skill"] is None
        assert sc["skill_entropy_normalized"] is None

    def test_skill_usage_frac_sums_to_one(self):
        sc = compute_scorecard(wins=1, losses=0, truncated=0, rally_lengths=[1],
                                skill_usage={"a": 3, "b": 1})
        assert sum(sc["skill_usage_frac"].values()) == pytest.approx(1.0)


class TestFormatScorecard:
    def test_returns_string_with_key_fields(self):
        sc = compute_scorecard(wins=3, losses=1, truncated=1, rally_lengths=[1, 2, 3, 4, 5],
                                skill_usage={"left": 5})
        s = format_scorecard(sc, label="vs random")
        assert "vs random" in s
        assert "win_rate" in s

    def test_handles_none_fields_gracefully(self):
        sc = compute_scorecard(wins=0, losses=0, truncated=0, rally_lengths=[], skill_usage={})
        s = format_scorecard(sc)
        assert "---" in s
