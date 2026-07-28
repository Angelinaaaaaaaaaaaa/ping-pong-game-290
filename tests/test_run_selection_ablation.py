"""Tests for nash_skills.v2.run_selection_ablation -- pure logic only.

TDD: written before implementation. Covers the two pure functions that
drive the ablation runner: matching a rally file to a model by state_dim,
and summarizing a probability vector into a dominant-skill / skew report.
Model loading and forward passes are integration behavior (already
exercised via nash_skills/v2/diag_skill_selection_probs.py) and are not
re-tested here.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from nash_skills.v2.run_selection_ablation import (
    match_rally_to_model,
    compute_skew_summary,
)


class TestMatchRallyToModel:
    def test_exact_match_returns_path(self):
        dim_map = {"a.pkl": 76, "b.pkl": 12}
        assert match_rally_to_model(76, dim_map) == "a.pkl"

    def test_no_match_returns_none(self):
        dim_map = {"a.pkl": 76, "b.pkl": 12}
        assert match_rally_to_model(116, dim_map) is None

    def test_multiple_matches_returns_alphabetically_first(self):
        dim_map = {"z.pkl": 76, "a.pkl": 76, "m.pkl": 76}
        assert match_rally_to_model(76, dim_map) == "a.pkl"

    def test_empty_map_returns_none(self):
        assert match_rally_to_model(76, {}) is None


class TestComputeSkewSummary:
    def test_dominant_skill_identified(self):
        probs = np.array([0.1, 0.6, 0.1, 0.1, 0.1])
        skills = ["left", "left_short", "center_safe", "right_short", "right"]
        result = compute_skew_summary(probs, skills)
        assert result["dominant"] == "left_short"

    def test_max_min_ratio_computed(self):
        probs = np.array([0.1, 0.4])
        skills = ["a", "b"]
        result = compute_skew_summary(probs, skills)
        assert abs(result["max_min_ratio"] - 4.0) < 1e-6

    def test_uniform_distribution_ratio_near_one(self):
        probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        skills = ["a", "b", "c", "d", "e"]
        result = compute_skew_summary(probs, skills)
        assert abs(result["max_min_ratio"] - 1.0) < 1e-6

    def test_zero_probability_does_not_raise(self):
        probs = np.array([1.0, 0.0, 0.0])
        skills = ["a", "b", "c"]
        result = compute_skew_summary(probs, skills)
        assert result["dominant"] == "a"
        assert result["max_min_ratio"] > 0  # finite due to epsilon guard

    def test_returns_dict_with_expected_keys(self):
        probs = np.array([0.5, 0.5])
        skills = ["a", "b"]
        result = compute_skew_summary(probs, skills)
        assert set(result.keys()) == {"dominant", "max_min_ratio"}
