"""Tests for nash_skills.v2.collect_data.make_skill_pair_schedule.

TDD: written before implementation. Pure function, no MuJoCo, no PPO --
covers the schedule-generation logic that drives which (skill1, skill2)
pairs collect_data.py collects rallies for. 'grid' preserves the existing
exhaustive-sweep default; 'random' is random-vs-random; 'fixed_random' is
fixed-vs-random (one player locked to a chosen skill, the other randomized
per rally).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import itertools

import numpy as np
import pytest

from nash_skills.v2.collect_data import make_skill_pair_schedule
from nash_skills.skills import SKILL_NAMES


class TestGridMode:
    def test_returns_all_pairs(self):
        schedule = make_skill_pair_schedule("grid", SKILL_NAMES, rallies_per_pair=50)
        pairs = [(s1, s2) for s1, s2, _ in schedule]
        assert pairs == list(itertools.product(SKILL_NAMES, SKILL_NAMES))

    def test_block_count_matches_rallies_per_pair(self):
        schedule = make_skill_pair_schedule("grid", SKILL_NAMES, rallies_per_pair=30)
        assert all(count == 30 for _, _, count in schedule)

    def test_total_blocks_is_n_squared(self):
        schedule = make_skill_pair_schedule("grid", SKILL_NAMES, rallies_per_pair=50)
        assert len(schedule) == len(SKILL_NAMES) ** 2

    def test_deterministic_across_calls(self):
        s1 = make_skill_pair_schedule("grid", SKILL_NAMES, rallies_per_pair=50)
        s2 = make_skill_pair_schedule("grid", SKILL_NAMES, rallies_per_pair=50)
        assert s1 == s2

    def test_ignores_rng_and_total_rallies(self):
        rng = np.random.default_rng(0)
        s1 = make_skill_pair_schedule("grid", SKILL_NAMES, rallies_per_pair=50, rng=rng)
        s2 = make_skill_pair_schedule("grid", SKILL_NAMES, rallies_per_pair=50, total_rallies=999)
        assert s1 == s2


class TestRandomMode:
    def test_default_total_matches_grid_budget(self):
        schedule = make_skill_pair_schedule(
            "random", SKILL_NAMES, rallies_per_pair=50, rng=np.random.default_rng(0)
        )
        assert len(schedule) == len(SKILL_NAMES) ** 2 * 50

    def test_explicit_total_rallies_overrides_default(self):
        schedule = make_skill_pair_schedule(
            "random", SKILL_NAMES, rallies_per_pair=50,
            total_rallies=100, rng=np.random.default_rng(0),
        )
        assert len(schedule) == 100

    def test_each_block_has_count_one(self):
        schedule = make_skill_pair_schedule(
            "random", SKILL_NAMES, rallies_per_pair=1,
            total_rallies=200, rng=np.random.default_rng(0),
        )
        assert all(count == 1 for _, _, count in schedule)

    def test_skills_drawn_from_valid_names(self):
        schedule = make_skill_pair_schedule(
            "random", SKILL_NAMES, rallies_per_pair=1,
            total_rallies=200, rng=np.random.default_rng(0),
        )
        for s1, s2, _ in schedule:
            assert s1 in SKILL_NAMES
            assert s2 in SKILL_NAMES

    def test_reproducible_with_same_seed(self):
        s1 = make_skill_pair_schedule(
            "random", SKILL_NAMES, rallies_per_pair=1,
            total_rallies=200, rng=np.random.default_rng(42),
        )
        s2 = make_skill_pair_schedule(
            "random", SKILL_NAMES, rallies_per_pair=1,
            total_rallies=200, rng=np.random.default_rng(42),
        )
        assert s1 == s2

    def test_different_seeds_produce_different_schedules(self):
        s1 = make_skill_pair_schedule(
            "random", SKILL_NAMES, rallies_per_pair=1,
            total_rallies=200, rng=np.random.default_rng(1),
        )
        s2 = make_skill_pair_schedule(
            "random", SKILL_NAMES, rallies_per_pair=1,
            total_rallies=200, rng=np.random.default_rng(2),
        )
        assert s1 != s2

    def test_both_same_and_different_pairs_can_occur(self):
        """Over enough draws, ego and opp should sometimes match and sometimes differ."""
        schedule = make_skill_pair_schedule(
            "random", SKILL_NAMES, rallies_per_pair=1,
            total_rallies=500, rng=np.random.default_rng(0),
        )
        same = sum(1 for s1, s2, _ in schedule if s1 == s2)
        diff = sum(1 for s1, s2, _ in schedule if s1 != s2)
        assert same > 0
        assert diff > 0

    def test_uses_fresh_default_rng_when_none_given(self):
        # Should not raise, and should still respect total_rallies.
        schedule = make_skill_pair_schedule(
            "random", SKILL_NAMES, rallies_per_pair=1, total_rallies=10
        )
        assert len(schedule) == 10


class TestFixedRandomMode:
    def test_default_total_matches_grid_budget(self):
        schedule = make_skill_pair_schedule(
            "fixed_random", SKILL_NAMES, rallies_per_pair=50,
            fixed_player=1, fixed_skill="left", rng=np.random.default_rng(0),
        )
        assert len(schedule) == len(SKILL_NAMES) ** 2 * 50

    def test_explicit_total_rallies_overrides_default(self):
        schedule = make_skill_pair_schedule(
            "fixed_random", SKILL_NAMES, rallies_per_pair=50,
            total_rallies=100, fixed_player=1, fixed_skill="left",
            rng=np.random.default_rng(0),
        )
        assert len(schedule) == 100

    def test_each_block_has_count_one(self):
        schedule = make_skill_pair_schedule(
            "fixed_random", SKILL_NAMES, rallies_per_pair=1,
            total_rallies=200, fixed_player=1, fixed_skill="left",
            rng=np.random.default_rng(0),
        )
        assert all(count == 1 for _, _, count in schedule)

    def test_player1_fixed_keeps_skill1_constant(self):
        schedule = make_skill_pair_schedule(
            "fixed_random", SKILL_NAMES, rallies_per_pair=1,
            total_rallies=200, fixed_player=1, fixed_skill="center_safe",
            rng=np.random.default_rng(0),
        )
        assert all(s1 == "center_safe" for s1, s2, _ in schedule)

    def test_player1_fixed_randomizes_skill2(self):
        schedule = make_skill_pair_schedule(
            "fixed_random", SKILL_NAMES, rallies_per_pair=1,
            total_rallies=200, fixed_player=1, fixed_skill="center_safe",
            rng=np.random.default_rng(0),
        )
        distinct_s2 = {s2 for _, s2, _ in schedule}
        assert len(distinct_s2) > 1

    def test_player2_fixed_keeps_skill2_constant(self):
        schedule = make_skill_pair_schedule(
            "fixed_random", SKILL_NAMES, rallies_per_pair=1,
            total_rallies=200, fixed_player=2, fixed_skill="right",
            rng=np.random.default_rng(0),
        )
        assert all(s2 == "right" for s1, s2, _ in schedule)

    def test_player2_fixed_randomizes_skill1(self):
        schedule = make_skill_pair_schedule(
            "fixed_random", SKILL_NAMES, rallies_per_pair=1,
            total_rallies=200, fixed_player=2, fixed_skill="right",
            rng=np.random.default_rng(0),
        )
        distinct_s1 = {s1 for s1, s2, _ in schedule}
        assert len(distinct_s1) > 1

    def test_randomized_skill_drawn_from_valid_names(self):
        schedule = make_skill_pair_schedule(
            "fixed_random", SKILL_NAMES, rallies_per_pair=1,
            total_rallies=200, fixed_player=1, fixed_skill="left",
            rng=np.random.default_rng(0),
        )
        for _, s2, _ in schedule:
            assert s2 in SKILL_NAMES

    def test_reproducible_with_same_seed(self):
        s1 = make_skill_pair_schedule(
            "fixed_random", SKILL_NAMES, rallies_per_pair=1,
            total_rallies=200, fixed_player=1, fixed_skill="left",
            rng=np.random.default_rng(42),
        )
        s2 = make_skill_pair_schedule(
            "fixed_random", SKILL_NAMES, rallies_per_pair=1,
            total_rallies=200, fixed_player=1, fixed_skill="left",
            rng=np.random.default_rng(42),
        )
        assert s1 == s2

    def test_missing_fixed_player_raises_value_error(self):
        with pytest.raises(ValueError, match="fixed_player"):
            make_skill_pair_schedule(
                "fixed_random", SKILL_NAMES, rallies_per_pair=1,
                total_rallies=10, fixed_skill="left",
                rng=np.random.default_rng(0),
            )

    def test_missing_fixed_skill_raises_value_error(self):
        with pytest.raises(ValueError, match="fixed_skill"):
            make_skill_pair_schedule(
                "fixed_random", SKILL_NAMES, rallies_per_pair=1,
                total_rallies=10, fixed_player=1,
                rng=np.random.default_rng(0),
            )

    def test_invalid_fixed_player_raises_value_error(self):
        with pytest.raises(ValueError, match="fixed_player"):
            make_skill_pair_schedule(
                "fixed_random", SKILL_NAMES, rallies_per_pair=1,
                total_rallies=10, fixed_player=3, fixed_skill="left",
                rng=np.random.default_rng(0),
            )

    def test_invalid_fixed_skill_raises_value_error(self):
        with pytest.raises(ValueError, match="fixed_skill"):
            make_skill_pair_schedule(
                "fixed_random", SKILL_NAMES, rallies_per_pair=1,
                total_rallies=10, fixed_player=1, fixed_skill="not_a_skill",
                rng=np.random.default_rng(0),
            )

    def test_uses_fresh_default_rng_when_none_given(self):
        schedule = make_skill_pair_schedule(
            "fixed_random", SKILL_NAMES, rallies_per_pair=1,
            total_rallies=10, fixed_player=1, fixed_skill="left",
        )
        assert len(schedule) == 10


class TestInvalidMode:
    def test_unknown_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="mode"):
            make_skill_pair_schedule("bogus", SKILL_NAMES, rallies_per_pair=50)
