"""
TDD tests for nash_skills/v2/plot_winrate_heatmap.py

Tests written FIRST (RED phase) before implementation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills.v2.plot_winrate_heatmap import build_winrate_matrix, build_counts_matrix
from nash_skills.skills import SKILL_NAMES, N_SKILLS


# ─── helpers ─────────────────────────────────────────────────────────────────

def _rally(skill1, skill2, winner):
    return {"skill1": skill1, "skill2": skill2, "winner": winner}


# ─── build_counts_matrix ─────────────────────────────────────────────────────

class TestBuildCountsMatrix:
    def test_returns_two_NxN_arrays(self):
        ego_w, total = build_counts_matrix([], SKILL_NAMES)
        assert ego_w.shape == (N_SKILLS, N_SKILLS)
        assert total.shape == (N_SKILLS, N_SKILLS)

    def test_all_zero_for_empty_rallies(self):
        ego_w, total = build_counts_matrix([], SKILL_NAMES)
        assert ego_w.sum() == 0
        assert total.sum() == 0

    def test_ego_win_increments_ego_and_total(self):
        rallies = [_rally("left", "right", winner=1)]
        ego_w, total = build_counts_matrix(rallies, SKILL_NAMES)
        r, c = SKILL_NAMES.index("left"), SKILL_NAMES.index("right")
        assert ego_w[r, c] == 1
        assert total[r, c] == 1

    def test_opp_win_increments_total_not_ego(self):
        rallies = [_rally("left", "right", winner=2)]
        ego_w, total = build_counts_matrix(rallies, SKILL_NAMES)
        r, c = SKILL_NAMES.index("left"), SKILL_NAMES.index("right")
        assert ego_w[r, c] == 0
        assert total[r, c] == 1

    def test_truncated_winner_zero_not_counted(self):
        rallies = [_rally("left", "right", winner=0)]
        ego_w, total = build_counts_matrix(rallies, SKILL_NAMES)
        assert total.sum() == 0

    def test_truncated_winner_none_not_counted(self):
        rallies = [_rally("left", "right", winner=None)]
        ego_w, total = build_counts_matrix(rallies, SKILL_NAMES)
        assert total.sum() == 0

    def test_multiple_rallies_accumulate(self):
        rallies = [
            _rally("left", "right", winner=1),
            _rally("left", "right", winner=1),
            _rally("left", "right", winner=2),
        ]
        ego_w, total = build_counts_matrix(rallies, SKILL_NAMES)
        r, c = SKILL_NAMES.index("left"), SKILL_NAMES.index("right")
        assert ego_w[r, c] == 2
        assert total[r, c] == 3

    def test_different_pairs_go_to_different_cells(self):
        rallies = [
            _rally("left", "right", winner=1),
            _rally("right", "left", winner=2),
        ]
        ego_w, total = build_counts_matrix(rallies, SKILL_NAMES)
        lr_r, lr_c = SKILL_NAMES.index("left"), SKILL_NAMES.index("right")
        rl_r, rl_c = SKILL_NAMES.index("right"), SKILL_NAMES.index("left")
        assert total[lr_r, lr_c] == 1
        assert total[rl_r, rl_c] == 1

    def test_diagonal_same_skill_pair(self):
        rallies = [_rally("center_safe", "center_safe", winner=1)]
        ego_w, total = build_counts_matrix(rallies, SKILL_NAMES)
        i = SKILL_NAMES.index("center_safe")
        assert total[i, i] == 1
        assert ego_w[i, i] == 1


# ─── build_winrate_matrix ─────────────────────────────────────────────────────

class TestBuildWinrateMatrix:
    def test_returns_NxN_float_array(self):
        matrix = build_winrate_matrix([], SKILL_NAMES)
        assert matrix.shape == (N_SKILLS, N_SKILLS)
        assert matrix.dtype in (np.float32, np.float64)

    def test_all_nan_for_empty_rallies(self):
        matrix = build_winrate_matrix([], SKILL_NAMES)
        assert np.all(np.isnan(matrix))

    def test_perfect_ego_win_gives_1_0(self):
        rallies = [_rally("left", "right", winner=1)] * 5
        matrix = build_winrate_matrix(rallies, SKILL_NAMES)
        r, c = SKILL_NAMES.index("left"), SKILL_NAMES.index("right")
        assert matrix[r, c] == pytest.approx(1.0)

    def test_perfect_opp_win_gives_0_0(self):
        rallies = [_rally("left", "right", winner=2)] * 3
        matrix = build_winrate_matrix(rallies, SKILL_NAMES)
        r, c = SKILL_NAMES.index("left"), SKILL_NAMES.index("right")
        assert matrix[r, c] == pytest.approx(0.0)

    def test_equal_wins_gives_0_5(self):
        rallies = [_rally("left", "right", winner=1),
                   _rally("left", "right", winner=2)]
        matrix = build_winrate_matrix(rallies, SKILL_NAMES)
        r, c = SKILL_NAMES.index("left"), SKILL_NAMES.index("right")
        assert matrix[r, c] == pytest.approx(0.5)

    def test_pair_with_no_rallies_is_nan(self):
        rallies = [_rally("left", "right", winner=1)]
        matrix = build_winrate_matrix(rallies, SKILL_NAMES)
        # "center_safe" vs "left_short" has no rallies → NaN
        r = SKILL_NAMES.index("center_safe")
        c = SKILL_NAMES.index("left_short")
        assert np.isnan(matrix[r, c])

    def test_truncated_only_pair_is_nan(self):
        rallies = [_rally("left", "right", winner=0),
                   _rally("left", "right", winner=None)]
        matrix = build_winrate_matrix(rallies, SKILL_NAMES)
        r, c = SKILL_NAMES.index("left"), SKILL_NAMES.index("right")
        assert np.isnan(matrix[r, c])

    def test_win_rate_range_0_to_1(self):
        rallies = [
            _rally(s1, s2, winner=w)
            for s1 in SKILL_NAMES
            for s2 in SKILL_NAMES
            for w in (1, 2)
        ]
        matrix = build_winrate_matrix(rallies, SKILL_NAMES)
        valid = matrix[~np.isnan(matrix)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_row_is_ego_skill_col_is_opp_skill(self):
        """Row index = ego skill index, col index = opp skill index."""
        rallies = [_rally("left_short", "center_safe", winner=1)]
        matrix = build_winrate_matrix(rallies, SKILL_NAMES)
        r = SKILL_NAMES.index("left_short")
        c = SKILL_NAMES.index("center_safe")
        assert matrix[r, c] == pytest.approx(1.0)
        # The transposed cell should be NaN (no rally for that pair)
        assert np.isnan(matrix[c, r])
