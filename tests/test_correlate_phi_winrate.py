"""Tests for nash_skills.v2.correlate_phi_winrate.

TDD: written before implementation. Pure Python (no MuJoCo, no real model
checkpoint needed for these unit tests -- phi_fn is a plain callable so the
aggregation/statistics logic is testable without torch model loading).
Cross-references the learned potential Phi against the EMPIRICAL win-rate
heatmap already available from plot_winrate_heatmap.py (meeting note item 11,
offline half).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills.v2.correlate_phi_winrate import (
    build_avg_phi_matrix,
    pearson_correlation_masked,
    correlate_phi_with_winrate,
)


def _rally(states):
    return {"skill1": "left", "skill2": "right", "winner": 1, "states": states}


class TestBuildAvgPhiMatrix:
    def test_averages_across_sampled_states(self):
        # phi_fn returns a table equal to state[0] broadcast -- so the
        # average over two rallies each with one state (values 2.0 and 4.0)
        # should be the (N,N) table filled with 3.0.
        rallies = [
            _rally([np.array([2.0, 0.0], dtype=np.float32)]),
            _rally([np.array([4.0, 0.0], dtype=np.float32)]),
        ]

        def phi_fn(state_vec):
            return np.full((3, 3), state_vec[0], dtype=np.float32)

        avg = build_avg_phi_matrix(rallies, phi_fn=phi_fn, state_dim=2,
                                    n_samples=10, rng=np.random.default_rng(0))
        assert avg.shape == (3, 3)
        assert np.allclose(avg, 3.0)

    def test_skips_states_with_wrong_dim(self):
        rallies = [
            _rally([np.array([1.0, 2.0, 3.0], dtype=np.float32)]),  # wrong dim
            _rally([np.array([9.0, 0.0], dtype=np.float32)]),        # right dim
        ]

        def phi_fn(state_vec):
            return np.full((2, 2), state_vec[0], dtype=np.float32)

        avg = build_avg_phi_matrix(rallies, phi_fn=phi_fn, state_dim=2,
                                    n_samples=10, rng=np.random.default_rng(0))
        assert np.allclose(avg, 9.0)

    def test_respects_n_samples_cap(self):
        calls = []

        def phi_fn(state_vec):
            calls.append(state_vec)
            return np.zeros((2, 2))

        rallies = [_rally([np.array([float(i), 0.0], dtype=np.float32)]) for i in range(50)]
        build_avg_phi_matrix(rallies, phi_fn=phi_fn, state_dim=2,
                              n_samples=5, rng=np.random.default_rng(0))
        assert len(calls) == 5

    def test_no_matching_states_returns_none(self):
        rallies = [_rally([np.array([1.0, 2.0, 3.0], dtype=np.float32)])]

        def phi_fn(state_vec):
            return np.zeros((2, 2))

        result = build_avg_phi_matrix(rallies, phi_fn=phi_fn, state_dim=2,
                                       n_samples=5, rng=np.random.default_rng(0))
        assert result is None

    def test_empty_rallies_returns_none(self):
        result = build_avg_phi_matrix([], phi_fn=lambda s: np.zeros((2, 2)),
                                       state_dim=2, n_samples=5)
        assert result is None


class TestPearsonCorrelationMasked:
    def test_perfect_positive_correlation(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[10.0, 20.0], [30.0, 40.0]])
        assert pearson_correlation_masked(a, b) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = -a
        assert pearson_correlation_masked(a, b) == pytest.approx(-1.0)

    def test_nan_cells_are_masked_out(self):
        a = np.array([[1.0, 2.0], [3.0, np.nan]])
        b = np.array([[10.0, 20.0], [30.0, 999.0]])
        r = pearson_correlation_masked(a, b)
        assert r == pytest.approx(1.0)

    def test_too_few_valid_pairs_returns_none(self):
        a = np.array([[1.0, np.nan], [np.nan, np.nan]])
        b = np.array([[10.0, np.nan], [np.nan, np.nan]])
        assert pearson_correlation_masked(a, b) is None

    def test_constant_array_returns_none(self):
        a = np.array([[1.0, 1.0], [1.0, 1.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        assert pearson_correlation_masked(a, b) is None


class TestCorrelatePhiWithWinrate:
    def test_reports_pearson_r_and_cells(self):
        avg_phi = np.array([[0.1, 0.9], [0.9, 0.1]])
        winrate = np.array([[0.2, 0.8], [0.8, 0.2]])
        result = correlate_phi_with_winrate(avg_phi, winrate, ["a", "b"])
        assert result["pearson_r"] == pytest.approx(1.0)
        assert result["n_cells"] == 4

    def test_excludes_nan_winrate_cells(self):
        avg_phi = np.array([[0.1, 0.9], [0.9, 0.1]])
        winrate = np.array([[0.2, np.nan], [0.8, 0.2]])
        result = correlate_phi_with_winrate(avg_phi, winrate, ["a", "b"])
        assert result["n_cells"] == 3

    def test_cell_contents(self):
        avg_phi = np.array([[0.5]])
        winrate = np.array([[0.7]])
        result = correlate_phi_with_winrate(avg_phi, winrate, ["only"])
        cell = result["cells"][0]
        assert cell["ego_skill"] == "only"
        assert cell["opp_skill"] == "only"
        assert cell["phi"] == pytest.approx(0.5)
        assert cell["win_rate"] == pytest.approx(0.7)
