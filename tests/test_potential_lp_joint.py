"""nash_skills.v2.potential_lp_joint: paper Eq. 11 solved exactly at one state (Kexin's V3 fit, v3/potential_lp.py,
generalised from her 5 x 3 table to the 15 x 15 joint profiles)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills.v2.potential_lp_joint import max_residual, solve_potential_lp


def test_exact_potential_game_has_zero_alpha_and_recovers_the_potential():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(15, 15))
    q1 = p + rng.normal(size=(1, 15))   # player 1's payoff = potential + a term player 1 cannot change
    q2 = p + rng.normal(size=(15, 1))
    phi, alpha = solve_potential_lp(q1, q2)
    assert alpha == pytest.approx(0.0, abs=1e-7)
    np.testing.assert_allclose(phi, p - p[0, 0], atol=1e-6)
    assert phi[0, 0] == 0.0


def test_matching_pennies_needs_alpha_two():
    m = np.array([[1.0, -1.0], [-1.0, 1.0]])
    phi, alpha = solve_potential_lp(m, -m)
    assert alpha == pytest.approx(2.0, abs=1e-7)
    assert max_residual(phi, m, -m) == pytest.approx(2.0, abs=1e-7)


def test_solution_respects_its_alpha_on_a_random_game():
    rng = np.random.default_rng(1)
    q1 = rng.normal(size=(15, 15))
    phi, alpha = solve_potential_lp(q1, -q1)
    assert alpha > 0 and max_residual(phi, q1, -q1) <= alpha + 1e-6
