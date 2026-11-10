"""Zero-sum matrix games: mixed equilibrium by linear programming."""
import numpy as np
from scipy.optimize import linprog

__all__ = ["solve_zero_sum"]


def _maximin(M):
    """Row player's maximin mix for payoff matrix M (row player maximizes)."""
    m, n = M.shape
    c = np.zeros(m + 1)
    c[-1] = -1.0                                   # maximize v
    a_ub = np.hstack([-M.T, np.ones((n, 1))])      # v - p^T M[:, j] <= 0 for every column j
    a_eq = np.hstack([np.ones((1, m)), np.zeros((1, 1))])
    bounds = [(0.0, 1.0)] * m + [(None, None)]
    res = linprog(c, A_ub=a_ub, b_ub=np.zeros(n), A_eq=a_eq, b_eq=[1.0], bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"zero-sum LP failed: {res.message}")
    p = np.clip(res.x[:m], 0.0, None)
    return p / p.sum(), float(res.x[-1])


def solve_zero_sum(M):
    """(row mix, column mix, value) of the zero-sum game where the row player gets M[i, j]."""
    M = np.asarray(M, dtype=float)
    p, value = _maximin(M)
    q, _ = _maximin(-M.T)
    return p, q, value
