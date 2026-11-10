"""
Paper Eq. 11 solved exactly at one state, as Kexin's V3 pipeline does (v3/potential_lp.py), generalised from her
5 x 3 role-only table to the Design B joint game. Given the learned values of every joint profile at a state --
player 1's Q1 and player 2's Q2 over the n x n profiles (n = 15 (target, preparation) pairs per player) -- find
the potential table Phi, with Phi[0, 0] = 0, and the smallest alpha such that

    |[Phi(a) - Phi(b)] - [Q_i(a) - Q_i(b)]| <= alpha

for every unilateral deviation between profiles a and b (player 1 changes its pair while player 2 keeps its pair,
or the reverse; Q_i is the deviating player's value). A linear program in n^2 + 1 variables with two inequality
rows per deviation, 2 n^2 (n - 1) rows in total.
"""
import functools

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

__all__ = ["solve_potential_lp", "max_residual"]


@functools.lru_cache(maxsize=4)
def _deviations(n):
    """Flat-index pairs (a, b) and the deviating player for every unordered unilateral deviation."""
    a, b, who = [], [], []
    for j in range(n):  # player 1 changes its pair, player 2 stays at j
        for i in range(n):
            for i2 in range(i + 1, n):
                a.append(i * n + j)
                b.append(i2 * n + j)
                who.append(1)
    for i in range(n):  # player 2 changes its pair, player 1 stays at i
        for j in range(n):
            for j2 in range(j + 1, n):
                a.append(i * n + j)
                b.append(i * n + j2)
                who.append(2)
    return np.array(a), np.array(b), np.array(who)


@functools.lru_cache(maxsize=4)
def _program(n):
    """Constraint matrix, objective and bounds; only the right-hand side depends on the state."""
    a, b, _who = _deviations(n)
    m, alpha = len(a), n * n
    rows = np.repeat(np.arange(2 * m), 3)
    # row 2k:   Phi[a] - Phi[b] - alpha <= dQ_k      row 2k+1: -Phi[a] + Phi[b] - alpha <= -dQ_k
    cols = np.stack([a, b, np.full(m, alpha), a, b, np.full(m, alpha)], axis=1).reshape(-1)
    vals = np.tile([1.0, -1.0, -1.0, -1.0, 1.0, -1.0], m)
    matrix = csr_matrix((vals, (rows, cols)), shape=(2 * m, n * n + 1))
    objective = np.zeros(n * n + 1)
    objective[alpha] = 1.0
    bounds = [(0.0, 0.0)] + [(None, None)] * (n * n - 1) + [(0.0, None)]
    return matrix, objective, bounds


def _value_changes(q1, q2):
    a, b, who = _deviations(q1.shape[0])
    f1, f2 = q1.reshape(-1), q2.reshape(-1)
    return np.where(who == 1, f1[a] - f1[b], f2[a] - f2[b])


def _checked(q1, q2):
    q1, q2 = np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)
    if q1.ndim != 2 or q1.shape[0] != q1.shape[1] or q1.shape != q2.shape:
        raise ValueError(f"expected two equal square value tables, got {q1.shape} and {q2.shape}")
    return q1, q2


def solve_potential_lp(q1, q2):
    """(Phi table with Phi[0, 0] = 0, alpha): the potential with the smallest largest residual at this state."""
    q1, q2 = _checked(q1, q2)
    n = q1.shape[0]
    matrix, objective, bounds = _program(n)
    dq = _value_changes(q1, q2)
    res = linprog(objective, A_ub=matrix, b_ub=np.stack([dq, -dq], axis=1).reshape(-1), bounds=bounds,
                  method="highs")
    if not res.success:
        raise RuntimeError(f"potential LP failed: {res.message}")
    return res.x[:n * n].reshape(n, n), float(res.x[-1])


def max_residual(phi, q1, q2):
    """Largest |Phi change - value change| over all unilateral deviations."""
    q1, q2 = _checked(q1, q2)
    a, b, _who = _deviations(q1.shape[0])
    p = np.asarray(phi, dtype=float).reshape(-1)
    return float(np.abs((p[a] - p[b]) - _value_changes(q1, q2)).max())
