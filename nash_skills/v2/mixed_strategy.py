"""
Mixed-strategy (compact policy family) utilities for the paper's
alpha-potential method, Section 4.1:

    theta^i = (w^{A,i}, w^{D,i}) in Delta_5 x Delta_3

Why this exists: until now the alpha-potential pipeline only ever used
PURE joint actions (one fixed (attack, defend) pair per rally), and its
decision rules took a discrete argmax over the 15x15 Phi table. The paper
instead learns Phi(x, theta) from continuations played under MIXED theta
held fixed ("include deterministic choices as well as mixed strategies so
that the data cover both the fifteen fixed target-preparation pairings
and randomized play"), then runs projected gradient ascent over theta
online (Eq. 18).

The pure-only version cannot be upgraded just by doing gradient ascent on
the existing table: its natural mixed extension (expected Phi under
independent mixing) is MULTILINEAR in the four w vectors, and a
multilinear function's maximum over a product of simplexes is always
attained at a vertex (fix all blocks but one -> linear in that block ->
max at a vertex; repeat). So ascent would only rediscover what the
exhaustive discrete argmax already finds, and could do worse through
local optima. Continuous optimization is only worth anything once Phi is
TRAINED on mixed theta -- which the existing 90-dim layout supports with
no architecture change, since its one-hot action blocks are exactly
theta evaluated at the vertices (see
nash_skills.v2.joint_action.write_mixed_strategy_pair).

Pure numpy/torch, no MuJoCo dependency (tests/test_mixed_strategy.py).
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from nash_skills.skills import N_SKILLS
from nash_skills.defend_positions import N_DEFEND_POSITIONS

__all__ = [
    "DEFAULT_P_VERTEX",
    "project_to_simplex",
    "theta_vertex",
    "sample_theta",
    "sample_action_from_theta",
    "PGAResult",
    "pga_maximize",
]

# Fraction of sampled thetas that are pure (a vertex in BOTH blocks) --
# keeps the 15 deterministic target-preparation pairings well covered, as
# the paper requires, while most samples explore the simplex interior.
DEFAULT_P_VERTEX = 0.3

Theta4 = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def project_to_simplex(v) -> torch.Tensor:
    """Euclidean projection of a 1-D vector onto {w >= 0, sum w = 1}
    (sort-based algorithm, Duchi et al. 2008)."""
    v = torch.as_tensor(v, dtype=torch.float32)
    n = v.numel()
    u, _ = torch.sort(v, descending=True)
    css = torch.cumsum(u, dim=0)
    k = torch.arange(1, n + 1, dtype=v.dtype)
    positive = (u - (css - 1.0) / k) > 0
    rho = int(torch.nonzero(positive, as_tuple=False).max().item()) + 1
    tau = (css[rho - 1] - 1.0) / rho
    return torch.clamp(v - tau, min=0.0)


def theta_vertex(attack_idx: int, defend_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """The pure strategy (attack_idx, defend_idx) as a simplex vertex."""
    wA = np.zeros(N_SKILLS)
    wD = np.zeros(N_DEFEND_POSITIONS)
    wA[attack_idx] = 1.0
    wD[defend_idx] = 1.0
    return wA, wD


def sample_theta(rng: np.random.Generator, p_vertex: float = DEFAULT_P_VERTEX):
    """One player's theta: a uniformly random pure pairing with probability
    p_vertex, otherwise uniform over Delta_5 x Delta_3 (Dirichlet(1))."""
    if rng.random() < p_vertex:
        return theta_vertex(int(rng.integers(N_SKILLS)), int(rng.integers(N_DEFEND_POSITIONS)))
    return rng.dirichlet(np.ones(N_SKILLS)), rng.dirichlet(np.ones(N_DEFEND_POSITIONS))


def _clean(w) -> np.ndarray:
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    return w / w.sum()


def sample_action_from_theta(rng: np.random.Generator, wA, wD) -> Tuple[int, int]:
    """Draw (attack_idx, defend_idx) independently from the two blocks."""
    return (int(rng.choice(N_SKILLS, p=_clean(wA))),
            int(rng.choice(N_DEFEND_POSITIONS, p=_clean(wD))))


@dataclass
class PGAResult:
    theta: Theta4            # (wA1, wD1, wA2, wD2), detached
    value: float             # Phi at theta
    start_values: List[float]  # final Phi from every start -- the spread is
                               # the paper's "sensitivity to local optima"


def _random_theta4(rng: np.random.Generator) -> Theta4:
    return tuple(torch.tensor(rng.dirichlet(np.ones(n)), dtype=torch.float32)
                 for n in (N_SKILLS, N_DEFEND_POSITIONS, N_SKILLS, N_DEFEND_POSITIONS))


def _ascend(phi_fn: Callable, theta0: Theta4, steps: int, lr: float) -> Tuple[Theta4, float]:
    theta = [torch.as_tensor(w, dtype=torch.float32).detach().clone() for w in theta0]
    for _ in range(steps):
        params = [w.clone().requires_grad_(True) for w in theta]
        value = phi_fn(*params)
        if not value.requires_grad:
            break
        grads = torch.autograd.grad(value, params, allow_unused=True)
        with torch.no_grad():
            theta = [project_to_simplex(w + lr * g) if g is not None else w
                     for w, g in zip(theta, grads)]
    with torch.no_grad():
        final = float(phi_fn(*theta))
    return tuple(theta), final


def pga_maximize(phi_fn: Callable, init: Optional[Theta4], *, steps: int, lr: float,
                 n_restarts: int, rng: np.random.Generator) -> PGAResult:
    """
    Projected gradient ascent of phi_fn(wA1, wD1, wA2, wD2) -> scalar
    tensor over the product of the four simplexes (paper Eq. 18): one run
    warm-started from `init` (the preceding solution, if any) plus
    n_restarts runs from random interior points; returns the best.
    """
    starts = ([init] if init is not None else []) + [_random_theta4(rng) for _ in range(n_restarts)]
    if not starts:
        raise ValueError("pga_maximize needs at least one start (init or n_restarts >= 1).")
    results = [_ascend(phi_fn, s, steps, lr) for s in starts]
    best_theta, best_value = max(results, key=lambda r: r[1])
    return PGAResult(theta=best_theta, value=best_value, start_values=[r[1] for r in results])
