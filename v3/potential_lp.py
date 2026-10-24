"""Reusable LP and Q-batching utilities for V3 alpha-potential training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from scipy.optimize import linprog

from nash_skills.skills import SKILL_NAMES
from v3.constants import DEFENSE_POSITIONS
from v3.model_input import build_joint_action_batch, iter_action_pairs


N_ATTACK = len(SKILL_NAMES)
N_DEFENSE = len(DEFENSE_POSITIONS)
N_PHI = N_ATTACK * N_DEFENSE
ALPHA_INDEX = N_PHI
N_VARS = N_PHI + 1


@dataclass(frozen=True)
class AlphaPotentialFit:
    """Result of one state-local approximate-potential LP."""

    phi: np.ndarray
    alpha: float
    q_range: float
    normalized_alpha: float
    status: str
    message: str


def _phi_index(attack: int, defense: int) -> int:
    return int(attack) * N_DEFENSE + int(defense)


def deviation_pairs() -> Iterable[tuple[str, int, int, int, int]]:
    """Yield every ordered unilateral attacker and defender deviation."""
    for attack in range(N_ATTACK):
        for defense in range(N_DEFENSE):
            for attack_prime in range(N_ATTACK):
                if attack_prime != attack:
                    yield ("attacker", attack, defense, attack_prime, defense)
            for defense_prime in range(N_DEFENSE):
                if defense_prime != defense:
                    yield ("defender", attack, defense, attack, defense_prime)


def _build_lp_constraints() -> tuple[np.ndarray, list[tuple[str, int, int, int, int]]]:
    deviations = list(deviation_pairs())
    rows: list[np.ndarray] = []
    for _, attack, defense, attack_prime, defense_prime in deviations:
        source = _phi_index(attack, defense)
        destination = _phi_index(attack_prime, defense_prime)
        phi_delta = np.zeros(N_VARS, dtype=float)
        phi_delta[destination] += 1.0
        phi_delta[source] -= 1.0
        row_positive = -phi_delta.copy()
        row_positive[ALPHA_INDEX] = -1.0
        row_negative = phi_delta.copy()
        row_negative[ALPHA_INDEX] = -1.0
        rows.extend((row_positive, row_negative))
    return np.asarray(rows, dtype=float), deviations


_A_UB, DEVIATIONS = _build_lp_constraints()


def deviation_q_delta(q: np.ndarray, deviation: tuple[str, int, int, int, int]) -> float:
    player, attack, defense, attack_prime, defense_prime = deviation
    if player == "attacker":
        return float(q[attack_prime, defense] - q[attack, defense])
    return float(-q[attack, defense_prime] + q[attack, defense])


def solve_alpha_potential(q: np.ndarray) -> AlphaPotentialFit:
    """Fit Phi and minimum alpha, anchoring Phi(0,0)=0."""
    q = np.asarray(q, dtype=float)
    expected_shape = (N_ATTACK, N_DEFENSE)
    if q.shape != expected_shape:
        raise ValueError(f"Expected Q shape {expected_shape}, got {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError("Q matrix contains non-finite values")

    b_ub = []
    for deviation in DEVIATIONS:
        q_delta = deviation_q_delta(q, deviation)
        b_ub.extend((-q_delta, q_delta))
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * N_PHI + [(0.0, None)]
    bounds[0] = (0.0, 0.0)
    result = linprog(
        c=np.r_[np.zeros(N_PHI, dtype=float), 1.0],
        A_ub=_A_UB,
        b_ub=np.asarray(b_ub, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Alpha-potential LP failed: {result.message}")
    phi = np.asarray(result.x[:N_PHI], dtype=float).reshape(expected_shape)
    alpha = float(result.x[ALPHA_INDEX])
    q_range = float(np.max(q) - np.min(q))
    normalized = 0.0 if q_range <= 1e-12 else alpha / q_range
    return AlphaPotentialFit(phi, alpha, q_range, float(normalized), str(result.status), str(result.message))


def predict_q_matrices(model, states: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Batched equivalent of the high-level evaluator's 5x3 Q inference."""
    state_values = np.asarray(states, dtype=np.float32)
    if state_values.ndim != 2:
        raise ValueError(f"Expected state matrix, got shape {state_values.shape}")
    if not len(state_values):
        return np.empty((0, N_ATTACK, N_DEFENSE), dtype=float)
    matrices = []
    model.eval()
    for start in range(0, len(state_values), max(1, int(batch_size))):
        state_batch = state_values[start:start + batch_size]
        rows = [build_joint_action_batch(state_batch, attack, defense) for attack, defense in iter_action_pairs()]
        inputs = np.concatenate(rows, axis=0)
        with torch.no_grad():
            values = model(torch.as_tensor(inputs, dtype=torch.float32)).cpu().numpy().reshape(
                N_ATTACK * N_DEFENSE, len(state_batch)
            )
        matrices.append(values.reshape(N_ATTACK, N_DEFENSE, len(state_batch)).transpose(2, 0, 1))
    return np.concatenate(matrices, axis=0).astype(float, copy=False)


__all__ = [
    "AlphaPotentialFit",
    "DEVIATIONS",
    "deviation_pairs",
    "deviation_q_delta",
    "predict_q_matrices",
    "solve_alpha_potential",
]
