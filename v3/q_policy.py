"""High-level V3 policies built from the trained one-hot Q model.

The learned payoff is attacker-perspective:

    Q(state, attack_skill, defense_preposition)

The state encoder used during V3 collection is role-based and canonical:
the attacker occupies the first role slots and the defender occupies the
second role slots. Counterfactual Q evaluation changes only the appended
one-hot action fields; it never mutates the physical state prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import linprog

from model_arch import SimpleModel
from nash_skills.skills import SKILL_NAMES
from v3.constants import DEFENSE_POSITIONS
from v3.model_input import (
    ACTION_DIM,
    MODEL_INPUT_DIM,
    MODEL_INPUT_SCHEMA_VERSION,
    build_joint_action_input,
    iter_action_pairs,
)
from v3.state_encoder import STATE_DIM, state_for_attacker
from v3.train_q_preposition_onehot import HIDDEN, safe_load_state_dict


INPUT_DIM = MODEL_INPUT_DIM


def encode_action_input(
    state: np.ndarray,
    attack_skill: str | int,
    defense_preposition: str | int,
) -> np.ndarray:
    return build_joint_action_input(state, attack_skill, defense_preposition)


def q_matrix(
    model: torch.nn.Module,
    base_state: np.ndarray,
    attacker_id: int | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Evaluate the trained model for all 5x3 semantic action pairs.

    ``attacker_id`` is retained as an ignored compatibility parameter for
    existing V3 callers. The state is already canonical; candidate actions
    affect only the appended one-hot action fields.
    """
    state = np.asarray(base_state, dtype=np.float32).reshape(-1)
    if state.shape[0] != STATE_DIM:
        raise ValueError(f"Expected {STATE_DIM}-dim state, got {state.shape[0]}")
    rows = [encode_action_input(state, skill, position) for skill, position in iter_action_pairs()]
    model.eval()
    with torch.no_grad():
        values = model(torch.tensor(np.asarray(rows), dtype=torch.float32, device=device))
    return values.reshape(len(SKILL_NAMES), len(DEFENSE_POSITIONS)).cpu().numpy()


def load_q_model(path: str, device: str = "cpu") -> SimpleModel:
    checkpoint = safe_load_state_dict(path)
    first_weight = checkpoint.get("fc.0.weight") if isinstance(checkpoint, dict) else None
    if first_weight is not None and int(first_weight.shape[1]) != INPUT_DIM:
        raise ValueError(
            f"Incompatible V3 Q checkpoint: expected input_dim {INPUT_DIM}, "
            f"got {int(first_weight.shape[1])}; historical 76/84 artifacts are not reusable"
        )
    model = SimpleModel(INPUT_DIM, HIDDEN, 1)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


@dataclass(frozen=True)
class MatrixGameSolution:
    attacker_probs: np.ndarray
    defender_probs: np.ndarray
    value: float
    status: str


def _normalize_distribution(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = np.maximum(values, 0.0)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return np.ones(values.shape[0], dtype=float) / values.shape[0]
    return values / total


def _pure_fallback(matrix: np.ndarray) -> MatrixGameSolution:
    attacker_row = int(np.argmax(np.min(matrix, axis=1)))
    defender_col = int(np.argmin(np.max(matrix, axis=0)))
    attacker = np.zeros(matrix.shape[0], dtype=float)
    defender = np.zeros(matrix.shape[1], dtype=float)
    attacker[attacker_row] = 1.0
    defender[defender_col] = 1.0
    value = float(matrix[attacker_row, defender_col])
    return MatrixGameSolution(attacker, defender, value, "pure_fallback")


def solve_zero_sum_matrix(matrix: np.ndarray) -> MatrixGameSolution:
    """Solve max_p min_q p^T Q q and min_q max_p p^T Q q via LP."""
    payoff = np.asarray(matrix, dtype=float)
    if payoff.shape != (len(SKILL_NAMES), len(DEFENSE_POSITIONS)):
        raise ValueError(f"Expected {(len(SKILL_NAMES), len(DEFENSE_POSITIONS))} payoff matrix, got {payoff.shape}")
    if not np.all(np.isfinite(payoff)):
        return _pure_fallback(np.nan_to_num(payoff, nan=0.0, posinf=1e6, neginf=-1e6))

    n_rows, n_cols = payoff.shape
    # Attacker: maximize v subject to Q.T @ p >= v.
    attacker_c = np.zeros(n_rows + 1, dtype=float)
    attacker_c[-1] = -1.0
    attacker_ub = np.zeros((n_cols, n_rows + 1), dtype=float)
    attacker_ub[:, :n_rows] = -payoff.T
    attacker_ub[:, -1] = 1.0
    attacker_result = linprog(
        attacker_c,
        A_ub=attacker_ub,
        b_ub=np.zeros(n_cols),
        A_eq=np.array([[*([1.0] * n_rows), 0.0]]),
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * n_rows + [(None, None)],
        method="highs",
    )

    # Defender: minimize v subject to Q @ q <= v.
    defender_c = np.zeros(n_cols + 1, dtype=float)
    defender_c[-1] = 1.0
    defender_ub = np.zeros((n_rows, n_cols + 1), dtype=float)
    defender_ub[:, :n_cols] = payoff
    defender_ub[:, -1] = -1.0
    defender_result = linprog(
        defender_c,
        A_ub=defender_ub,
        b_ub=np.zeros(n_rows),
        A_eq=np.array([[*([1.0] * n_cols), 0.0]]),
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * n_cols + [(None, None)],
        method="highs",
    )

    if not attacker_result.success or not defender_result.success:
        return _pure_fallback(payoff)

    attacker = _normalize_distribution(attacker_result.x[:n_rows])
    defender = _normalize_distribution(defender_result.x[:n_cols])
    value = float(0.5 * (attacker_result.x[-1] + defender_result.x[-1]))
    return MatrixGameSolution(attacker, defender, value, "linprog_highs")


def greedy_ibr_distributions(
    matrix: np.ndarray,
    iterations: int = 4,
    initial_defender_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic alternating best responses for the 5x3 game."""
    position = len(DEFENSE_POSITIONS) // 2 if initial_defender_index is None else int(initial_defender_index)
    shot = int(np.argmax(matrix[:, position]))
    for _ in range(max(1, iterations)):
        shot = int(np.argmax(matrix[:, position]))
        position = int(np.argmin(matrix[shot, :]))
    attacker = np.zeros(len(SKILL_NAMES), dtype=float)
    defender = np.zeros(len(DEFENSE_POSITIONS), dtype=float)
    attacker[shot] = 1.0
    defender[position] = 1.0
    return attacker, defender


def uniform_distribution(size: int) -> np.ndarray:
    return np.ones(size, dtype=float) / float(size)


def one_hot_distribution(size: int, index: int) -> np.ndarray:
    result = np.zeros(size, dtype=float)
    result[int(index)] = 1.0
    return result


__all__ = [
    "ACTION_DIM",
    "DEFENSE_POSITIONS",
    "INPUT_DIM",
    "MODEL_INPUT_SCHEMA_VERSION",
    "MatrixGameSolution",
    "SKILL_NAMES",
    "encode_action_input",
    "greedy_ibr_distributions",
    "load_q_model",
    "one_hot_distribution",
    "q_matrix",
    "solve_zero_sum_matrix",
    "state_for_attacker",
    "uniform_distribution",
]
