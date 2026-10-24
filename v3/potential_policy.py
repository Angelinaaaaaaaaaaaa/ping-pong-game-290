"""Inference helpers for the V3 attacker/defender potential game."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from model_arch import SimpleModel
from nash_skills.skills import SKILL_NAMES
from v3.constants import DEFENSE_POSITIONS
from v3.q_policy import encode_action_input
from v3.model_input import MODEL_INPUT_DIM
from v3.state_encoder import STATE_DIM
from v3.train_q_preposition_onehot import HIDDEN, INPUT_DIM, safe_load_state_dict


def new_action_counts() -> dict[str, np.ndarray]:
    """Create zeroed empirical counts; uniform pseudocounts are added at use."""
    return {
        "attack": np.zeros(len(SKILL_NAMES), dtype=np.int64),
        "defense": np.zeros(len(DEFENSE_POSITIONS), dtype=np.int64),
    }


def record_joint_actions(
    histories: dict[int, dict[str, np.ndarray]],
    attacker_id: int,
    attack_index: int,
    defender_id: int,
    defense_index: int,
) -> None:
    """Record a completed joint action by the physical player who chose it."""
    histories[attacker_id]["attack"][int(attack_index)] += 1
    histories[defender_id]["defense"][int(defense_index)] += 1


def _smoothed_probabilities(counts: np.ndarray, expected_size: int) -> np.ndarray:
    values = np.asarray(counts, dtype=float).reshape(-1)
    if values.shape != (expected_size,) or np.any(values < 0):
        raise ValueError(f"Expected {expected_size} nonnegative action counts")
    smoothed = values + 1.0
    return smoothed / smoothed.sum()


def potential_adaptive_action(
    phi: np.ndarray,
    role: str,
    opponent_attack_counts: np.ndarray,
    opponent_defense_counts: np.ndarray,
) -> tuple[int, dict[str, Any]]:
    """Choose an action against the opponent's empirical, smoothed action mix."""
    potential = _validated_potential(phi)
    attack_probs = _smoothed_probabilities(opponent_attack_counts, len(SKILL_NAMES))
    defense_probs = _smoothed_probabilities(opponent_defense_counts, len(DEFENSE_POSITIONS))
    if role == "attacker":
        values = potential @ defense_probs
        names = SKILL_NAMES
    elif role == "defender":
        values = attack_probs @ potential
        names = DEFENSE_POSITIONS
    else:
        raise ValueError(f"Unknown role {role!r}")
    selected = int(np.argmax(values))
    return selected, {
        "opponent_attack_counts": np.asarray(opponent_attack_counts, dtype=int).tolist(),
        "opponent_attack_probabilities": attack_probs.tolist(),
        "opponent_defense_counts": np.asarray(opponent_defense_counts, dtype=int).tolist(),
        "opponent_defense_probabilities": defense_probs.tolist(),
        "expected_potential_by_action": {
            name: float(value) for name, value in zip(names, values)
        },
        "selected_action": names[selected],
    }


def potential_minimax_action(phi: np.ndarray, role: str) -> tuple[int, dict[str, Any]]:
    """Choose the action maximizing its worst-case potential value."""
    potential = _validated_potential(phi)
    if role == "attacker":
        values = np.min(potential, axis=1)
        names = SKILL_NAMES
    elif role == "defender":
        values = np.min(potential, axis=0)
        names = DEFENSE_POSITIONS
    else:
        raise ValueError(f"Unknown role {role!r}")
    selected = int(np.argmax(values))
    return selected, {
        "worst_case_potential_by_action": {
            name: float(value) for name, value in zip(names, values)
        },
        "selected_action": names[selected],
    }


def _validated_potential(phi: np.ndarray) -> np.ndarray:
    potential = np.asarray(phi, dtype=float)
    expected_shape = (len(SKILL_NAMES), len(DEFENSE_POSITIONS))
    if potential.shape != expected_shape:
        raise ValueError(f"Potential matrix must have shape {expected_shape}")
    if not np.all(np.isfinite(potential)):
        raise ValueError("Potential matrix must contain only finite values")
    return potential


def load_potential_model(path: str, device: str = "cpu") -> SimpleModel:
    checkpoint = safe_load_state_dict(path)
    first_weight = checkpoint.get("fc.0.weight") if isinstance(checkpoint, dict) else None
    if first_weight is not None and int(first_weight.shape[1]) != MODEL_INPUT_DIM:
        raise ValueError(
            f"Incompatible V3 potential checkpoint: expected input_dim {MODEL_INPUT_DIM}, "
            f"got {int(first_weight.shape[1])}; historical 76/84 artifacts are not reusable"
        )
    model = SimpleModel(INPUT_DIM, HIDDEN, 1, last_layer_activation=None)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def potential_matrix(model: torch.nn.Module, base_state: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Evaluate Phi for every semantic attack/defense pair in a canonical state."""
    state = np.asarray(base_state, dtype=np.float32).reshape(-1)
    if state.shape[0] != STATE_DIM:
        raise ValueError(f"Expected {STATE_DIM}-dim state, got {state.shape[0]}")

    rows = [
        encode_action_input(state, attack, defense)
        for attack in SKILL_NAMES
        for defense in DEFENSE_POSITIONS
    ]
    model.eval()
    with torch.no_grad():
        values = model(torch.as_tensor(np.asarray(rows), dtype=torch.float32, device=device))
    return values.reshape(len(SKILL_NAMES), len(DEFENSE_POSITIONS)).cpu().numpy()


def potential_ibr(
    phi: np.ndarray,
    initial_attack: int,
    initial_defense: int,
    *,
    max_iter: int = 15,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Coordinate-ascent dynamics with cycle and iteration-limit handling."""
    potential = _validated_potential(phi)

    attack, defense = int(initial_attack), int(initial_defense)
    initial = (attack, defense)
    visited: list[tuple[int, int]] = [initial]
    seen = {initial}
    accepted_moves: list[dict[str, Any]] = []
    status = "max_iter_reached"
    iterations = 0

    for iteration in range(1, max_iter + 1):
        iterations = iteration
        changed = False
        proposed_attack = int(np.argmax(potential[:, defense]))
        if potential[proposed_attack, defense] - potential[attack, defense] > tolerance:
            attack = proposed_attack
            changed = True
            joint = (attack, defense)
            accepted_moves.append({
                "role": "attacker",
                "attack": SKILL_NAMES[attack],
                "defense": DEFENSE_POSITIONS[defense],
                "attack_index": attack,
                "defense_index": defense,
                "phi": float(potential[joint]),
            })
            if joint in seen:
                status = "cycle_detected"
                break
            seen.add(joint)
            visited.append(joint)

        proposed_defense = int(np.argmax(potential[attack, :]))
        if potential[attack, proposed_defense] - potential[attack, defense] > tolerance:
            defense = proposed_defense
            changed = True
            joint = (attack, defense)
            accepted_moves.append({
                "role": "defender",
                "attack": SKILL_NAMES[attack],
                "defense": DEFENSE_POSITIONS[defense],
                "attack_index": attack,
                "defense_index": defense,
                "phi": float(potential[joint]),
            })
            if joint in seen:
                status = "cycle_detected"
                break
            seen.add(joint)
            visited.append(joint)

        if not changed:
            status = "stable"
            break

    if status in ("cycle_detected", "max_iter_reached"):
        attack, defense = max(visited, key=lambda joint: potential[joint])
    final_phi = float(potential[attack, defense])
    return {
        "initial_joint_action": {
            "attack": SKILL_NAMES[initial[0]],
            "defense": DEFENSE_POSITIONS[initial[1]],
            "attack_index": initial[0],
            "defense_index": initial[1],
        },
        "final_joint_action": {
            "attack": SKILL_NAMES[int(attack)],
            "defense": DEFENSE_POSITIONS[int(defense)],
            "attack_index": int(attack),
            "defense_index": int(defense),
        },
        "iterations": iterations,
        "status": status,
        "final_phi": final_phi,
        "attack_best_response_gap": float(np.max(potential[:, defense]) - final_phi),
        "defense_best_response_gap": float(np.max(potential[attack, :]) - final_phi),
        "accepted_attacker_moves": sum(move["role"] == "attacker" for move in accepted_moves),
        "accepted_defender_moves": sum(move["role"] == "defender" for move in accepted_moves),
        "accepted_moves": accepted_moves,
        "visited_joint_actions": [{"attack": a, "defense": d} for a, d in visited],
    }
