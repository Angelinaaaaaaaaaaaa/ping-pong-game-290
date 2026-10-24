#!/usr/bin/env python3
"""Evaluate V3 role-switching policies built from the one-hot Q model.

Policies are assigned to physical players.  At each net crossing the receiver
is the attacker and the other player is the defender.  A player therefore
uses its policy's attacker distribution when receiving and its defender
distribution when defending.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nash_skills.skills import SKILL_NAMES, SKILL_PROFILE_NAMES, skill_index  # noqa: E402
from v3.constants import DEFENSE_POSITIONS  # noqa: E402
from v3.eval_fixed_preposition import (  # noqa: E402
    HISTORY,
    attack_action_diagnostics,
    build_ppo_action,
    defense_action_diagnostics,
    maybe_render,
    opponent,
    position_snapshot,
    receiver_from_crossing,
)
from v3.q_policy import (  # noqa: E402
    MatrixGameSolution,
    greedy_ibr_distributions,
    load_q_model,
    one_hot_distribution,
    q_matrix,
    solve_zero_sum_matrix,
    state_for_attacker,
    uniform_distribution,
)
from v3.potential_policy import (  # noqa: E402
    load_potential_model,
    new_action_counts,
    potential_adaptive_action,
    potential_ibr,
    potential_matrix,
    potential_minimax_action,
    record_joint_actions,
)


POLICY_NAMES = (
    "nash_q", "greedy_q", "br_q", "potential_br", "potential_ibr",
    "potential_adaptive_rally", "potential_adaptive_match", "potential_minimax",
    "random", "fixed",
)


def seed_everything(seed: int) -> None:
    """Seed process-level RNGs used by policy/model/simulator code."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _capture_env_step(env, action):
    from nash_skills.eval_matchup import _capture_env_step as capture

    return capture(env, action)


def _infer_winner(obs, info) -> str:
    from nash_skills.eval_matchup import _infer_winner as infer

    return str(infer(obs, info))


def _last_hitter_from_reason(reason: str) -> int | None:
    if reason.startswith("missed_"):
        receiver = reason.split("_", 2)[1]
        return 2 if receiver == "ego" else 1
    if reason.startswith("ego_shot_out"):
        return 1
    if reason.startswith("opp_shot_out"):
        return 2
    return None


def _receiver_from_reason(reason: str) -> int | None:
    if reason.startswith("missed_"):
        label = reason.split("_", 2)[1]
        return 1 if label == "ego" else 2
    if "_before_valid_" in reason:
        label = reason.split("_before_valid_", 1)[1].split("_", 1)[0]
        return 1 if label == "ego" else 2
    return None


def _mark_successful_crossing(record: dict[str, Any]) -> None:
    record["contact_observed"] = True
    record["successful_return_observed"] = True
    record["attacker_miss"] = False
    record["attacker_out_of_bounds"] = False
    record["outcome_source"] = "following_net_crossing"


def _mark_terminal_outcome(record: dict[str, Any], reason: str) -> None:
    attacker_id = int(record["attacker_id"])
    last_hitter = _last_hitter_from_reason(reason)
    receiver = _receiver_from_reason(reason)
    if reason == "max_steps" or not reason:
        return

    if last_hitter is None:
        record["outcome_source"] = "unrecognized_termination_reason"
        return

    if "shot_out_before_valid" in reason:
        if attacker_id == last_hitter:
            record["contact_observed"] = True
            record["successful_return_observed"] = False
            record["attacker_miss"] = False
            record["attacker_out_of_bounds"] = True
            record["outcome_source"] = "terminal_shot_out_semantics"
        else:
            record["outcome_source"] = "terminal_hitter_does_not_match_decision_attacker"
        return

    if "missed_" in reason:
        if attacker_id == receiver:
            record["contact_observed"] = False
            record["successful_return_observed"] = False
            record["attacker_miss"] = True
            record["attacker_out_of_bounds"] = False
            record["outcome_source"] = "terminal_return_miss_semantics"
        elif attacker_id == last_hitter:
            record["contact_observed"] = True
            record["successful_return_observed"] = True
            record["attacker_miss"] = False
            record["attacker_out_of_bounds"] = False
            record["outcome_source"] = "terminal_return_miss_semantics"
        else:
            record["outcome_source"] = "terminal_hitter_does_not_match_decision_attacker"


def _rate(records: list[dict[str, Any]], field: str) -> float | str:
    values = [record[field] for record in records if record.get(field) is not None]
    return float(np.mean(np.asarray(values, dtype=bool))) if values else ""


def _distribution_for_role(
    policy: str,
    role: str,
    matrix: np.ndarray,
    solution: MatrixGameSolution,
    *,
    fixed_shot: str,
    fixed_defense: str,
    greedy_iterations: int,
) -> np.ndarray:
    if policy == "nash_q":
        return solution.attacker_probs.copy() if role == "attacker" else solution.defender_probs.copy()
    if policy == "random":
        return uniform_distribution(len(SKILL_NAMES) if role == "attacker" else len(DEFENSE_POSITIONS))
    if policy == "fixed":
        if role == "attacker":
            return one_hot_distribution(len(SKILL_NAMES), skill_index(fixed_shot))
        return one_hot_distribution(len(DEFENSE_POSITIONS), DEFENSE_POSITIONS.index(fixed_defense))
    if policy != "greedy_q":
        raise ValueError(f"Unknown policy {policy!r}")
    greedy_attacker, greedy_defender = greedy_ibr_distributions(matrix, iterations=greedy_iterations)
    return greedy_attacker if role == "attacker" else greedy_defender


def _br_q_distribution(
    *,
    role: str,
    matrix: np.ndarray,
    solution: MatrixGameSolution,
    opponent_id: int,
    last_attack_skill: dict[int, str | None],
    last_defense_position: dict[int, str | None],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a BR-Q action distribution from only the previous action history."""
    if role == "attacker":
        previous_opponent_action = last_defense_position[opponent_id]
        if previous_opponent_action is not None:
            defense_index = DEFENSE_POSITIONS.index(previous_opponent_action)
            best_response_index = int(np.argmax(matrix[:, defense_index]))
            return one_hot_distribution(len(SKILL_NAMES), best_response_index), {
                "predicted_opponent_action": previous_opponent_action,
                "prediction_source": "previous_observation",
                "best_response_index": best_response_index,
                "selection_mode": "best_response",
            }
        return solution.attacker_probs.copy(), {
            "predicted_opponent_action": None,
            "prediction_source": "nash_fallback",
            "best_response_index": None,
            "selection_mode": "nash_fallback",
        }

    if role == "defender":
        previous_opponent_action = last_attack_skill[opponent_id]
        if previous_opponent_action is not None:
            attack_index = SKILL_NAMES.index(previous_opponent_action)
            best_response_index = int(np.argmin(matrix[attack_index, :]))
            return one_hot_distribution(len(DEFENSE_POSITIONS), best_response_index), {
                "predicted_opponent_action": previous_opponent_action,
                "prediction_source": "previous_observation",
                "best_response_index": best_response_index,
                "selection_mode": "best_response",
            }
        return solution.defender_probs.copy(), {
            "predicted_opponent_action": None,
            "prediction_source": "nash_fallback",
            "best_response_index": None,
            "selection_mode": "nash_fallback",
        }

    raise ValueError(f"Unknown role {role!r}")


def _sample(distribution: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(distribution), p=np.asarray(distribution, dtype=float)))


def _policy_label(policy: str | dict[str, str], fixed_shot: str, fixed_defense: str) -> str:
    if isinstance(policy, dict):
        if policy["attack"] != policy["defense"]:
            return f"attack={policy['attack']},defense={policy['defense']}"
        policy = policy["attack"]
    if policy != "fixed":
        return policy
    return f"fixed(shot={fixed_shot},defense={fixed_defense})"


def _fixed_configuration(
    policies: dict[int, dict[str, str]],
    fixed_shots: dict[int, str],
    fixed_defenses: dict[int, str],
) -> dict[str, Any]:
    """Describe fixed-player settings explicitly for downstream diagnostics."""
    fixed_players = [
        player for player in (1, 2)
        if "fixed" in (policies[player]["attack"], policies[player]["defense"])
    ]
    configuration: dict[str, Any] = {
        "fixed_player_ids": fixed_players,
        "fixed_player": fixed_players[0] if len(fixed_players) == 1 else "",
        "fixed_attack_skill": (
            fixed_shots[fixed_players[0]]
            if len(fixed_players) == 1 and policies[fixed_players[0]]["attack"] == "fixed"
            else ""
        ),
        "fixed_defender_position": (
            fixed_defenses[fixed_players[0]]
            if len(fixed_players) == 1 and policies[fixed_players[0]]["defense"] == "fixed"
            else ""
        ),
        "p1_policy_name": policies[1]["attack"] if policies[1]["attack"] == policies[1]["defense"] else "",
        "p2_policy_name": policies[2]["attack"] if policies[2]["attack"] == policies[2]["defense"] else "",
        "p1_attack_policy": policies[1]["attack"],
        "p1_defense_policy": policies[1]["defense"],
        "p2_attack_policy": policies[2]["attack"],
        "p2_defense_policy": policies[2]["defense"],
        "p1_fixed_attack_skill": fixed_shots[1] if policies[1]["attack"] == "fixed" else "",
        "p1_fixed_defender_position": fixed_defenses[1] if policies[1]["defense"] == "fixed" else "",
        "p2_fixed_attack_skill": fixed_shots[2] if policies[2]["attack"] == "fixed" else "",
        "p2_fixed_defender_position": fixed_defenses[2] if policies[2]["defense"] == "fixed" else "",
    }
    return configuration


def run_episode(
    *,
    episode_index: int,
    env,
    ppo,
    model,
    potential_model,
    policies: dict[int, dict[str, str]],
    match_action_counts: dict[int, dict[str, np.ndarray]] | None = None,
    fixed_shots: dict[int, str],
    fixed_defenses: dict[int, str],
    rng: np.random.Generator,
    device: str,
    max_steps: int,
    greedy_iterations: int,
    render: bool,
    debug_decisions: bool,
    episode_seed: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env.set_skills("left", "right")
    env.clear_defense_position()
    if episode_seed is None:
        obs, info = env.reset()
    else:
        seed_everything(int(episode_seed))
        obs, info = env.reset(seed=int(episode_seed))
    previous_ball_x = float(obs[36])
    decisions: list[dict[str, Any]] = []
    steps = 0
    done = False
    winner = "truncated"
    termination_reason = "max_steps"
    last_attack_skill: dict[int, str | None] = {1: None, 2: None}
    last_defense_position: dict[int, str | None] = {1: None, 2: None}
    rally_action_counts = {1: new_action_counts(), 2: new_action_counts()}
    if match_action_counts is None:
        match_action_counts = {1: new_action_counts(), 2: new_action_counts()}

    while steps < max_steps and not done:
        action = build_ppo_action(ppo, obs, info)
        (obs, _reward, done, _truncated, info), _lines = _capture_env_step(env, action)
        maybe_render(env, render)
        steps += 1
        current_ball_x = float(obs[36])
        attacker_id = receiver_from_crossing(previous_ball_x, current_ball_x)

        if attacker_id is not None:
            if decisions:
                _mark_successful_crossing(decisions[-1])

            defender_id = opponent(attacker_id)
            base_state = state_for_attacker(obs, info, attacker_id)
            matrix = q_matrix(model, base_state, attacker_id, device=device)
            phi = potential_matrix(potential_model, base_state, device=device)
            nash_solution = solve_zero_sum_matrix(matrix)

            attacker_policy = policies[attacker_id]["attack"]
            defender_policy = policies[defender_id]["defense"]
            center_attack = len(SKILL_NAMES) // 2
            center_defense = len(DEFENSE_POSITIONS) // 2
            previous_attack = last_attack_skill[attacker_id]
            previous_defense = last_defense_position[defender_id]
            ibr_info = None
            if "potential_ibr" in (attacker_policy, defender_policy):
                initial_attack = SKILL_NAMES.index(previous_attack) if previous_attack is not None and previous_defense is not None else center_attack
                initial_defense = DEFENSE_POSITIONS.index(previous_defense) if previous_attack is not None and previous_defense is not None else center_defense
                ibr_info = potential_ibr(phi, initial_attack, initial_defense)

            if attacker_policy == "br_q":
                attacker_probs, attacker_br = _br_q_distribution(
                    role="attacker",
                    matrix=matrix,
                    solution=nash_solution,
                    opponent_id=defender_id,
                    last_attack_skill=last_attack_skill,
                    last_defense_position=last_defense_position,
                )
            elif attacker_policy == "potential_br":
                held_defense = DEFENSE_POSITIONS.index(previous_defense) if previous_defense is not None else center_defense
                potential_action = int(np.argmax(phi[:, held_defense]))
                attacker_probs = one_hot_distribution(len(SKILL_NAMES), potential_action)
                attacker_br = {
                    "held_defense": DEFENSE_POSITIONS[held_defense],
                    "candidate_potential_values": {
                        name: float(value) for name, value in zip(SKILL_NAMES, phi[:, held_defense])
                    },
                    "selected_action": SKILL_NAMES[potential_action],
                }
            elif attacker_policy == "potential_ibr":
                attacker_probs = one_hot_distribution(len(SKILL_NAMES), ibr_info["final_joint_action"]["attack_index"])
                attacker_br = None
            elif attacker_policy in ("potential_adaptive_rally", "potential_adaptive_match"):
                histories = rally_action_counts if attacker_policy == "potential_adaptive_rally" else match_action_counts
                potential_action, attacker_br = potential_adaptive_action(
                    phi,
                    "attacker",
                    histories[defender_id]["attack"],
                    histories[defender_id]["defense"],
                )
                attacker_br["opponent_id"] = defender_id
                attacker_probs = one_hot_distribution(len(SKILL_NAMES), potential_action)
            elif attacker_policy == "potential_minimax":
                potential_action, attacker_br = potential_minimax_action(phi, "attacker")
                attacker_probs = one_hot_distribution(len(SKILL_NAMES), potential_action)
            else:
                attacker_probs = _distribution_for_role(
                    attacker_policy,
                    "attacker",
                    matrix,
                    nash_solution,
                    fixed_shot=fixed_shots[attacker_id],
                    fixed_defense=fixed_defenses[attacker_id],
                    greedy_iterations=greedy_iterations,
                )
                attacker_br = None
            if defender_policy == "br_q":
                defender_probs, defender_br = _br_q_distribution(
                    role="defender",
                    matrix=matrix,
                    solution=nash_solution,
                    opponent_id=attacker_id,
                    last_attack_skill=last_attack_skill,
                    last_defense_position=last_defense_position,
                )
            elif defender_policy == "potential_br":
                held_attack = SKILL_NAMES.index(previous_attack) if previous_attack is not None else center_attack
                potential_action = int(np.argmax(phi[held_attack, :]))
                defender_probs = one_hot_distribution(len(DEFENSE_POSITIONS), potential_action)
                defender_br = {
                    "held_attack": SKILL_NAMES[held_attack],
                    "candidate_potential_values": {
                        name: float(value) for name, value in zip(DEFENSE_POSITIONS, phi[held_attack, :])
                    },
                    "selected_action": DEFENSE_POSITIONS[potential_action],
                }
            elif defender_policy == "potential_ibr":
                defender_probs = one_hot_distribution(len(DEFENSE_POSITIONS), ibr_info["final_joint_action"]["defense_index"])
                defender_br = None
            elif defender_policy in ("potential_adaptive_rally", "potential_adaptive_match"):
                histories = rally_action_counts if defender_policy == "potential_adaptive_rally" else match_action_counts
                potential_action, defender_br = potential_adaptive_action(
                    phi,
                    "defender",
                    histories[attacker_id]["attack"],
                    histories[attacker_id]["defense"],
                )
                defender_br["opponent_id"] = attacker_id
                defender_probs = one_hot_distribution(len(DEFENSE_POSITIONS), potential_action)
            elif defender_policy == "potential_minimax":
                potential_action, defender_br = potential_minimax_action(phi, "defender")
                defender_probs = one_hot_distribution(len(DEFENSE_POSITIONS), potential_action)
            else:
                defender_probs = _distribution_for_role(
                    defender_policy,
                    "defender",
                    matrix,
                    nash_solution,
                    fixed_shot=fixed_shots[defender_id],
                    fixed_defense=fixed_defenses[defender_id],
                    greedy_iterations=greedy_iterations,
                )
                defender_br = None

            # Both role actions are computed from the pre-decision history.
            selected_shot_index = _sample(attacker_probs, rng)
            selected_defense_index = _sample(defender_probs, rng)
            selected_shot = SKILL_NAMES[selected_shot_index]
            selected_defense = DEFENSE_POSITIONS[selected_defense_index]
            if attacker_br is not None:
                attacker_br["selected_best_response"] = selected_shot
            if defender_br is not None:
                defender_br["selected_best_response"] = selected_defense

            # Update observations only after both current actions have been selected.
            last_attack_skill[attacker_id] = selected_shot
            last_defense_position[defender_id] = selected_defense
            # Encode before committing the newly selected actions. The action
            # pair is carried in the log and appended only for model input.
            logged_state = state_for_attacker(obs, info, attacker_id)

            if attacker_id == 1:
                env.set_skills(selected_shot, env._skill2)
            else:
                env.set_skills(env._skill1, selected_shot)
            commitment = env.set_defense_position(defender_id, selected_defense)
            record_joint_actions(
                rally_action_counts, attacker_id, selected_shot_index, defender_id, selected_defense_index
            )
            record_joint_actions(
                match_action_counts, attacker_id, selected_shot_index, defender_id, selected_defense_index
            )
            defender_position = position_snapshot(env, defender_id)
            semantic_diagnostics = {
                **attack_action_diagnostics(attacker_id, selected_shot, env.skill_profile),
                **defense_action_diagnostics(
                    defender_id, selected_defense, commitment, defender_position
                ),
            }
            fixed_configuration = _fixed_configuration(policies, fixed_shots, fixed_defenses)

            decisions.append({
                "episode_index": episode_index,
                "decision_index": len(decisions) + 1,
                "state": logged_state.astype(float).tolist(),
                "q_matrix": matrix.astype(float).tolist(),
                "potential_matrix": phi.astype(float).tolist(),
                "attacker_policy_probabilities": attacker_probs.astype(float).tolist(),
                "defender_policy_probabilities": defender_probs.astype(float).tolist(),
                "nash_attacker_probabilities": nash_solution.attacker_probs.astype(float).tolist(),
                "nash_defender_probabilities": nash_solution.defender_probs.astype(float).tolist(),
                "nash_value": nash_solution.value,
                "nash_solver_status": nash_solution.status,
                "attacker_policy": attacker_policy,
                "defender_policy": defender_policy,
                "potential_ibr_initial_joint_action": "" if ibr_info is None else ibr_info["initial_joint_action"],
                "potential_ibr_final_joint_action": "" if ibr_info is None else ibr_info["final_joint_action"],
                "potential_ibr_iterations": "" if ibr_info is None else ibr_info["iterations"],
                "potential_ibr_status": "" if ibr_info is None else ibr_info["status"],
                "potential_ibr_final_phi": "" if ibr_info is None else ibr_info["final_phi"],
                "potential_ibr_attack_best_response_gap": "" if ibr_info is None else ibr_info["attack_best_response_gap"],
                "potential_ibr_defense_best_response_gap": "" if ibr_info is None else ibr_info["defense_best_response_gap"],
                "potential_ibr_accepted_attacker_moves": "" if ibr_info is None else ibr_info["accepted_attacker_moves"],
                "potential_ibr_accepted_defender_moves": "" if ibr_info is None else ibr_info["accepted_defender_moves"],
                "potential_ibr_accepted_moves": "" if ibr_info is None else ibr_info["accepted_moves"],
                "attacker_policy_details": attacker_br if attacker_policy.startswith("potential_") else None,
                "defender_policy_details": defender_br if defender_policy.startswith("potential_") else None,
                "fixed_player": fixed_configuration["fixed_player"],
                "fixed_attack_skill": fixed_configuration["fixed_attack_skill"],
                "fixed_defender_position": fixed_configuration["fixed_defender_position"],
                "br_attacker_predicted_opponent_action": attacker_br["predicted_opponent_action"] if attacker_policy == "br_q" else "",
                "br_attacker_prediction_source": attacker_br["prediction_source"] if attacker_policy == "br_q" else "",
                "br_attacker_selected_best_response": attacker_br["selected_best_response"] if attacker_policy == "br_q" else "",
                "br_attacker_selection_mode": attacker_br["selection_mode"] if attacker_policy == "br_q" else "",
                "br_defender_predicted_opponent_action": defender_br["predicted_opponent_action"] if defender_policy == "br_q" else "",
                "br_defender_prediction_source": defender_br["prediction_source"] if defender_policy == "br_q" else "",
                "br_defender_selected_best_response": defender_br["selected_best_response"] if defender_policy == "br_q" else "",
                "br_defender_selection_mode": defender_br["selection_mode"] if defender_policy == "br_q" else "",
                "attacker_id": attacker_id,
                "defender_id": defender_id,
                "attack_skill": selected_shot,
                "defense_preposition": selected_defense,
                **semantic_diagnostics,
                "defender_target_world": [commitment.target.x, commitment.target.y],
                "defender_target_qpos": [commitment.qpos_target.x, commitment.qpos_target.y],
                "defender_actual_world_at_commit": [
                    defender_position["world_x"],
                    defender_position["world_y"],
                ],
                "defender_actual_qpos_at_commit": [
                    defender_position["qpos_x"],
                    defender_position["qpos_y"],
                ],
                "defender_coordinate_frames": {
                    "target": "world_and_raw_gantry_qpos",
                    "actual": "world_and_raw_gantry_qpos",
                },
                "contact_observed": None,
                "successful_return_observed": None,
                "attacker_miss": None,
                "attacker_out_of_bounds": None,
                "outcome_source": "pending",
            })

            if debug_decisions or render:
                print(
                    f"decision ep={episode_index} n={len(decisions)} "
                    f"attacker=P{attacker_id} defender=P{defender_id} "
                    f"shot={selected_shot} defense={selected_defense} "
                    f"target_world=({commitment.target.x:.3f},{commitment.target.y:.3f}) "
                    f"target_qpos=({commitment.qpos_target.x:.3f},{commitment.qpos_target.y:.3f}) "
                    f"actual_world=({defender_position['world_x']:.3f},{defender_position['world_y']:.3f}) "
                    f"target_side={semantic_diagnostics['defense_target_player_relative_side']} "
                    f"actual_side={semantic_diagnostics['defense_player_relative_side']}",
                    flush=True,
                )

        previous_ball_x = current_ball_x
        if done:
            winner = str(info.get("winner") or _infer_winner(obs, info))
            termination_reason = str(info.get("termination_reason", "done"))

    if done:
        winner_player = 1 if winner == "ego" else 2 if winner == "opp" else None
        if decisions:
            _mark_terminal_outcome(decisions[-1], termination_reason)
    else:
        winner_player = None

    for decision in decisions:
        decision["rally_winner"] = winner
        decision["winner_player"] = winner_player
        decision["termination_reason"] = termination_reason
        decision["truncated"] = not done

    episode = {
        "episode_index": episode_index,
        "p1_policy": _policy_label(policies[1], fixed_shots[1], fixed_defenses[1]),
        "p2_policy": _policy_label(policies[2], fixed_shots[2], fixed_defenses[2]),
        "winner": winner,
        "winner_player": winner_player if winner_player is not None else "",
        "termination_reason": termination_reason,
        "truncated": not done,
        "steps": steps,
        "decision_count": len(decisions),
        "contact_rate": _rate(decisions, "contact_observed"),
        "successful_return_rate": _rate(decisions, "successful_return_observed"),
        "miss_rate": _rate(decisions, "attacker_miss"),
        "out_of_bounds_rate": _rate(decisions, "attacker_out_of_bounds"),
    }
    return episode, decisions


def _decision_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    for key in (
        "state",
        "q_matrix",
        "potential_matrix",
        "attacker_policy_probabilities",
        "defender_policy_probabilities",
        "nash_attacker_probabilities",
        "nash_defender_probabilities",
        "defender_target_world",
        "defender_target_qpos",
        "defender_actual_world_at_commit",
        "defender_actual_qpos_at_commit",
        "defender_coordinate_frames",
        "potential_ibr_initial_joint_action",
        "potential_ibr_final_joint_action",
        "potential_ibr_accepted_moves",
        "attacker_policy_details",
        "defender_policy_details",
    ):
        result[f"{key}_json"] = json.dumps(result.pop(key), separators=(",", ":"))
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _frequency(rows: list[dict[str, Any]], key: str, group_key: str) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = {"p1": Counter(), "p2": Counter()}
    for row in rows:
        group = "p1" if int(row[group_key]) == 1 else "p2"
        counts[group][str(row[key])] += 1
    return {group: dict(sorted(counter.items())) for group, counter in counts.items()}


def build_summary(
    episodes: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
    example_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    winner_counts = Counter(str(row["winner"]) for row in episodes)
    outcome_fields = {
        "contact_rate": "contact_observed",
        "successful_return_rate": "successful_return_observed",
        "miss_rate": "attacker_miss",
        "out_of_bounds_rate": "attacker_out_of_bounds",
    }
    outcome_rates = {name: _rate(decisions, field) for name, field in outcome_fields.items()}

    nash_attacker = np.asarray([row["nash_attacker_probabilities"] for row in decisions], dtype=float)
    nash_defender = np.asarray([row["nash_defender_probabilities"] for row in decisions], dtype=float)
    avg_nash_attacker = np.mean(nash_attacker, axis=0).tolist() if len(decisions) else []
    avg_nash_defender = np.mean(nash_defender, axis=0).tolist() if len(decisions) else []

    ibr_decisions = [row for row in decisions if row.get("potential_ibr_status")]
    iteration_distribution = dict(sorted(Counter(
        str(row["potential_ibr_iterations"]) for row in ibr_decisions
    ).items(), key=lambda item: int(item[0])))
    accepted_move_totals = [
        int(row["potential_ibr_accepted_attacker_moves"])
        + int(row["potential_ibr_accepted_defender_moves"])
        for row in ibr_decisions
    ]
    accepted_move_distribution = dict(sorted(Counter(map(str, accepted_move_totals)).items(), key=lambda item: int(item[0])))
    accepted_attacker_distribution = dict(sorted(Counter(
        str(row["potential_ibr_accepted_attacker_moves"]) for row in ibr_decisions
    ).items(), key=lambda item: int(item[0])))
    accepted_defender_distribution = dict(sorted(Counter(
        str(row["potential_ibr_accepted_defender_moves"]) for row in ibr_decisions
    ).items(), key=lambda item: int(item[0])))
    local_maximum_count = sum(
        float(row["potential_ibr_attack_best_response_gap"]) <= 1e-6
        and float(row["potential_ibr_defense_best_response_gap"]) <= 1e-6
        for row in ibr_decisions
    )
    final_joint_actions = {
        (row["potential_ibr_final_joint_action"]["attack"], row["potential_ibr_final_joint_action"]["defense"])
        for row in ibr_decisions
    }
    potential_ibr_summary = {
        "decision_count": len(ibr_decisions),
        "both_gaps_le_1e-6_count": local_maximum_count,
        "both_gaps_le_1e-6_percent": 100.0 * local_maximum_count / len(ibr_decisions) if ibr_decisions else "",
        "iteration_distribution": iteration_distribution,
        "accepted_move_distribution": accepted_move_distribution,
        "accepted_attacker_move_distribution": accepted_attacker_distribution,
        "accepted_defender_move_distribution": accepted_defender_distribution,
        "unique_final_joint_action_count": len(final_joint_actions),
        "unique_final_joint_actions": [
            {"attack": attack, "defense": defense}
            for attack, defense in sorted(final_joint_actions)
        ],
    }

    examples = []
    for row in decisions[:example_count]:
        examples.append({
            "episode_index": row["episode_index"],
            "decision_index": row["decision_index"],
            "attacker_id": row["attacker_id"],
            "defender_id": row["defender_id"],
            "attack_skill": row["attack_skill"],
            "defense_preposition": row["defense_preposition"],
            "q_matrix": row["q_matrix"],
            "nash_attacker_probabilities": row["nash_attacker_probabilities"],
            "nash_defender_probabilities": row["nash_defender_probabilities"],
            "nash_value": row["nash_value"],
        })

    summary = {
        "metadata": metadata,
        "episode_count": len(episodes),
        "decision_count": len(decisions),
        "winner_counts": dict(sorted(winner_counts.items())),
        "p1_win_rate": winner_counts.get("ego", 0) / len(episodes) if episodes else "",
        "p2_win_rate": winner_counts.get("opp", 0) / len(episodes) if episodes else "",
        "shot_selection_frequencies": _frequency(decisions, "attack_skill", "attacker_id"),
        "defender_position_frequencies": _frequency(decisions, "defense_preposition", "defender_id"),
        "average_nash_attacker_probabilities": dict(zip(SKILL_NAMES, avg_nash_attacker)),
        "average_nash_defender_probabilities": dict(zip(DEFENSE_POSITIONS, avg_nash_defender)),
        "outcome_rates": outcome_rates,
        "termination_reasons": dict(sorted(Counter(str(row["termination_reason"]) for row in decisions).items())),
        "potential_ibr_diagnostics": potential_ibr_summary,
        "q_matrix_examples": examples,
    }

    summary_rows: list[dict[str, Any]] = []
    summary_rows.extend([
        {"metric": "episode_count", "group": "all", "action": "", "value": len(episodes)},
        {"metric": "decision_count", "group": "all", "action": "", "value": len(decisions)},
        {"metric": "p1_win_rate", "group": "all", "action": "", "value": summary["p1_win_rate"]},
        {"metric": "p2_win_rate", "group": "all", "action": "", "value": summary["p2_win_rate"]},
    ])
    for group, values in summary["shot_selection_frequencies"].items():
        total = sum(values.values())
        for action, count in values.items():
            summary_rows.append({"metric": "shot_selection_frequency", "group": group, "action": action, "value": count / total if total else ""})
    for group, values in summary["defender_position_frequencies"].items():
        total = sum(values.values())
        for action, count in values.items():
            summary_rows.append({"metric": "defender_position_frequency", "group": group, "action": action, "value": count / total if total else ""})
    for action, value in summary["average_nash_attacker_probabilities"].items():
        summary_rows.append({"metric": "average_nash_attacker_probability", "group": "all", "action": action, "value": value})
    for action, value in summary["average_nash_defender_probabilities"].items():
        summary_rows.append({"metric": "average_nash_defender_probability", "group": "all", "action": action, "value": value})
    for metric, value in outcome_rates.items():
        summary_rows.append({"metric": metric, "group": "all_decisions", "action": "", "value": value})
    for reason, count in summary["termination_reasons"].items():
        summary_rows.append({"metric": "termination_reason_frequency", "group": "all_decisions", "action": reason, "value": count / len(decisions) if decisions else ""})
    summary_rows.extend([
        {"metric": "potential_ibr_local_maximum_percent", "group": "potential_ibr", "action": "both_gaps_le_1e-6", "value": potential_ibr_summary["both_gaps_le_1e-6_percent"]},
        {"metric": "potential_ibr_unique_final_joint_actions", "group": "potential_ibr", "action": "", "value": potential_ibr_summary["unique_final_joint_action_count"]},
    ])
    for distribution_name, distribution in (
        ("iteration", iteration_distribution),
        ("accepted_moves", accepted_move_distribution),
        ("accepted_attacker_moves", accepted_attacker_distribution),
        ("accepted_defender_moves", accepted_defender_distribution),
    ):
        for value, count in distribution.items():
            summary_rows.append({
                "metric": f"potential_ibr_{distribution_name}_distribution",
                "group": "potential_ibr",
                "action": value,
                "value": count,
            })

    example_rows = [
        {
            "example_index": index,
            "episode_index": row["episode_index"],
            "decision_index": row["decision_index"],
            "attacker_id": row["attacker_id"],
            "defender_id": row["defender_id"],
            "attack_skill": row["attack_skill"],
            "defense_preposition": row["defense_preposition"],
            "q_matrix_json": json.dumps(row["q_matrix"], separators=(",", ":")),
            "nash_attacker_probabilities_json": json.dumps(row["nash_attacker_probabilities"], separators=(",", ":")),
            "nash_defender_probabilities_json": json.dumps(row["nash_defender_probabilities"], separators=(",", ":")),
            "nash_value": row["nash_value"],
        }
        for index, row in enumerate(decisions[:example_count])
    ]
    return summary, summary_rows, example_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/v3/q_preposition_state74_onehot.pth")
    parser.add_argument("--ppo", default="logs/best_model_tracker1/best_model")
    parser.add_argument("--potential-model", default="models/v3/potential_state74.pth")
    parser.add_argument("--p1-policy", choices=POLICY_NAMES, default=None, help="Legacy shorthand: set both P1 role policies.")
    parser.add_argument("--p2-policy", choices=POLICY_NAMES, default=None, help="Legacy shorthand: set both P2 role policies.")
    for player, default in ((1, "nash_q"), (2, "random")):
        parser.add_argument(f"--p{player}-attack-policy", choices=POLICY_NAMES, default=None)
        parser.add_argument(f"--p{player}-defense-policy", choices=POLICY_NAMES, default=None)
    parser.add_argument("--p1-shot", choices=SKILL_NAMES, default="left")
    parser.add_argument("--p2-shot", choices=SKILL_NAMES, default="right")
    parser.add_argument("--p1-defense", choices=DEFENSE_POSITIONS, default="center")
    parser.add_argument("--p2-defense", choices=DEFENSE_POSITIONS, default="center")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--greedy-iterations", type=int, default=4)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--reset-mode", choices=["clean", "ready", "carryover"], default="ready")
    parser.add_argument("--skill-profile", choices=SKILL_PROFILE_NAMES, default="aggressive")
    parser.add_argument("--gantry-speed-scale", type=float, default=1.0)
    parser.add_argument("--planner", choices=["legacy", "legacy_legal", "optimized"], default="legacy")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the live MuJoCo viewer; do not set MUJOCO_GL=egl with this option.",
    )
    parser.add_argument(
        "--debug-decisions",
        action="store_true",
        help="Print the selected semantic actions and world/raw defender targets.",
    )
    parser.add_argument("--output-dir", default="skill_eval/v3/q_policy_eval_canonical")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1 or args.steps < 1:
        raise ValueError("--episodes and --steps must be positive")

    from stable_baselines3 import PPO
    from v3.env_wrapper import V3PrePositionEnv

    model = load_q_model(args.model, device=args.device)
    potential_model = load_potential_model(args.potential_model, device=args.device)
    ppo = PPO.load(args.ppo)
    env = V3PrePositionEnv(
        proc_id=1,
        history=HISTORY,
        reset_mode=args.reset_mode,
        skill_profile=args.skill_profile,
        gantry_speed_scale=args.gantry_speed_scale,
        planner=args.planner,
    )
    rng = np.random.default_rng(args.seed)
    policies = {}
    for player, default in ((1, "nash_q"), (2, "random")):
        legacy = getattr(args, f"p{player}_policy") or default
        policies[player] = {
            "attack": getattr(args, f"p{player}_attack_policy") or legacy,
            "defense": getattr(args, f"p{player}_defense_policy") or legacy,
        }
    fixed_shots = {1: args.p1_shot, 2: args.p2_shot}
    fixed_defenses = {1: args.p1_defense, 2: args.p2_defense}
    episodes: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    match_action_counts = {1: new_action_counts(), 2: new_action_counts()}

    try:
        for episode_index in range(args.episodes):
            episode, episode_decisions = run_episode(
                episode_index=episode_index,
                env=env,
                ppo=ppo,
                model=model,
                potential_model=potential_model,
                policies=policies,
                match_action_counts=match_action_counts,
                fixed_shots=fixed_shots,
                fixed_defenses=fixed_defenses,
                rng=rng,
                device=args.device,
                max_steps=args.steps,
                greedy_iterations=args.greedy_iterations,
                render=args.render,
                debug_decisions=args.debug_decisions,
                episode_seed=args.seed + episode_index,
            )
            episodes.append(episode)
            decisions.extend(episode_decisions)
            print(
                f"episode={episode_index} winner={episode['winner']} "
                f"reason={episode['termination_reason']} decisions={episode['decision_count']}",
                flush=True,
            )
    finally:
        env.close()

    output_dir = Path(args.output_dir)
    metadata = {
        "model": args.model,
        "ppo": args.ppo,
        "potential_model": args.potential_model,
        "p1_policy": _policy_label(policies[1], args.p1_shot, args.p1_defense),
        "p2_policy": _policy_label(policies[2], args.p2_shot, args.p2_defense),
        "episodes_requested": args.episodes,
        "steps_per_episode": args.steps,
        "seed": args.seed,
        "skill_profile": args.skill_profile,
        "planner": args.planner,
        "q_input_dim": 82,
        "q_state_dim": 74,
        "q_state_encoder": "v3_role_symmetric_canonical_state74_v2",
        "q_model_input_schema": "v3_state74_action_onehot_v1",
        "q_action_shape": [len(SKILL_NAMES), len(DEFENSE_POSITIONS)],
        "contact_metric_status": "inferred_from_crossings_and_termination_reasons",
        "landing_data_status": "not_recorded_by_random_v3_collection",
    }
    metadata.update(_fixed_configuration(policies, fixed_shots, fixed_defenses))
    summary, summary_rows, example_rows = build_summary(
        episodes,
        decisions,
        metadata=metadata,
        example_count=max(0, args.examples),
    )

    decision_csv_rows = [_decision_csv_row(row) for row in decisions]
    _write_csv(output_dir / "decisions.csv", decision_csv_rows)
    _write_json(output_dir / "decisions.json", {"metadata": metadata, "decisions": decisions})
    _write_csv(output_dir / "episodes.csv", episodes)
    _write_json(output_dir / "episodes.json", {"metadata": metadata, "episodes": episodes})
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "q_matrix_examples.csv", example_rows)

    print(f"Saved evaluation outputs under: {output_dir}")
    print(json.dumps({
        "episodes": len(episodes),
        "decisions": len(decisions),
        "p1_win_rate": summary["p1_win_rate"],
        "p2_win_rate": summary["p2_win_rate"],
        "outcome_rates": summary["outcome_rates"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
