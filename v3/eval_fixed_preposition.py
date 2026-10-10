#!/usr/bin/env python3
"""Evaluate fixed v3.0 attacker skill vs defender pre-position choices."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nash_skills.skills import SKILL_NAMES, SKILL_PROFILE_NAMES  # noqa: E402
from v3.constants import (  # noqa: E402
    DEFENSE_POSITIONS,
    GantryTarget,
    gantry_to_world_position,
    target_for_defender,
    validate_defense_position,
    world_to_gantry_target,
)
from v3.skill_semantics import (  # noqa: E402
    skill_player_relative_side,
    skill_target_xy,
    world_target_xy,
)


HISTORY = 4
PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
TABLE_SHIFT = 1.5


@dataclasses.dataclass
class CombinationSummary:
    planner: str
    attack_skill: str
    defense_position: str
    episodes: int
    attacker_wins: int
    defender_wins: int
    truncated_episodes: int
    attacker_contacts: int
    defender_contacts: int
    attacker_successes: int
    defender_successes: int
    total_steps: int
    rally_lengths: list[int]

    @property
    def attacker_win_rate(self) -> float | None:
        return self.attacker_wins / self.episodes if self.episodes else None

    @property
    def avg_steps(self) -> float | None:
        return self.total_steps / (self.episodes + self.truncated_episodes) if self.episodes + self.truncated_episodes else None

    @property
    def avg_rally_length(self) -> float | None:
        return float(np.mean(self.rally_lengths)) if self.rally_lengths else None


def gantry_for_player(env, player: int) -> tuple[float, float]:
    values = env.defender_gantry(player)
    return float(values[0]), float(values[1])


def position_snapshot(env, player: int) -> dict[str, float]:
    """Return the player's current position in both coordinate frames.

    ``defender_gantry`` exposes raw MuJoCo qpos.  Evaluation logs also need
    the corresponding physical/world position because P2's lateral qpos sign
    is opposite to world y.
    """
    qpos_x, qpos_y = gantry_for_player(env, player)
    world = gantry_to_world_position(player, GantryTarget(qpos_x, qpos_y))
    return {
        "qpos_x": qpos_x,
        "qpos_y": qpos_y,
        "world_x": float(world.x),
        "world_y": float(world.y),
    }


def player_relative_side(player: int, world_y: float, tolerance: float = 0.05) -> str:
    """Interpret world y in the named player's own left/center/right frame."""
    if abs(float(world_y)) <= tolerance:
        return "center"
    if player == 1:
        return "left" if world_y > 0 else "right"
    if player == 2:
        return "left" if world_y < 0 else "right"
    raise ValueError("player must be 1 or 2")


def defense_action_diagnostics(
    defender: int,
    semantic_action: str,
    commitment,
    actual: dict[str, float],
) -> dict[str, Any]:
    """Expose semantic, world, raw-qpos, and realized-side meanings."""
    return {
        "defense_semantic_action": semantic_action,
        "defense_world_y": float(commitment.target.y),
        "defender_raw_qpos_y": float(commitment.qpos_target.y),
        "defender_actual_world_y": float(actual["world_y"]),
        "defender_actual_raw_qpos_y": float(actual["qpos_y"]),
        "defense_target_player_relative_side": player_relative_side(defender, commitment.target.y),
        "defense_player_relative_side": player_relative_side(defender, actual["world_y"]),
    }


def attack_action_diagnostics(
    attacker: int,
    semantic_action: str,
    skill_profile: str,
) -> dict[str, Any]:
    """Expose attacker-relative and world target meanings for a skill."""
    local_x, local_y = skill_target_xy(semantic_action, profile=skill_profile)
    world_x, world_y = world_target_xy(attacker, semantic_action, profile=skill_profile)
    return {
        "attack_semantic_action": semantic_action,
        "attack_player_relative_side": skill_player_relative_side(
            semantic_action, profile=skill_profile
        ),
        "attack_target_local_y": float(local_y),
        "attack_target_world_x": float(world_x),
        "attack_target_world_y": float(world_y),
    }


def receiver_after_crossing(ball_x: float) -> int:
    return 2 if ball_x > TABLE_SHIFT else 1


def receiver_from_crossing(prev_ball_x: float, curr_ball_x: float) -> int | None:
    if (prev_ball_x - TABLE_SHIFT) * (curr_ball_x - TABLE_SHIFT) >= 0:
        return None
    return receiver_after_crossing(curr_ball_x)


def opponent(player: int) -> int:
    return 2 if player == 1 else 1


def maybe_render(env, enabled: bool) -> None:
    if enabled:
        env.render()
        time.sleep(0.01)


def blank_episode_row(
    *,
    combo_index: int,
    episode_index: int,
    planner: str,
    attack_skill: str,
    defense_position: str,
    reset_mode: str,
    skill_profile: str,
    gantry_speed_scale: float,
    max_steps: int,
) -> dict[str, Any]:
    return {
        "combo_index": combo_index,
        "episode_index": episode_index,
        "planner": planner,
        "attack_skill": attack_skill,
        "defense_position": defense_position,
        "reset_mode": reset_mode,
        "skill_profile": skill_profile,
        "gantry_speed_scale": gantry_speed_scale,
        "max_steps": max_steps,
        "decision_count": 0,
        "rally_length": 0,
        "physics_steps": 0,
        "truncated": False,
        "winner": "",
        "winner_role": "",
        "termination_reason": "",
        "attacker_contacts": 0,
        "defender_contacts": 0,
        "attacker_successes": 0,
        "defender_successes": 0,
        "successful_return": False,
        "miss": False,
        "out_of_bounds": False,
        "p1_starting_role": "",
        "p2_starting_role": "",
        "p1_attacker_decisions": 0,
        "p2_attacker_decisions": 0,
        "p1_defender_decisions": 0,
        "p2_defender_decisions": 0,
        "p1_attacker_wins": 0,
        "p2_attacker_wins": 0,
        "p1_defender_wins": 0,
        "p2_defender_wins": 0,
        "p1_contacts": 0,
        "p2_contacts": 0,
        "p1_successes": 0,
        "p2_successes": 0,
        "last_attacker": "",
        "last_defender": "",
        "defender_target_world_x": "",
        "defender_target_world_y": "",
        "defender_target_qpos_x": "",
        "defender_target_qpos_y": "",
        "defender_target_x": "",
        "defender_target_y": "",
        "defender_target_player": "",
        "defender_target_decision_index": "",
        "attack_semantic_action": "",
        "attack_player_relative_side": "",
        "attack_target_local_y": "",
        "attack_target_world_x": "",
        "attack_target_world_y": "",
        "defense_semantic_action": "",
        "defense_world_y": "",
        "defender_raw_qpos_y": "",
        "defender_actual_world_y": "",
        "defender_actual_raw_qpos_y": "",
        "defense_target_player_relative_side": "",
        "defense_player_relative_side": "",
        "defender_actual_qpos_x_at_commit": "",
        "defender_actual_qpos_y_at_commit": "",
        "defender_actual_world_x_at_commit": "",
        "defender_actual_world_y_at_commit": "",
        "defender_actual_x_at_commit": "",
        "defender_actual_y_at_commit": "",
        "defender_actual_qpos_x_at_receive": "",
        "defender_actual_qpos_y_at_receive": "",
        "defender_actual_world_x_at_receive": "",
        "defender_actual_world_y_at_receive": "",
        "defender_actual_x_at_receive": "",
        "defender_actual_y_at_receive": "",
        "defender_actual_at_receive_player": "",
        "defender_actual_at_receive_decision_index": "",
        "defender_actual_qpos_x_at_contact": "",
        "defender_actual_qpos_y_at_contact": "",
        "defender_actual_world_x_at_contact": "",
        "defender_actual_world_y_at_contact": "",
        "defender_actual_x_at_contact": "",
        "defender_actual_y_at_contact": "",
        "defender_actual_qpos_x_final": "",
        "defender_actual_qpos_y_final": "",
        "defender_actual_world_x_final": "",
        "defender_actual_world_y_final": "",
        "defender_actual_x_final": "",
        "defender_actual_y_final": "",
        "out_of_bounds_reason": "",
        "decision_events": [],
    }


def commit_decision(
    env,
    *,
    attacker: int,
    attack_skill: str,
    defense_position: str,
    skill_profile: str,
    row: dict[str, Any],
    physics_step: int,
    debug: bool,
) -> None:
    defender = opponent(attacker)
    if attacker == 1:
        env.set_skills(attack_skill, env._skill2)
    else:
        env.set_skills(env._skill1, attack_skill)
    commitment = env.set_defense_position(defender, defense_position)
    actual = position_snapshot(env, defender)
    semantic_diagnostics = {
        **attack_action_diagnostics(attacker, attack_skill, skill_profile),
        **defense_action_diagnostics(defender, defense_position, commitment, actual),
    }
    row["decision_count"] += 1
    row["last_attacker"] = attacker
    row["last_defender"] = defender
    row[f"p{attacker}_attacker_decisions"] += 1
    row[f"p{defender}_defender_decisions"] += 1
    if row["decision_count"] == 1:
        row[f"p{attacker}_starting_role"] = "attacker"
        row[f"p{defender}_starting_role"] = "defender"
    row["defender_target_world_x"] = commitment.target.x
    row["defender_target_world_y"] = commitment.target.y
    row["defender_target_qpos_x"] = commitment.qpos_target.x
    row["defender_target_qpos_y"] = commitment.qpos_target.y
    # Backward-compatible aliases: targets are world coordinates.
    row["defender_target_x"] = commitment.target.x
    row["defender_target_y"] = commitment.target.y
    row["defender_target_player"] = defender
    row["defender_target_decision_index"] = row["decision_count"]
    row.update(semantic_diagnostics)
    row["defender_actual_qpos_x_at_commit"] = actual["qpos_x"]
    row["defender_actual_qpos_y_at_commit"] = actual["qpos_y"]
    row["defender_actual_world_x_at_commit"] = actual["world_x"]
    row["defender_actual_world_y_at_commit"] = actual["world_y"]
    # Backward-compatible aliases: actual_* values are raw qpos.
    row["defender_actual_x_at_commit"] = actual["qpos_x"]
    row["defender_actual_y_at_commit"] = actual["qpos_y"]
    event = {
        "decision_index": row["decision_count"],
        "physics_step": physics_step,
        "attacker_player": attacker,
        "attacker_skill": attack_skill,
        "attack_semantic_action": semantic_diagnostics["attack_semantic_action"],
        "attack_player_relative_side": semantic_diagnostics["attack_player_relative_side"],
        "attack_target_local_y": semantic_diagnostics["attack_target_local_y"],
        "attack_target_world_x": semantic_diagnostics["attack_target_world_x"],
        "attack_target_world_y": semantic_diagnostics["attack_target_world_y"],
        "defender_player": defender,
        "defender_pre_position": defense_position,
        "defense_semantic_action": semantic_diagnostics["defense_semantic_action"],
        "defense_world_y": semantic_diagnostics["defense_world_y"],
        "defender_raw_qpos_y": semantic_diagnostics["defender_raw_qpos_y"],
        "defender_actual_world_y": semantic_diagnostics["defender_actual_world_y"],
        "defender_actual_raw_qpos_y": semantic_diagnostics["defender_actual_raw_qpos_y"],
        "defense_target_player_relative_side": semantic_diagnostics["defense_target_player_relative_side"],
        "defense_player_relative_side": semantic_diagnostics["defense_player_relative_side"],
        "defender_coordinate_frames": {
            "target": "world_and_raw_gantry_qpos",
            "actual": "world_and_raw_gantry_qpos",
        },
        "defender_target_world": [commitment.target.x, commitment.target.y],
        "defender_target_qpos": [commitment.qpos_target.x, commitment.qpos_target.y],
        "defender_actual_world_at_commit": [actual["world_x"], actual["world_y"]],
        "defender_actual_qpos_at_commit": [actual["qpos_x"], actual["qpos_y"]],
        "defender_actual_at_receive": None,
        "defender_actual_at_contact": None,
    }
    row["decision_events"].append(event)
    if debug:
        print(
            "decision "
            f"{event['decision_index']}: "
            f"attacker=P{attacker} skill={attack_skill} "
            f"defender=P{defender} pre_position={defense_position} "
            f"target_world=({commitment.target.x:.3f}, {commitment.target.y:.3f}) "
            f"target_qpos=({commitment.qpos_target.x:.3f}, {commitment.qpos_target.y:.3f})",
            flush=True,
        )


def update_receive_coordinate(env, row: dict[str, Any]) -> None:
    """Attach receive position to the decision that created the commitment.

    This runs immediately before the next crossing is committed. Therefore
    ``last_defender`` and the latest event both refer to the previous
    decision, not the decision that will be appended next.
    """
    events = row.get("decision_events") or []
    if not events:
        return
    event = events[-1]
    defender = event.get("defender_player")
    if defender not in (1, 2):
        return
    actual = position_snapshot(env, int(defender))
    event["defender_actual_world_at_receive"] = [actual["world_x"], actual["world_y"]]
    event["defender_actual_qpos_at_receive"] = [actual["qpos_x"], actual["qpos_y"]]
    event["defender_actual_at_receive"] = [actual["qpos_x"], actual["qpos_y"]]
    row["defender_actual_qpos_x_at_receive"] = actual["qpos_x"]
    row["defender_actual_qpos_y_at_receive"] = actual["qpos_y"]
    row["defender_actual_world_x_at_receive"] = actual["world_x"]
    row["defender_actual_world_y_at_receive"] = actual["world_y"]
    row["defender_actual_x_at_receive"] = actual["qpos_x"]
    row["defender_actual_y_at_receive"] = actual["qpos_y"]
    row["defender_actual_at_receive_player"] = int(defender)
    row["defender_actual_at_receive_decision_index"] = event["decision_index"]


def update_contact_coordinate(env, row: dict[str, Any], lines: list[str]) -> None:
    events = row.get("decision_events") or []
    if not events:
        return
    event = events[-1]
    defender = event.get("defender_player")
    if defender not in (1, 2):
        return
    defender_label = "ego" if int(defender) == 1 else "opp"
    if any(f"by {defender_label}" in line for line in lines):
        actual = position_snapshot(env, int(defender))
        event["defender_actual_world_at_contact"] = [actual["world_x"], actual["world_y"]]
        event["defender_actual_qpos_at_contact"] = [actual["qpos_x"], actual["qpos_y"]]
        event["defender_actual_at_contact"] = [actual["qpos_x"], actual["qpos_y"]]
        row["defender_actual_qpos_x_at_contact"] = actual["qpos_x"]
        row["defender_actual_qpos_y_at_contact"] = actual["qpos_y"]
        row["defender_actual_world_x_at_contact"] = actual["world_x"]
        row["defender_actual_world_y_at_contact"] = actual["world_y"]
        row["defender_actual_x_at_contact"] = actual["qpos_x"]
        row["defender_actual_y_at_contact"] = actual["qpos_y"]


def run_combination(
    *,
    planner: str,
    attack_skill: str,
    defense_position: str,
    episodes: int,
    max_steps: int,
    warmup: int,
    max_total_steps: int | None,
    ppo_path: str,
    reset_mode: str,
    skill_profile: str,
    gantry_speed_scale: float,
    seed: int | None,
    combo_index: int,
    render: bool,
    debug_decisions: bool,
) -> tuple[CombinationSummary, list[dict[str, Any]]]:
    from stable_baselines3 import PPO
    from nash_skills.eval_matchup import _capture_env_step, _infer_winner, _parse_contact_lines
    from v3.env_wrapper import V3PrePositionEnv

    if attack_skill not in SKILL_NAMES:
        raise ValueError(f"Unknown attack skill '{attack_skill}'. Choose from: {SKILL_NAMES}")
    validate_defense_position(defense_position)
    if planner not in {"legacy", "legacy_legal", "optimized"}:
        raise ValueError("planner must be 'legacy', 'legacy_legal', or 'optimized'")

    if seed is not None:
        np.random.seed(seed + combo_index)

    ppo = PPO.load(ppo_path)
    env = V3PrePositionEnv(
        proc_id=1,
        history=HISTORY,
        reset_mode=reset_mode,
        skill_profile=skill_profile,
        gantry_speed_scale=gantry_speed_scale,
        planner=planner,
    )
    env.set_skills("left", "right")

    obs, info = env.reset()
    prev_ball_x = float(obs[36])
    total_steps = 0

    while total_steps < warmup:
        action = build_ppo_action(ppo, obs, info)
        (obs, _, done, _, info), _lines = _capture_env_step(env, action)
        maybe_render(env, render)
        total_steps += 1
        curr_ball_x = float(obs[36])
        attacker = receiver_from_crossing(prev_ball_x, curr_ball_x)
        if attacker is not None:
            env.set_skills(attack_skill if attacker == 1 else env._skill1, attack_skill if attacker == 2 else env._skill2)
            env.set_defense_position(opponent(attacker), defense_position)
        prev_ball_x = curr_ball_x
        if done:
            obs, info = env.reset()
            prev_ball_x = float(obs[36])

    rows: list[dict[str, Any]] = []
    completed = 0
    truncated = 0
    attacker_wins = 0
    defender_wins = 0
    attacker_contacts = 0
    defender_contacts = 0
    attacker_successes = 0
    defender_successes = 0
    rally_lengths: list[int] = []

    while completed < episodes:
        if max_total_steps is not None and total_steps >= max_total_steps:
            break

        row = blank_episode_row(
            combo_index=combo_index,
            episode_index=len(rows),
            planner=planner,
            attack_skill=attack_skill,
            defense_position=defense_position,
            reset_mode=reset_mode,
            skill_profile=skill_profile,
            gantry_speed_scale=gantry_speed_scale,
            max_steps=max_steps,
        )
        env.set_skills("left", "right")
        env.clear_defense_position()
        obs, info = env.reset()
        prev_ball_x = float(obs[36])
        steps = 0
        rally_length = 0

        while True:
            if max_total_steps is not None and total_steps >= max_total_steps:
                row["truncated"] = True
                row["termination_reason"] = "max_total_steps"
                truncated += 1
                break

            action = build_ppo_action(ppo, obs, info)
            (obs, _, done, _, info), lines = _capture_env_step(env, action)
            maybe_render(env, render)
            total_steps += 1
            steps += 1

            e_c, o_c, e_s, o_s = _parse_contact_lines(lines)
            row["p1_contacts"] += e_c
            row["p2_contacts"] += o_c
            row["p1_successes"] += e_s
            row["p2_successes"] += o_s
            last_attacker = row.get("last_attacker")
            if last_attacker == 1:
                attacker_contacts += e_c
                attacker_successes += e_s
                defender_contacts += o_c
                defender_successes += o_s
                row["attacker_contacts"] += e_c
                row["attacker_successes"] += e_s
                row["defender_contacts"] += o_c
                row["defender_successes"] += o_s
            elif last_attacker == 2:
                attacker_contacts += o_c
                attacker_successes += o_s
                defender_contacts += e_c
                defender_successes += e_s
                row["attacker_contacts"] += o_c
                row["attacker_successes"] += o_s
                row["defender_contacts"] += e_c
                row["defender_successes"] += e_s

            update_contact_coordinate(env, row, lines)
            curr_ball_x = float(obs[36])

            attacker = receiver_from_crossing(prev_ball_x, curr_ball_x)
            if attacker is not None:
                update_receive_coordinate(env, row)
                rally_length += 1
                commit_decision(
                    env,
                    attacker=attacker,
                    attack_skill=attack_skill,
                    defense_position=defense_position,
                    skill_profile=skill_profile,
                    row=row,
                    physics_step=steps,
                    debug=debug_decisions or render,
                )

            prev_ball_x = curr_ball_x

            if done:
                winner = _infer_winner(obs, info)
                row["winner"] = winner
                winner_player = 1 if winner == "ego" else 2
                last_attacker = row.get("last_attacker")
                last_defender = row.get("last_defender")
                if winner_player == last_attacker:
                    row["winner_role"] = "attacker"
                    attacker_wins += 1
                elif winner_player == last_defender:
                    row["winner_role"] = "defender"
                    defender_wins += 1
                else:
                    row["winner_role"] = "unknown"
                row["termination_reason"] = str(info.get("termination_reason", "done"))
                row["out_of_bounds_reason"] = row["termination_reason"]
                row["successful_return"] = bool(row["attacker_successes"] or row["defender_successes"])
                row["miss"] = "miss" in row["termination_reason"]
                row["out_of_bounds"] = "out" in row["termination_reason"]
                if row["winner_role"] in ("attacker", "defender") and winner_player in (1, 2):
                    row[f"p{winner_player}_{row['winner_role']}_wins"] = 1
                completed += 1
                rally_lengths.append(rally_length)
                break

            if steps >= max_steps:
                row["truncated"] = True
                row["termination_reason"] = "max_steps"
                truncated += 1
                break

        row["physics_steps"] = steps
        row["rally_length"] = rally_length
        final_defender = row.get("last_defender")
        if final_defender in (1, 2):
            actual = position_snapshot(env, int(final_defender))
            row["defender_actual_qpos_x_final"] = actual["qpos_x"]
            row["defender_actual_qpos_y_final"] = actual["qpos_y"]
            row["defender_actual_world_x_final"] = actual["world_x"]
            row["defender_actual_world_y_final"] = actual["world_y"]
            row["defender_actual_x_final"] = actual["qpos_x"]
            row["defender_actual_y_final"] = actual["qpos_y"]
        row["decision_events"] = json.dumps(row["decision_events"])
        rows.append(row)

    env.close()

    summary = CombinationSummary(
        planner=planner,
        attack_skill=attack_skill,
        defense_position=defense_position,
        episodes=completed,
        attacker_wins=attacker_wins,
        defender_wins=defender_wins,
        truncated_episodes=truncated,
        attacker_contacts=attacker_contacts,
        defender_contacts=defender_contacts,
        attacker_successes=attacker_successes,
        defender_successes=defender_successes,
        total_steps=sum(int(row["physics_steps"]) for row in rows),
        rally_lengths=rally_lengths,
    )
    return summary, rows


def build_ppo_action(ppo, obs, info) -> np.ndarray:
    from nash_skills.eval_matchup import _build_obs1, _build_obs2

    obs1 = _build_obs1(obs, info)
    obs2 = _build_obs2(obs, info)
    action1, _ = ppo.predict(obs1, deterministic=True)
    action2, _ = ppo.predict(obs2, deterministic=True)
    action = np.zeros(18, dtype=np.float32)
    action[:9] = action1[:9]
    action[9:] = action2[:9]
    return action


def summary_row(summary: CombinationSummary) -> dict[str, Any]:
    p1_target = target_for_defender(1, summary.defense_position)
    p2_target = target_for_defender(2, summary.defense_position)
    p1_qpos = world_to_gantry_target(1, p1_target)
    p2_qpos = world_to_gantry_target(2, p2_target)
    return {
        "planner": summary.planner,
        "attack_skill": summary.attack_skill,
        "defense_position": summary.defense_position,
        "defender_target_world_x": p1_target.x,
        "defender_target_world_y": p1_target.y,
        "defender_target_qpos_x": "",
        "defender_target_qpos_y": "",
        "defender_target_qpos_x_p1": p1_qpos.x,
        "defender_target_qpos_y_p1": p1_qpos.y,
        "defender_target_qpos_x_p2": p2_qpos.x,
        "defender_target_qpos_y_p2": p2_qpos.y,
        "defender_target_x": p1_target.x,
        "defender_target_y": p1_target.y,
        "defender_target_y_p1": p1_target.y,
        "defender_target_y_p2": p2_target.y,
        "episodes": summary.episodes,
        "truncated_episodes": summary.truncated_episodes,
        "attacker_wins": summary.attacker_wins,
        "defender_wins": summary.defender_wins,
        "attacker_win_rate": summary.attacker_win_rate,
        "attacker_contacts": summary.attacker_contacts,
        "defender_contacts": summary.defender_contacts,
        "attacker_successes": summary.attacker_successes,
        "defender_successes": summary.defender_successes,
        "avg_steps": summary.avg_steps,
        "avg_rally_length": summary.avg_rally_length,
    }


def safe_rate(numerator: float, denominator: float) -> float | str:
    return numerator / denominator if denominator else ""


def mean_int(rows: list[dict[str, Any]], key: str) -> float | str:
    values = [int(row[key]) for row in rows if row.get(key) not in ("", None)]
    return float(np.mean(values)) if values else ""


def build_summary_rows(rows: list[dict[str, Any]], summaries: list[CombinationSummary]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["attack_skill"]), str(row["defense_position"])), []).append(row)

    output: list[dict[str, Any]] = []
    for summary in summaries:
        key = (summary.attack_skill, summary.defense_position)
        combo_rows = grouped.get(key, [])
        p1_target = target_for_defender(1, summary.defense_position)
        p2_target = target_for_defender(2, summary.defense_position)
        p1_qpos = world_to_gantry_target(1, p1_target)
        p2_qpos = world_to_gantry_target(2, p2_target)
        completed_rows = [row for row in combo_rows if not row.get("truncated") and row.get("winner")]
        episode_count = len(combo_rows)
        output.append({
            "planner": summary.planner,
            "attack_skill": summary.attack_skill,
            "defense_position": summary.defense_position,
            "split": "all",
            "physical_player": "",
            "defender_target_world_x": p1_target.x,
            "defender_target_world_y": p1_target.y,
            "defender_target_qpos_x": "",
            "defender_target_qpos_y": "",
            "defender_target_qpos_x_p1": p1_qpos.x,
            "defender_target_qpos_y_p1": p1_qpos.y,
            "defender_target_qpos_x_p2": p2_qpos.x,
            "defender_target_qpos_y_p2": p2_qpos.y,
            "defender_target_x": p1_target.x,
            "defender_target_y": p1_target.y,
            "defender_target_y_p1": p1_target.y,
            "defender_target_y_p2": p2_target.y,
            "episode_count": episode_count,
            "completed_episodes": len(completed_rows),
            "truncated_episodes": sum(1 for row in combo_rows if row.get("truncated")),
            "attacker_win_rate": safe_rate(sum(1 for row in completed_rows if row.get("winner_role") == "attacker"), len(completed_rows)),
            "defender_return_success_rate": safe_rate(sum(int(row["defender_successes"]) for row in combo_rows), sum(int(row["defender_contacts"]) for row in combo_rows)),
            "miss_rate": safe_rate(sum(1 for row in combo_rows if row.get("miss")), episode_count),
            "out_of_bounds_rate": safe_rate(sum(1 for row in combo_rows if row.get("out_of_bounds")), episode_count),
            "avg_rally_length": mean_int(combo_rows, "rally_length"),
            "avg_physics_steps": mean_int(combo_rows, "physics_steps"),
            "attacker_decisions": sum(int(row["decision_count"]) for row in combo_rows),
            "defender_decisions": sum(int(row["decision_count"]) for row in combo_rows),
            "attacker_wins": sum(1 for row in completed_rows if row.get("winner_role") == "attacker"),
            "defender_wins": sum(1 for row in completed_rows if row.get("winner_role") == "defender"),
            "player_contacts": "",
            "player_successes": "",
            "player_return_success_rate": "",
        })

        for player in (1, 2):
            target = target_for_defender(player, summary.defense_position)
            player_contacts = sum(int(row[f"p{player}_contacts"]) for row in combo_rows)
            player_successes = sum(int(row[f"p{player}_successes"]) for row in combo_rows)
            attacker_decisions = sum(int(row[f"p{player}_attacker_decisions"]) for row in combo_rows)
            defender_decisions = sum(int(row[f"p{player}_defender_decisions"]) for row in combo_rows)
            player_attacker_wins = sum(int(row[f"p{player}_attacker_wins"]) for row in combo_rows)
            player_defender_wins = sum(int(row[f"p{player}_defender_wins"]) for row in combo_rows)
            output.append({
                "planner": summary.planner,
                "attack_skill": summary.attack_skill,
                "defense_position": summary.defense_position,
                "split": "physical_player",
                "physical_player": player,
                "defender_target_world_x": target.x,
                "defender_target_world_y": target.y,
                "defender_target_qpos_x": world_to_gantry_target(player, target).x,
                "defender_target_qpos_y": world_to_gantry_target(player, target).y,
                "defender_target_qpos_x_p1": p1_qpos.x if player == 1 else "",
                "defender_target_qpos_y_p1": p1_qpos.y if player == 1 else "",
                "defender_target_qpos_x_p2": p2_qpos.x if player == 2 else "",
                "defender_target_qpos_y_p2": p2_qpos.y if player == 2 else "",
                "defender_target_x": target.x,
                "defender_target_y": target.y,
                "defender_target_y_p1": p1_target.y,
                "defender_target_y_p2": p2_target.y,
                "episode_count": episode_count,
                "completed_episodes": len(completed_rows),
                "truncated_episodes": sum(1 for row in combo_rows if row.get("truncated")),
                "attacker_win_rate": safe_rate(player_attacker_wins, attacker_decisions),
                "defender_return_success_rate": safe_rate(player_successes, player_contacts),
                "miss_rate": "",
                "out_of_bounds_rate": "",
                "avg_rally_length": mean_int(combo_rows, "rally_length"),
                "avg_physics_steps": mean_int(combo_rows, "physics_steps"),
                "attacker_decisions": attacker_decisions,
                "defender_decisions": defender_decisions,
                "attacker_wins": player_attacker_wins,
                "defender_wins": player_defender_wins,
                "player_contacts": player_contacts,
                "player_successes": player_successes,
                "player_return_success_rate": safe_rate(player_successes, player_contacts),
            })
    return output


def write_outputs(output_csv: str, output_json: str, summaries: list[CombinationSummary], rows: list[dict[str, Any]]) -> None:
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    summary_path = Path(output_csv)
    episode_path = summary_path.with_name(f"{summary_path.stem}_episodes{summary_path.suffix}")

    summary_rows = build_summary_rows(rows, summaries)
    if not summary_rows:
        summary_rows = [summary_row(s) for s in summaries]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    episode_fields = list(rows[0].keys()) if rows else list(blank_episode_row(
        combo_index=0,
        episode_index=0,
        planner=summaries[0].planner if summaries else "legacy",
        attack_skill="left",
        defense_position="center",
        reset_mode="ready",
        skill_profile="aggressive",
        gantry_speed_scale=1.0,
        max_steps=0,
    ).keys())
    with episode_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=episode_fields)
        writer.writeheader()
        writer.writerows(rows)

    with Path(output_json).open("w") as f:
        json.dump(
            {
                "metadata": {
                    "evaluator": "v3.eval_fixed_preposition",
                    "planner": summaries[0].planner if summaries else "legacy",
                },
                "summaries": summary_rows,
                "episodes": rows,
            },
            f,
            indent=2,
        )

    print(f"Summary CSV saved to: {summary_path}")
    print(f"Episode CSV saved to: {episode_path}")
    print(f"JSON saved to: {output_json}")


def combinations(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.all_combinations:
        return [(skill, pos) for skill in SKILL_NAMES for pos in DEFENSE_POSITIONS]
    return [(args.attack_skill, args.defense_position)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v3.0 fixed attacker skill vs defender pre-position evaluator")
    parser.add_argument("--attack-skill", choices=SKILL_NAMES, default="left")
    parser.add_argument("--defense-position", choices=DEFENSE_POSITIONS, default="center")
    parser.add_argument("--all-combinations", action="store_true", help="Run all 5 shot skills x 3 defense positions.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--max-total-steps", type=int, default=None)
    parser.add_argument("--ppo", default=PPO_MODEL_PATH)
    parser.add_argument("--reset-mode", choices=["clean", "ready", "carryover"], default="ready")
    parser.add_argument("--skill-profile", choices=SKILL_PROFILE_NAMES, default="aggressive")
    parser.add_argument("--gantry-speed-scale", type=float, default=1.0)
    parser.add_argument("--planner", choices=["legacy", "legacy_legal", "optimized"], default="legacy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the existing live MuJoCo viewer via env.render(); do not use MUJOCO_GL=egl for this mode.",
    )
    parser.add_argument(
        "--debug-decisions",
        action="store_true",
        help="Print attacker/defender roles and defender target at every net-crossing decision.",
    )
    parser.add_argument("--output-csv", default="skill_eval/v3/fixed_preposition_summary.csv")
    parser.add_argument("--output-json", default="skill_eval/v3/fixed_preposition_results.json")
    return parser


def planner_output_path(path: str, planner: str, default_path: str) -> str:
    """Keep optimized default artifacts separate from the legacy artifacts."""
    if planner == "legacy" or path != default_path:
        return path
    output = Path(path)
    return str(output.with_name(f"{output.stem}_optimized{output.suffix}"))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_csv = planner_output_path(
        args.output_csv,
        args.planner,
        "skill_eval/v3/fixed_preposition_summary.csv",
    )
    args.output_json = planner_output_path(
        args.output_json,
        args.planner,
        "skill_eval/v3/fixed_preposition_results.json",
    )
    print(f"Planner: {args.planner}", flush=True)
    all_summaries: list[CombinationSummary] = []
    all_rows: list[dict[str, Any]] = []

    for combo_index, (attack_skill, defense_position) in enumerate(combinations(args)):
        print(
            f"\n=== planner={args.planner} {attack_skill} vs defense {defense_position} ===",
            flush=True,
        )
        summary, rows = run_combination(
            planner=args.planner,
            attack_skill=attack_skill,
            defense_position=defense_position,
            episodes=args.episodes,
            max_steps=args.steps,
            warmup=args.warmup,
            max_total_steps=args.max_total_steps,
            ppo_path=args.ppo,
            reset_mode=args.reset_mode,
            skill_profile=args.skill_profile,
            gantry_speed_scale=args.gantry_speed_scale,
            seed=args.seed,
            combo_index=combo_index,
            render=args.render,
            debug_decisions=args.debug_decisions,
        )
        all_summaries.append(summary)
        all_rows.extend(rows)
        win_rate = "n/a" if summary.attacker_win_rate is None else f"{summary.attacker_win_rate:.3f}"
        print(
            f"episodes={summary.episodes} truncated={summary.truncated_episodes} "
            f"attacker_wr={win_rate} avg_steps={summary.avg_steps}",
            flush=True,
        )

    write_outputs(args.output_csv, args.output_json, all_summaries, all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
