#!/usr/bin/env python3
"""Trace matched ``right`` attack -> ``left`` defense transitions.

This is a read-only runtime diagnostic.  It does not change any skill or
defensive mapping.  It records the position at contact, immediately after
contact, and when the same physical player becomes the next defender.  The
crossing step is separated from the first step on which the V3 pre-position
override can affect the action.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from v3.audit_rendered_semantics import _build_ppo_action, _crossed, _receiver_after_crossing  # noqa: E402
from v3.constants import (  # noqa: E402
    GantryTarget,
    gantry_to_world_position,
    target_for_defender,
    world_to_gantry_target,
)
from v3.env_wrapper import V3PrePositionEnv  # noqa: E402
from v3.skill_semantics import world_target_xy  # noqa: E402


TABLE_SHIFT = 1.5
RACKET_BODY = {1: "tennis_racket", 2: "tennis_racket_opp"}
BASE_BODY = {1: "base", 2: "base_opp"}
PLAYER_HITTER = {1: "ego", 2: "opp"}


def _gantry(env: V3PrePositionEnv, player: int) -> np.ndarray:
    return env.defender_gantry(player)


def _racket_world(env: V3PrePositionEnv, player: int) -> np.ndarray:
    return np.asarray(env._env.data.body(RACKET_BODY[player]).xpos, dtype=float).copy()


def _racket_local(env: V3PrePositionEnv, player: int) -> np.ndarray:
    return env._local_pose(player)[:3].copy()


def _base_world(env: V3PrePositionEnv, player: int) -> np.ndarray:
    return np.asarray(env._env.data.body(BASE_BODY[player]).xpos, dtype=float).copy()


def _position_fields(env: V3PrePositionEnv, player: int, prefix: str) -> dict[str, Any]:
    gantry = _gantry(env, player)
    base_world = _base_world(env, player)
    racket_world = _racket_world(env, player)
    racket_local = _racket_local(env, player)
    return {
        f"{prefix}_gantry_x_qpos": float(gantry[0]),
        f"{prefix}_gantry_y_qpos": float(gantry[1]),
        f"{prefix}_gantry_base_world_x": float(base_world[0]),
        f"{prefix}_gantry_base_world_y": float(base_world[1]),
        f"{prefix}_racket_world_x": float(racket_world[0]),
        f"{prefix}_racket_world_y": float(racket_world[1]),
        f"{prefix}_racket_world_z": float(racket_world[2]),
        f"{prefix}_racket_planner_local_x": float(racket_local[0]),
        f"{prefix}_racket_planner_local_y": float(racket_local[1]),
        f"{prefix}_racket_planner_local_z": float(racket_local[2]),
    }


def _contact_event(env: V3PrePositionEnv, player: int) -> dict[str, Any] | None:
    for event in reversed(env._episode_events):
        if int(event.get("player", -1)) == player:
            return dict(event)
    return None


def _new_attack_row(
    env: V3PrePositionEnv,
    *,
    episode_id: int,
    decision_index: int,
    attacker: int,
    defender: int,
    skill_profile: str,
    crossing_step: int,
) -> dict[str, Any]:
    current_defender_target = target_for_defender(defender, "left")
    next_defender_target = target_for_defender(attacker, "left")
    current_defender_qpos_target = world_to_gantry_target(defender, current_defender_target)
    next_defender_qpos_target = world_to_gantry_target(attacker, next_defender_target)
    target_x, target_y = world_target_xy(attacker, "right", profile=skill_profile)
    row: dict[str, Any] = {
        "episode_id": episode_id,
        "decision_index": decision_index,
        "attacker_id": attacker,
        "defender_id": defender,
        "attack_skill": "right",
        "defense_position": "left",
        "offensive_target_world_x": target_x,
        "offensive_target_world_y": target_y,
        "current_defender_left_target_world_x": current_defender_target.x,
        "current_defender_left_target_world_y": current_defender_target.y,
        "current_defender_target_applied_qpos_y": current_defender_qpos_target.y,
        "current_defender_target_effective_base_world_y": current_defender_target.y,
        "next_defender_left_target_world_x": next_defender_target.x,
        "next_defender_left_target_world_y": next_defender_target.y,
        "next_defender_target_applied_qpos_y": next_defender_qpos_target.y,
        "next_defender_target_effective_base_world_y": next_defender_target.y,
        "decision_crossing_step": crossing_step,
        "current_defender_preposition_control_first_step": crossing_step + 1,
        "contact_step": "",
        "next_defender_crossing_step": "",
        "defense_commitment_set_step": "",
        "first_preposition_control_step": "",
        "post_contact_to_defensive_target_distance_qpos_y": "",
        "next_defender_to_target_distance_qpos_y": "",
    }
    row.update(_position_fields(env, attacker, "decision_start_attacker"))
    row.update(_position_fields(env, defender, "current_defender_at_decision_start"))
    return row


def trace_episode(
    *,
    ppo_path: str,
    skill_profile: str,
    planner: str,
    seed: int,
    steps: int,
    episode_id: int,
) -> list[dict[str, Any]]:
    from stable_baselines3 import PPO

    ppo = PPO.load(ppo_path)
    env = V3PrePositionEnv(
        proc_id=1,
        reset_mode="ready",
        skill_profile=skill_profile,
        planner=planner,
    )
    rows: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    active_decision: dict[str, Any] | None = None
    prev_hitter: str | None = None

    try:
        env.set_skills("right", "right")
        obs, info = env.reset(seed=seed)
        prev_ball_x = float(obs[36])
        prev_hitter = env._env.rally_state.get("last_hitter")

        for physics_step in range(1, steps + 1):
            # This is the state immediately before the action for the first
            # step after a detected contact.
            if pending is not None and physics_step == int(pending["contact_step"]) + 1:
                pending.update(_position_fields(env, int(pending["attacker_id"]), "immediately_after_contact"))

            action = _build_ppo_action(ppo, obs, info)
            obs, _, done, _, info = env.step(action)
            curr_ball_x = float(obs[36])
            after_hitter = env._env.rally_state.get("last_hitter")

            if (
                active_decision is not None
                and after_hitter != prev_hitter
                and after_hitter == PLAYER_HITTER[int(active_decision["attacker_id"])]
            ):
                attacker = int(active_decision["attacker_id"])
                event = _contact_event(env, attacker)
                active_decision["contact_step"] = physics_step
                active_decision.update(_position_fields(env, attacker, "at_contact"))
                active_decision.update(
                    _position_fields(
                        env,
                        int(active_decision["defender_id"]),
                        "current_defender_at_attacker_contact",
                    )
                )
                active_decision["planner_contact_event"] = event or {}
                if event:
                    active_decision["planner_selected_interception_local"] = event.get("selected_interception_point", "")
                    active_decision["planner_predicted_landing_world_x"] = event.get("predicted_landing_x", "")
                    active_decision["planner_predicted_landing_world_y"] = event.get("predicted_landing_y", "")
                    active_decision["planner_target_racket_pose_world"] = event.get("target_racket_pose", "")
                    active_decision["planner_local_target_racket_pose"] = event.get("planner_local_target_pose", "")
                    active_decision["planner_local_actual_racket_pose"] = event.get("planner_local_racket_pose", "")
                pending = active_decision

            if _crossed(prev_ball_x, curr_ball_x):
                next_attacker = _receiver_after_crossing(curr_ball_x)
                next_defender = 2 if next_attacker == 1 else 1

                if pending is not None and int(pending["attacker_id"]) == next_defender:
                    pending["next_defender_crossing_step"] = physics_step
                    pending["defense_commitment_set_step"] = physics_step
                    pending["first_preposition_control_step"] = physics_step + 1
                    pending.update(_position_fields(env, next_defender, "when_next_defender"))
                    pending["post_contact_to_defensive_target_distance_qpos_y"] = abs(
                        float(pending["at_contact_gantry_y_qpos"])
                        - float(pending["next_defender_target_applied_qpos_y"])
                    )
                    pending["next_defender_to_target_distance_qpos_y"] = abs(
                        float(pending["when_next_defender_gantry_y_qpos"])
                        - float(pending["next_defender_target_applied_qpos_y"])
                    )
                    pending["post_contact_to_defensive_target_distance_world_y"] = abs(
                        float(pending["at_contact_gantry_base_world_y"])
                        - float(pending["next_defender_left_target_world_y"])
                    )
                    pending["next_defender_to_target_distance_world_y"] = abs(
                        float(pending["when_next_defender_gantry_base_world_y"])
                        - float(pending["next_defender_left_target_world_y"])
                    )
                    pending = None

                # The newly received attack is always right, and the player
                # who just hit becomes the left-positioned defender.
                if next_attacker == 1:
                    env.set_skills("right", env._skill2)
                else:
                    env.set_skills(env._skill1, "right")
                env.set_defense_position(next_defender, "left")
                active_decision = _new_attack_row(
                    env,
                    episode_id=episode_id,
                    decision_index=len(rows) + 1,
                    attacker=next_attacker,
                    defender=next_defender,
                    skill_profile=skill_profile,
                    crossing_step=physics_step,
                )
                rows.append(active_decision)

            prev_ball_x = curr_ball_x
            prev_hitter = after_hitter
            if done:
                break
    finally:
        env.close()

    # Keep the output flat enough for CSV while retaining structured planner
    # values in the JSON artifact.
    for row in rows:
        row["planner_contact_event_json"] = json.dumps(row.pop("planner_contact_event", {}), default=str)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo", default="logs/best_model_tracker1/best_model")
    parser.add_argument("--skill-profile", default="aggressive")
    parser.add_argument("--planner", choices=["legacy", "legacy_legal", "optimized"], default="legacy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=190)
    parser.add_argument("--output-dir", default="v3/diagnostics/right_left_transition")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for episode in range(args.episodes):
        rows.extend(
            trace_episode(
                ppo_path=args.ppo,
                skill_profile=args.skill_profile,
                planner=args.planner,
                seed=args.seed + episode,
                steps=args.steps,
                episode_id=episode,
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "transitions.csv", rows)
    (output_dir / "transitions.json").write_text(json.dumps(rows, indent=2, default=str))

    print("right-attack -> left-defense transitions")
    for row in rows:
        print(
            f"episode={row['episode_id']} decision={row['decision_index']} "
            f"attacker=P{row['attacker_id']} contact={row['contact_step']} "
            f"next_defender_step={row['next_defender_crossing_step']} "
            f"control_first={row['first_preposition_control_step']} "
            f"contact_y={row.get('at_contact_gantry_y_qpos', '')} "
            f"next_defender_y={row.get('when_next_defender_gantry_y_qpos', '')} "
            f"current_defender_y_at_contact={row.get('current_defender_at_attacker_contact_gantry_y_qpos', '')} "
            f"current_target_world_y={row['current_defender_left_target_world_y']} "
            f"current_target_qpos_y={row['current_defender_target_applied_qpos_y']} "
            f"next_target_world_y={row['next_defender_left_target_world_y']} "
            f"next_target_qpos_y={row['next_defender_target_applied_qpos_y']} "
            f"distance={row['post_contact_to_defensive_target_distance_qpos_y']}"
        )
    print(f"Wrote {output_dir / 'transitions.csv'}")
    print(f"Wrote {output_dir / 'transitions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
