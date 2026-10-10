#!/usr/bin/env python3
"""Audit rendered V3 defensive and offensive left/right semantics.

This script does not change any mapping or controller. It records the live
transition from one crossing decision to the next and writes a target table
for the selected skill profile.
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

from v3.skill_semantics import SKILL_NAMES, get_skill, skill_target_xy, world_target_xy  # noqa: E402
from v3.constants import gantry_to_world_position, GantryTarget, target_for_defender, world_to_gantry_target  # noqa: E402
from v3.diagnose_defense_semantics import _world_axes  # noqa: E402
from v3.env_wrapper import V3PrePositionEnv  # noqa: E402


TABLE_SHIFT = 1.5


def _receiver_after_crossing(ball_x: float) -> int:
    return 2 if ball_x > TABLE_SHIFT else 1


def _crossed(prev_x: float, curr_x: float) -> bool:
    return (prev_x - TABLE_SHIFT) * (curr_x - TABLE_SHIFT) < 0


def _pose_local(env: V3PrePositionEnv, player: int) -> list[float]:
    pose = env._env.curr_target if player == 1 else env._env.curr_target_opp
    return np.asarray(pose, dtype=float).tolist()


def _side_from_world_y(world_y: float, own_left_world_y: float) -> str:
    if abs(world_y) < 1e-9:
        return "center"
    return "left" if np.sign(world_y) == np.sign(own_left_world_y) else "right"


def offensive_target_rows(env: V3PrePositionEnv, profile: str) -> list[dict[str, Any]]:
    frames = {player: _world_axes(env, player) for player in (1, 2)}
    rows: list[dict[str, Any]] = []
    for attacker in (1, 2):
        receiver = 2 if attacker == 1 else 1
        receiver_left_y = frames[receiver]["world_y_for_own_left"]
        for skill in SKILL_NAMES:
            local_x, local_y = skill_target_xy(skill, profile=profile)
            world_x, world_y = world_target_xy(attacker, skill, profile=profile)
            rows.append(
                {
                    "profile": profile,
                    "attacker_id": attacker,
                    "receiver_id": receiver,
                    "skill": skill,
                    "local_target_x": local_x,
                    "local_target_y": local_y,
                    "world_target_x": world_x,
                    "world_target_y": world_y,
                    "attacker_perspective_side": _side_from_world_y(
                        local_y, 1.0
                    ),
                    "receiver_perspective_side": _side_from_world_y(world_y, receiver_left_y),
                }
            )
    return rows


def _build_ppo_action(ppo: Any, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    from nash_skills.eval_matchup import _build_obs1, _build_obs2

    obs1 = _build_obs1(obs, info)
    obs2 = _build_obs2(obs, info)
    actions, _ = ppo.predict(np.asarray([obs1, obs2]), deterministic=True)
    action1, action2 = actions[0], actions[1]
    action = np.zeros(18, dtype=np.float32)
    action[:9] = action1[:9]
    action[9:] = action2[:9]
    return action


def _planner_target(env: V3PrePositionEnv, player: int) -> list[float]:
    plan = env._latest_plans.get(player) or {}
    target = plan.get("target_pose_local")
    if target is None:
        return []
    return np.asarray(target, dtype=float).tolist()


def trace_episode(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from stable_baselines3 import PPO

    ppo = PPO.load(args.ppo)
    env = V3PrePositionEnv(
        proc_id=1,
        reset_mode="ready",
        skill_profile=args.skill_profile,
        planner=args.planner,
    )
    env.set_skills("left", "right")
    obs, info = env.reset(seed=args.seed)
    prev_ball_x = float(obs[36])
    steps: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    active_decision: dict[str, Any] | None = None

    try:
        for physics_step in range(1, args.steps + 1):
            commitment = env.defense_commitment
            defender = commitment.defender if commitment else ""
            target_world_y = commitment.target.y if commitment else ""
            target_qpos_y = commitment.qpos_target.y if commitment else ""
            actual_before = env.defender_gantry(defender)[1] if defender in (1, 2) else ""
            actual_world_before = (
                gantry_to_world_position(
                    defender,
                    GantryTarget(*env.defender_gantry(defender)),
                ).y
                if defender in (1, 2) else ""
            )
            attacker = active_decision["attacker_id"] if active_decision else ""
            attacker_pose_before = _pose_local(env, attacker) if attacker in (1, 2) else []
            step_row = {
                "physics_step": physics_step,
                "ball_x_before": prev_ball_x,
                "active_attacker_id": attacker,
                "active_defender_id": defender,
                "defense_target_y": target_world_y,
                "defense_target_world_y": target_world_y,
                "defense_target_qpos_y": target_qpos_y,
                "defender_actual_y_before_step": actual_before,
                "defender_actual_world_y_before_step": actual_world_before,
                "attacker_racket_target_pose_local_before": attacker_pose_before,
                "phase": "defensive_preposition" if commitment else "no_commitment",
            }

            action = _build_ppo_action(ppo, obs, info)
            obs, _, done, _, info = env.step(action)
            curr_ball_x = float(obs[36])

            if active_decision and physics_step == active_decision["first_execution_step"]:
                active_decision["offensive_target_first_observed_step"] = physics_step
                active_decision["offensive_planner_target_local_first_observed"] = _planner_target(
                    env, active_decision["attacker_id"]
                )
                active_decision["attacker_actual_y_after_first_execution_step"] = float(
                    env.defender_gantry(active_decision["attacker_id"])[1]
                )

            attacker_after = active_decision["attacker_id"] if active_decision else ""
            step_row.update(
                {
                    "ball_x_after": curr_ball_x,
                    "defender_actual_y_after_step": (
                        env.defender_gantry(defender)[1] if defender in (1, 2) else ""
                    ),
                    "attacker_racket_target_pose_local_after": (
                        _pose_local(env, attacker_after) if attacker_after in (1, 2) else []
                    ),
                    "attacker_planner_target_local_after": (
                        _planner_target(env, attacker_after) if attacker_after in (1, 2) else []
                    ),
                }
            )
            steps.append(step_row)

            if _crossed(prev_ball_x, curr_ball_x):
                next_attacker = _receiver_after_crossing(curr_ball_x)
                next_defender = 2 if next_attacker == 1 else 1
                actual_attacker_y_before_return = float(env.defender_gantry(next_attacker)[1])
                actual_defender_y_before_commit = float(env.defender_gantry(next_defender)[1])
                if next_attacker == 1:
                    env.set_skills(args.attack_skill, env._skill2)
                else:
                    env.set_skills(env._skill1, args.attack_skill)
                new_commitment = env.set_defense_position(next_defender, args.defense_position)
                decision = {
                    "decision_index": len(decisions) + 1,
                    "crossing_physics_step": physics_step,
                    "attacker_id": next_attacker,
                    "defender_id": next_defender,
                    "attack_skill": args.attack_skill,
                    "defense_position": args.defense_position,
                    "defense_target_y": new_commitment.target.y,
                    "defense_target_world_y": new_commitment.target.y,
                    "defense_target_qpos_y": new_commitment.qpos_target.y,
                    "defense_target_applied_step": physics_step,
                    "defense_target_coordinate_frame": "semantic_world_plus_raw_qpos",
                    "defender_actual_y_before_commit": actual_defender_y_before_commit,
                    "attacker_actual_y_before_offensive_execution": actual_attacker_y_before_return,
                    "offensive_skill_set_step": physics_step,
                    "first_execution_step": physics_step + 1,
                    "offensive_target_local": list(skill_target_xy(args.attack_skill, profile=args.skill_profile)),
                    "offensive_target_world": list(world_target_xy(next_attacker, args.attack_skill, profile=args.skill_profile)),
                    "offensive_planner_target_local_first_observed": [],
                }
                decisions.append(decision)
                active_decision = decision

            prev_ball_x = curr_ball_x
            if done:
                break
    finally:
        env.close()
    return steps, decisions


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo", default="logs/best_model_tracker1/best_model")
    parser.add_argument("--attack-skill", default="right")
    parser.add_argument("--defense-position", default="left")
    parser.add_argument("--skill-profile", default="aggressive")
    parser.add_argument("--planner", choices=["legacy", "legacy_legal", "optimized"], default="legacy")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="v3/diagnostics/rendered_semantics")
    args = parser.parse_args()
    if args.episodes != 1:
        raise ValueError("This diagnostic intentionally traces exactly one episode")

    env = V3PrePositionEnv(proc_id=1, reset_mode="ready", skill_profile=args.skill_profile, planner=args.planner)
    env.reset(seed=args.seed)
    target_rows = offensive_target_rows(env, args.skill_profile)
    env.close()
    step_rows, decision_rows = trace_episode(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "offensive_target_semantics.csv", target_rows)
    _write_csv(output_dir / "episode_trace_steps.csv", step_rows)
    _write_csv(output_dir / "episode_trace_decisions.csv", decision_rows)
    (output_dir / "offensive_target_semantics.json").write_text(json.dumps(target_rows, indent=2))
    (output_dir / "episode_trace_decisions.json").write_text(json.dumps(decision_rows, indent=2))
    print("Offensive target semantics:")
    for row in target_rows:
        print(
            f"P{row['attacker_id']} -> P{row['receiver_id']} {row['skill']}: "
            f"local=({row['local_target_x']:.3f},{row['local_target_y']:.3f}) "
            f"world=({row['world_target_x']:.3f},{row['world_target_y']:.3f}) "
            f"attacker_side={row['attacker_perspective_side']} "
            f"receiver_side={row['receiver_perspective_side']}"
        )
    print(f"\nWrote {output_dir / 'offensive_target_semantics.csv'}")
    print(f"Wrote {output_dir / 'episode_trace_steps.csv'}")
    print(f"Wrote {output_dir / 'episode_trace_decisions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
