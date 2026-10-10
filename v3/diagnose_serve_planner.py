#!/usr/bin/env python3
"""Compare legacy and optimized racket planning on saved fixed-skill serves.

The source diagnostic CSVs predate V3 planner instrumentation, so this script
reconstructs each saved episode's reset using the same deterministic episode
seed and replays it twice.  Both planner runs use the same PPO checkpoint,
skills, reset configuration, and initial random state.  No planner code is
changed by this diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nash_skills.skills import SKILL_NAMES, skill_index  # noqa: E402
from v3.eval_fixed_preposition import HISTORY, build_ppo_action  # noqa: E402
from v3.racket_target_planner import (  # noqa: E402
    GRAVITY,
    TABLE_Z,
    estimate_legacy,
    mirror_position_to_world,
    _predict_receiver_bounce,
)
from v3.planner_env import PlannerSkillEnv  # noqa: E402


DT = 0.01
DEFAULT_SOURCE = "data/diagnostics_skill_randomization/fixed_pair_ready_seed0_n20/p2_right/episodes.csv"
DEFAULT_OUTPUT_DIR = "skill_eval/v3/serve_planner_diagnostic"


def episode_rng_seed(seed: int, skill1: str, episode_index: int, skill2: str) -> int:
    """Match diagnostic_fixed_skill.py's saved-episode reset seed."""
    return int(
        (
            (seed + 1_000_003) * 1_000_003
            + skill_index(skill1) * 10_007
            + skill_index(skill2) * 101
            + episode_index
        )
        % (2**32 - 1)
    )


def load_rows(path: str, skill1: str, skill2: str, failure_only: bool) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    selected = [
        row for row in rows
        if row.get("player1_skill") == skill1 and row.get("player2_skill") == skill2
    ]
    if failure_only:
        selected = [
            row for row in selected
            if str(row.get("successful_returns", "0")) in {"", "0"}
            or str(row.get("winner", "")).lower() == "truncated"
        ]
    if not selected:
        raise ValueError(
            f"No rows found for {skill1} vs {skill2} in {path}; "
            "use --include-successes or provide the matching saved episode CSV."
        )
    return selected


def load_metadata(source_csv: str) -> dict[str, Any]:
    metadata_path = Path(source_csv).parent / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return {}


def pose_snapshot(env: PlannerSkillEnv, player: int) -> np.ndarray:
    return env._local_pose(player).astype(float).copy()


def ball_snapshot(env: PlannerSkillEnv, player: int) -> tuple[np.ndarray, np.ndarray]:
    position, velocity = env._local_ball(player)
    return position.astype(float).copy(), velocity.astype(float).copy()


def pose_position_error(target: Any, actual: Any) -> float | None:
    if target is None or actual is None:
        return None
    return float(np.linalg.norm(np.asarray(target, dtype=float)[:3] - np.asarray(actual, dtype=float)[:3]))


def pose_orientation_error(target: Any, actual: Any) -> float | None:
    if target is None or actual is None:
        return None
    return float(
        (R.from_quat(np.asarray(target)[3:]) * R.from_quat(np.asarray(actual)[3:]).inv()).magnitude()
    )


def scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def json_value(value: Any) -> str:
    if value is None or (isinstance(value, str) and value == ""):
        return ""
    if isinstance(value, (list, dict, tuple, np.ndarray)):
        return json.dumps(scalar(value), separators=(",", ":"))
    return str(scalar(value))


def first_receiver(env: PlannerSkillEnv) -> int:
    for player in (1, 2):
        if env._is_current_receiver(player):
            return player
    return 1 if float(env._env.data.body("ball").xpos[0]) < 1.5 else 2


def planned_contact_seconds(result: Any) -> float | None:
    if result is None:
        return None
    bounce_time = result.predicted_bounce_time
    contact_after_bounce = result.contact_time_after_bounce
    if bounce_time is None or contact_after_bounce is None:
        return None
    return float(bounce_time + contact_after_bounce)


def estimate_legacy_contact_seconds(
    ball_position: np.ndarray,
    ball_velocity: np.ndarray,
    contact_position: Any,
) -> tuple[float | None, dict[str, Any]]:
    if contact_position is None:
        return None, {}
    bounce = _predict_receiver_bounce(ball_position, ball_velocity, table_z=TABLE_Z, gravity=GRAVITY)
    contact = np.asarray(contact_position, dtype=float)
    if not bounce.get("legal"):
        return None, bounce
    post_velocity = np.asarray(bounce["post_velocity"], dtype=float)
    if abs(post_velocity[0]) <= 1e-8:
        return None, bounce
    after_bounce = (float(contact[0]) - float(bounce["position"][0])) / float(post_velocity[0])
    if after_bounce <= 0:
        return None, bounce
    return float(bounce["time"] + after_bounce), bounce


def candidate_rows(
    *,
    episode_key: str,
    attempt: dict[str, Any],
    selected_result: Any,
    receiver: int,
) -> list[dict[str, Any]]:
    result = attempt["result"]
    diagnostics = list(result.candidate_pool_diagnostics)
    diagnostics.extend(
        item for item in result.explicit_candidate_diagnostics
        if item.get("rejection_reason")
    )
    diagnostics.extend(
        item for item in result.grid_candidate_diagnostics
        if item.get("rejection_reason")
    )
    output = []
    for index, candidate in enumerate(diagnostics):
        contact_time = candidate.get("contact_time_after_bounce")
        bounce_time = result.predicted_bounce_time
        time_until = (
            float(bounce_time) + float(contact_time)
            if bounce_time not in (None, "") and contact_time not in (None, "")
            else None
        )
        position_change = candidate.get("position_change")
        orientation_change = candidate.get(
            "minimum_orientation_change_rad",
            candidate.get("orientation_change_rad"),
        )
        output.append({
            "episode_key": episode_key,
            "planner": "optimized",
            "candidate_source": candidate.get("source_method", "explicit"),
            "planning_step": attempt["step"],
            "attacker_player": receiver,
            "attack_skill": attempt["skill"],
            "candidate_index": index,
            "contact_height": candidate.get("contact_height", ""),
            "contact_position": candidate.get("contact_position", ""),
            "time_until_contact": time_until if time_until is not None else "",
            "time_from_bounce_to_contact": contact_time if contact_time is not None else "",
            "position_change": position_change if position_change is not None else "",
            "orientation_change_rad": orientation_change if orientation_change is not None else "",
            "required_translation_speed": (
                float(position_change) / time_until
                if position_change is not None and time_until and time_until > 0 else ""
            ),
            "required_rotation_speed_rad_s": (
                float(orientation_change) / time_until
                if orientation_change is not None and time_until and time_until > 0 else ""
            ),
            "ballistic_valid": (
                candidate.get("contact_time_after_bounce") not in (None, "")
                and candidate.get("rejection_reason") not in {
                    "wrong_direction_or_flight",
                    "no_valid_flight_time_root",
                }
            ),
            "rejection_reason": candidate.get("rejection_reason", ""),
            "executability_cost": candidate.get("candidate_cost", ""),
            "selected": bool(candidate.get("selected")),
            "predicted_bounce_position": result.predicted_bounce_position_local,
            "predicted_bounce_time": result.predicted_bounce_time,
            "current_racket_pose": attempt["current_pose"],
            "target_racket_pose": (
                candidate.get(
                    "target_pose",
                    selected_result.target_pose_local if selected_result is not None else "",
                )
            ),
        })
    return output


def run_episode(
    *,
    source_row: dict[str, str],
    model: Any,
    planner: str,
    reset_mode: str,
    skill_profile: str,
    gantry_speed_scale: float,
    max_steps: int,
    render: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from v3.env_wrapper import V3PrePositionEnv

    skill1 = str(source_row["player1_skill"])
    skill2 = str(source_row["player2_skill"])
    episode_index = int(source_row["episode_index"])
    seed = int(source_row["seed"])
    episode_key = str(source_row.get("key") or f"{seed}:{skill1}:{skill2}:{episode_index}")
    np.random.seed(episode_rng_seed(seed, skill1, episode_index, skill2))

    env = V3PrePositionEnv(
        proc_id=1,
        history=HISTORY,
        reset_mode=reset_mode,
        skill_profile=skill_profile,
        gantry_speed_scale=gantry_speed_scale,
        planner=planner,
    )
    env.set_skills(skill1, skill2)
    attempts: list[dict[str, Any]] = []
    target_updates: list[dict[str, Any]] = []
    snapshots: dict[int, dict[int, dict[str, Any]]] = {}
    state = {"step": 0}

    original_plan = env._planner_result
    original_set_p1 = env._env.set_target_pose
    original_set_p2 = env._env.set_target_pose_opp

    def record_target(player: int, pose: np.ndarray) -> None:
        ball_position, ball_velocity = ball_snapshot(env, player)
        target_updates.append({
            "step": int(state["step"]),
            "player": player,
            "target_pose": np.asarray(pose, dtype=float).copy(),
            "current_pose": pose_snapshot(env, player),
            "ball_position": ball_position,
            "ball_velocity": ball_velocity,
        })

    def traced_plan(player: int, kwargs: dict[str, Any]):
        ball_position, ball_velocity = ball_snapshot(env, player)
        attempt = {
            "step": int(state["step"]),
            "player": player,
            "skill": skill1 if player == 1 else skill2,
            "ball_position": ball_position,
            "ball_velocity": ball_velocity,
            "current_pose": pose_snapshot(env, player),
            "result": None,
            "target_pose": None,
        }
        result = original_plan(player, kwargs)
        attempt["result"] = result
        attempts.append(attempt)
        return result

    def traced_set_p1(pose):
        record_target(1, pose)
        return original_set_p1(pose)

    def traced_set_p2(pose):
        record_target(2, pose)
        return original_set_p2(pose)

    env._planner_result = traced_plan
    env._env.set_target_pose = traced_set_p1
    env._env.set_target_pose_opp = traced_set_p2

    receiver = 1
    contact_player: int | None = None
    contact_step: int | None = None
    contact_pose: np.ndarray | None = None
    contact_ball: tuple[np.ndarray, np.ndarray] | None = None
    done = False
    termination_reason = "step_limit"
    winner = "truncated"

    try:
        obs, info = env.reset()
        receiver = first_receiver(env)
        for step in range(1, max_steps + 1):
            state["step"] = step
            snapshots[step - 1] = {
                player: {
                    "racket_pose": pose_snapshot(env, player),
                    "ball_position": ball_snapshot(env, player)[0],
                    "ball_velocity": ball_snapshot(env, player)[1],
                }
                for player in (1, 2)
            }
            before_hitter = env._env.rally_state.get("last_hitter")
            action = build_ppo_action(model, obs, info)
            obs, _reward, done, _truncated, info = env.step(action)
            if render:
                env.render()
            snapshots[step] = {
                player: {
                    "racket_pose": pose_snapshot(env, player),
                    "ball_position": ball_snapshot(env, player)[0],
                    "ball_velocity": ball_snapshot(env, player)[1],
                }
                for player in (1, 2)
            }
            after_hitter = env._env.rally_state.get("last_hitter")
            if after_hitter in {"ego", "opp"} and after_hitter != before_hitter:
                contact_player = 1 if after_hitter == "ego" else 2
                contact_step = step
                contact_pose = pose_snapshot(env, contact_player)
                contact_ball = ball_snapshot(env, contact_player)
                break
            if done:
                winner = str(info.get("winner", "unknown"))
                termination_reason = str(info.get("termination_reason", "done"))
                break
        if contact_player is None and done:
            winner = str(info.get("winner", "unknown"))
            termination_reason = str(info.get("termination_reason", "done"))
    finally:
        env.close()

    receiver_attempts = [a for a in attempts if a["player"] == receiver]
    receiver_updates = [u for u in target_updates if u["player"] == receiver]
    first_attempt = receiver_attempts[0] if receiver_attempts else None
    first_update = receiver_updates[0] if receiver_updates else None
    final_update = receiver_updates[-1] if receiver_updates else None
    if planner == "legacy" and first_attempt is None and first_update is not None:
        # Legacy execution calls the original target updater directly, so
        # there is no PlannerSkillEnv planner callback to trace. Reconstruct
        # the legacy diagnostic plan from the first target update only; this
        # does not participate in environment control.
        first_attempt = {
            "step": first_update["step"],
            "player": receiver,
            "skill": skill1 if receiver == 1 else skill2,
            "ball_position": first_update["ball_position"],
            "ball_velocity": first_update["ball_velocity"],
            "current_pose": first_update["current_pose"],
            "result": estimate_legacy(
                first_update["ball_position"],
                first_update["ball_velocity"],
                x_target=float(env._x_target1 if receiver == 1 else env._x_target2),
                y_target=float(
                    env._env.side_target if receiver == 1 else env._env.side_target_opp
                ) * 0.38,
            ),
            "target_pose": first_update["target_pose"],
        }
    result = first_attempt["result"] if first_attempt else None
    selected_attempt = next(
        (
            attempt
            for attempt in receiver_attempts
            if attempt["result"] is not None and attempt["result"].selected
        ),
        first_attempt,
    )
    selected_result = (
        selected_attempt["result"]
        if selected_attempt is not None
        and selected_attempt["result"] is not None
        and selected_attempt["result"].selected
        else None
    )
    if selected_result is not None:
        result = selected_result

    contact_position = result.contact_position_local if result is not None else None
    bounce_position = result.predicted_bounce_position_local if result is not None else None
    bounce_time = result.predicted_bounce_time if result is not None else None
    contact_height = result.sampled_contact_height if result is not None else None
    time_until_contact = planned_contact_seconds(result)
    legacy_bounce = {}
    if planner == "legacy" and first_attempt is not None:
        time_until_contact, legacy_bounce = estimate_legacy_contact_seconds(
            first_attempt["ball_position"], first_attempt["ball_velocity"], contact_position
        )
        if bounce_position is None and legacy_bounce.get("position") is not None:
            bounce_position = np.asarray(legacy_bounce["position"])
        if bounce_time is None:
            bounce_time = legacy_bounce.get("time")
        if contact_height is None and contact_position is not None:
            contact_height = float(np.asarray(contact_position)[2])

    target_pose = first_update["target_pose"] if first_update is not None else None
    final_target_pose = final_update["target_pose"] if final_update is not None else None
    current_pose = first_attempt["current_pose"] if first_attempt is not None else None
    position_change = pose_position_error(target_pose, current_pose)
    orientation_change = pose_orientation_error(target_pose, current_pose)
    actual_at_planned = None
    actual_position_error = None
    actual_orientation_error = None
    min_distance = None
    planned_contact_step = None
    if first_update is not None and time_until_contact is not None:
        planned_contact_step = int(round(first_update["step"] + time_until_contact / DT))
        actual_at_planned = snapshots.get(planned_contact_step, snapshots.get(max(snapshots)))
        if actual_at_planned is not None:
            actual_pose = actual_at_planned[receiver]["racket_pose"]
            actual_position_error = pose_position_error(target_pose, actual_pose)
            actual_orientation_error = pose_orientation_error(target_pose, actual_pose)
            radius = max(2, int(round(0.04 / DT)))
            distances = []
            for step in range(max(0, planned_contact_step - radius), planned_contact_step + radius + 1):
                sample = snapshots.get(step, {}).get(receiver)
                if sample is not None:
                    distances.append(float(np.linalg.norm(sample["racket_pose"][:3] - sample["ball_position"])))
            if distances:
                min_distance = min(distances)

    actual_bounce_position = None
    for bounce in env._env.rally_state.get("bounce_history", []):
        side = "p1" if bounce.get("side") == "ego" else "p2"
        if (receiver == 1 and side == "p1") or (receiver == 2 and side == "p2"):
            actual_bounce_position = np.asarray(
                (bounce.get("x"), bounce.get("y"), bounce.get("z")),
                dtype=float,
            )
            if receiver == 2:
                # bounce_history is world-frame; compare in the same
                # receiver-local frame used by the planner.
                actual_bounce_position = mirror_position_to_world(actual_bounce_position)
            break
    bounce_error = None
    if actual_bounce_position is not None and bounce_position is not None:
        bounce_error = float(np.linalg.norm(np.asarray(actual_bounce_position) - np.asarray(bounce_position)))

    candidate_data = []
    if planner == "optimized" and selected_attempt is not None:
        candidate_data = candidate_rows(
            episode_key=episode_key,
            attempt=selected_attempt,
            selected_result=selected_result,
            receiver=receiver,
        )

    return {
        "episode_key": episode_key,
        "source_seed": seed,
        "source_episode_index": episode_index,
        "planner": planner,
        "player1_skill": skill1,
        "player2_skill": skill2,
        "receiver_player": receiver,
        "first_target_step": first_update["step"] if first_update else "",
        "first_planner_call_step": first_attempt["step"] if first_attempt else "",
        "ball_position_at_planning": first_attempt["ball_position"] if first_attempt else "",
        "ball_velocity_at_planning": first_attempt["ball_velocity"] if first_attempt else "",
        "predicted_bounce_position": bounce_position if bounce_position is not None else "",
        "predicted_bounce_time": bounce_time if bounce_time is not None else "",
        "selected_contact_position": contact_position if contact_position is not None else "",
        "selected_contact_height": contact_height if contact_height is not None else "",
        "time_until_planned_contact": time_until_contact if time_until_contact is not None else "",
        "time_from_bounce_to_contact": (
            result.contact_time_after_bounce if result is not None and result.contact_time_after_bounce is not None
            else (time_until_contact - bounce_time if time_until_contact is not None and bounce_time is not None else "")
        ),
        "racket_pose_at_planning": current_pose if current_pose is not None else "",
        "first_target_racket_pose": target_pose if target_pose is not None else "",
        "final_target_before_contact": final_target_pose if final_target_pose is not None else "",
        "target_update_count_before_contact": len(receiver_updates),
        "translation_distance": position_change if position_change is not None else "",
        "orientation_change_rad": orientation_change if orientation_change is not None else "",
        "required_average_translation_speed": (
            position_change / time_until_contact
            if position_change is not None and time_until_contact and time_until_contact > 0 else ""
        ),
        "required_average_rotation_speed_rad_s": (
            orientation_change / time_until_contact
            if orientation_change is not None and time_until_contact and time_until_contact > 0 else ""
        ),
        "planned_contact_step": planned_contact_step if planned_contact_step is not None else "",
        "actual_racket_pose_at_planned_contact": (
            actual_at_planned[receiver]["racket_pose"] if actual_at_planned is not None else ""
        ),
        "actual_position_error_at_planned_contact": actual_position_error if actual_position_error is not None else "",
        "actual_orientation_error_rad_at_planned_contact": actual_orientation_error if actual_orientation_error is not None else "",
        "minimum_racket_ball_distance": min_distance if min_distance is not None else "",
        "actual_contact_step": contact_step if contact_step is not None else "",
        "actual_contact_player": contact_player if contact_player is not None else "",
        "racket_ball_contact_occurred": contact_player == receiver,
        "actual_contact_racket_pose": contact_pose if contact_pose is not None else "",
        "actual_bounce_position": actual_bounce_position if actual_bounce_position is not None else "",
        "bounce_position_error": bounce_error if bounce_error is not None else "",
        "planner_reason": result.reason if result is not None else "no_plan_recorded",
        "solution_method": result.solution_method if result is not None else "",
        "explicit_success": bool(result is not None and result.solution_method == "explicit"),
        "cached_plan_used": bool(result is not None and result.solution_method == "explicit" and first_attempt is not None),
        "winner": winner,
        "termination_reason": termination_reason,
        "source_successful_returns": source_row.get("successful_returns", ""),
        "source_winner": source_row.get("winner", ""),
        "candidate_count": len(candidate_data),
    }, candidate_data


def classify_pair(pair: dict[str, Any]) -> str:
    legacy = pair.get("legacy", {})
    optimized = pair.get("optimized", {})
    if not optimized.get("racket_ball_contact_occurred"):
        if optimized.get("first_target_step", "") == "":
            return "optimized_planner_no_target"
        if legacy.get("racket_ball_contact_occurred") and optimized.get("first_target_step", 0) > legacy.get("first_target_step", 0):
            return "1_optimized_starts_later"
        speed = optimized.get("required_average_translation_speed", "")
        distance = optimized.get("translation_distance", "")
        available = optimized.get("time_until_planned_contact", "")
        if distance not in ("", None) and available not in ("", None) and float(available) > 0:
            if float(distance) > 1.0 or float(speed) > 3.0:
                return "3_pose_too_far_for_available_time"
        if legacy.get("racket_ball_contact_occurred") and int(optimized.get("target_update_count_before_contact", 0)) <= 1:
            return "4_cached_target_stale"
        if optimized.get("bounce_position_error", "") not in ("", None) and float(optimized["bounce_position_error"]) > 0.1:
            return "6_trajectory_prediction_differs"
        if available not in ("", None) and float(available) > 0:
            return "5_enough_theoretical_time_but_tracking_failed"
        return "2_optimized_contact_timing_or_pose"
    if legacy.get("racket_ball_contact_occurred") and optimized.get("racket_ball_contact_occurred"):
        return "no_optimized_miss_both_contacted"
    return "optimized_contacted_legacy_did_not"


SUMMARY_FIELDS = ["cause", "count", "fraction"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["episode_key"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_value(value) for key, value in row.items()})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose V3 planner misses on saved fixed left-vs-right serves.")
    parser.add_argument("--source-csv", default=DEFAULT_SOURCE)
    parser.add_argument("--skill1", choices=SKILL_NAMES, default="left")
    parser.add_argument("--skill2", choices=SKILL_NAMES, default="right")
    parser.add_argument("--include-successes", action="store_true")
    parser.add_argument("--ppo", default="logs/best_model_tracker1/best_model")
    parser.add_argument("--reset-mode", choices=["clean", "ready", "carryover"], default=None)
    parser.add_argument("--skill-profile", choices=["current", "moderate_aggressive", "aggressive"], default="current")
    parser.add_argument("--gantry-speed-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)

    metadata = load_metadata(args.source_csv)
    metadata_args = metadata.get("args", {}) if isinstance(metadata, dict) else {}
    reset_mode = args.reset_mode or str(metadata_args.get("reset_mode") or "clean")
    rows = load_rows(args.source_csv, args.skill1, args.skill2, not args.include_successes)
    if args.limit is not None:
        rows = rows[:args.limit]

    from stable_baselines3 import PPO

    print(f"Loading PPO: {args.ppo}", flush=True)
    model = PPO.load(args.ppo)
    paired: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for index, source_row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {source_row.get('key', source_row.get('episode_index'))}", flush=True)
        legacy, _ = run_episode(
            source_row=source_row,
            model=model,
            planner="legacy",
            reset_mode=reset_mode,
            skill_profile=args.skill_profile,
            gantry_speed_scale=args.gantry_speed_scale,
            max_steps=args.steps,
            render=args.render,
        )
        optimized, optimized_candidates = run_episode(
            source_row=source_row,
            model=model,
            planner="optimized",
            reset_mode=reset_mode,
            skill_profile=args.skill_profile,
            gantry_speed_scale=args.gantry_speed_scale,
            max_steps=args.steps,
            render=args.render,
        )
        pair = {"episode_key": legacy["episode_key"], "legacy": legacy, "optimized": optimized}
        pair["diagnostic_cause"] = classify_pair(pair)
        flat = {"episode_key": pair["episode_key"], "diagnostic_cause": pair["diagnostic_cause"]}
        for planner, result in (("legacy", legacy), ("optimized", optimized)):
            for key, value in result.items():
                if key in {"episode_key", "planner"}:
                    continue
                flat[f"{planner}_{key}"] = value
        paired.append(flat)
        candidates.extend(optimized_candidates)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / "serve_planner_comparison.csv"
    candidate_path = output_dir / "optimized_candidate_diagnostics.csv"
    summary_path = output_dir / "serve_planner_summary.csv"
    write_csv(episode_path, paired)
    write_csv(candidate_path, candidates)

    counts = Counter(str(row["diagnostic_cause"]) for row in paired)
    summary = [
        {"cause": cause, "count": count, "fraction": count / len(paired) if paired else ""}
        for cause, count in sorted(counts.items())
    ]
    write_csv(summary_path, summary)
    report = {
        "source_csv": args.source_csv,
        "reset_mode": reset_mode,
        "skill_profile": args.skill_profile,
        "episode_count": len(paired),
        "candidate_count": len(candidates),
        "cause_counts": dict(counts),
        "outputs": {
            "episode_csv": str(episode_path),
            "candidate_csv": str(candidate_path),
            "summary_csv": str(summary_path),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, default=scalar))
    print(json.dumps(report, indent=2, default=scalar), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
