#!/usr/bin/env python3
"""Observational diagnostics for player skill symmetry."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from diagnostic_fixed_skill import build_ppo_obs
from mujoco_env_comp import TABLE_SHIFT
from nash_skills.skills import SKILL_NAMES, get_skill

TARGET_FIELDS = [
    "player",
    "skill_name",
    "side_target",
    "relative_landing_x",
    "relative_landing_y",
    "world_landing_x",
    "world_landing_y",
    "target_pose_x",
    "target_pose_y",
    "target_pose_z",
    "target_pose_world_x",
    "target_pose_world_y",
    "target_pose_world_z",
]

EVENT_FIELDS = [
    "rally_id",
    "step",
    "event_type",
    "player",
    "chosen_skill",
    "target_pose_x",
    "target_pose_y",
    "target_pose_z",
    "target_pose_world_x",
    "target_pose_world_y",
    "target_pose_world_z",
    "gantry_x",
    "gantry_y",
    "racket_x",
    "racket_y",
    "racket_z",
    "target_error_x",
    "target_error_y",
    "target_error_z",
    "target_error_norm",
    "ball_x",
    "ball_y",
    "ball_z",
    "ball_vx",
    "ball_vy",
    "ball_vz",
    "ball_landing_x",
    "ball_landing_y",
    "return_success",
    "winner",
    "truncated",
    "termination_reason",
]

INITIAL_FIELDS = [
    "rally_id",
    "seed",
    "player1_skill",
    "player2_skill",
    "p1_gantry_x",
    "p1_gantry_y",
    "p2_gantry_x",
    "p2_gantry_y",
    "p1_racket_x",
    "p1_racket_y",
    "p1_racket_z",
    "p2_racket_x",
    "p2_racket_y",
    "p2_racket_z",
    "p2_racket_mirror_x",
    "p2_racket_mirror_y",
    "p2_racket_mirror_z",
    "racket_mirror_error_x",
    "racket_mirror_error_y",
    "racket_mirror_error_z",
    "racket_mirror_error_norm",
    "ball_x",
    "ball_y",
    "ball_z",
    "ball_vx",
    "ball_vy",
    "ball_vz",
    "ball_mirror_side",
]

RALLY_FIELDS = [
    "rally_id",
    "seed",
    "player1_skill",
    "player2_skill",
    "winner",
    "truncated",
    "rally_length",
    "num_contacts",
    "num_successes",
    "termination_reason",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def player_world_landing(player: int, relative_x: float, relative_y: float) -> tuple[float, float]:
    if player == 1:
        return relative_x, relative_y
    return 2 * TABLE_SHIFT - relative_x, -relative_y


def player_world_pose(player: int, pose: np.ndarray) -> tuple[float, float, float]:
    if player == 1:
        return float(pose[0]), float(pose[1]), float(pose[2])
    return float(2 * TABLE_SHIFT - pose[0]), float(-pose[1]), float(pose[2])


def static_target_rows() -> list[dict[str, Any]]:
    rows = []
    for player in (1, 2):
        for skill in SKILL_NAMES:
            side, x_target = get_skill(skill)
            rel_y = float(side * 0.38)
            world_x, world_y = player_world_landing(player, float(x_target), rel_y)
            rows.append({
                "player": player,
                "skill_name": skill,
                "side_target": side,
                "relative_landing_x": float(x_target),
                "relative_landing_y": rel_y,
                "world_landing_x": world_x,
                "world_landing_y": world_y,
                "target_pose_x": "",
                "target_pose_y": "",
                "target_pose_z": "",
                "target_pose_world_x": "",
                "target_pose_world_y": "",
                "target_pose_world_z": "",
            })
    return rows


def parse_contact_lines(lines: list[str]) -> list[dict[str, Any]]:
    contacts = []
    for line in lines:
        if "Returned successfully by ego" in line:
            player = 1
        elif "Returned successfully by opp" in line:
            player = 2
        else:
            continue
        parts = line.split()
        try:
            x_land = float(parts[-2])
            y_land = float(parts[-1])
        except (ValueError, IndexError):
            x_land = ""
            y_land = ""
        contacts.append({
            "player": player,
            "ball_landing_x": x_land,
            "ball_landing_y": y_land,
            "return_success": True,
        })
    return contacts


def capture_step(env, action):
    buffer = StringIO()
    with redirect_stdout(buffer):
        result = env.step(action)
    return result, buffer.getvalue().splitlines()


def infer_winner(obs, info) -> str:
    from nash_skills.winner_inference import infer_terminal_winner

    inferred = infer_terminal_winner(obs, info, fallback=None)
    if inferred == "ego":
        return "player1"
    if inferred == "opp":
        return "player2"
    return "truncated"


def pose_event(env, info: dict[str, Any], rally_id: str, step: int, player: int, skill: str) -> dict[str, Any]:
    pose = np.asarray(info["target"] if player == 1 else info["target_opp"], dtype=float)
    world_x, world_y, world_z = player_world_pose(player, pose)
    if player == 1:
        gantry = np.asarray(env.data.qpos[0:2], dtype=float)
        racket = np.asarray(env.data.body("tennis_racket").xpos, dtype=float)
    else:
        gantry = np.asarray(env.data.qpos[9:11], dtype=float)
        racket = np.asarray(env.data.body("tennis_racket_opp").xpos, dtype=float)
    ball = np.asarray(env.data.body("ball").xpos, dtype=float)
    ball_vel = np.asarray(env.data.qvel[-6:-3], dtype=float)
    target_world = np.array([world_x, world_y, world_z], dtype=float)
    err = target_world - racket
    return {
        "rally_id": rally_id,
        "step": step,
        "event_type": "target",
        "player": player,
        "chosen_skill": skill,
        "target_pose_x": float(pose[0]),
        "target_pose_y": float(pose[1]),
        "target_pose_z": float(pose[2]),
        "target_pose_world_x": world_x,
        "target_pose_world_y": world_y,
        "target_pose_world_z": world_z,
        "gantry_x": float(gantry[0]),
        "gantry_y": float(gantry[1]),
        "racket_x": float(racket[0]),
        "racket_y": float(racket[1]),
        "racket_z": float(racket[2]),
        "target_error_x": float(err[0]),
        "target_error_y": float(err[1]),
        "target_error_z": float(err[2]),
        "target_error_norm": float(np.linalg.norm(err)),
        "ball_x": float(ball[0]),
        "ball_y": float(ball[1]),
        "ball_z": float(ball[2]),
        "ball_vx": float(ball_vel[0]),
        "ball_vy": float(ball_vel[1]),
        "ball_vz": float(ball_vel[2]),
        "ball_landing_x": "",
        "ball_landing_y": "",
        "return_success": "",
        "winner": "",
        "truncated": "",
        "termination_reason": "",
    }


def contact_event(env, rally_id: str, step: int, contact: dict[str, Any], p1_skill: str, p2_skill: str) -> dict[str, Any]:
    player = int(contact["player"])
    skill = p1_skill if player == 1 else p2_skill
    base = pose_event(env, {"target": env.curr_target, "target_opp": env.curr_target_opp}, rally_id, step, player, skill)
    base.update({
        "event_type": "contact",
        "ball_landing_x": contact["ball_landing_x"],
        "ball_landing_y": contact["ball_landing_y"],
        "return_success": contact["return_success"],
    })
    return base


def terminal_event(env, info: dict[str, Any], rally_id: str, step: int, p1_skill: str, p2_skill: str, winner: str, reason: str) -> dict[str, Any]:
    player = 1 if winner == "player2" else 2 if winner == "player1" else 0
    if player == 0:
        base = pose_event(env, info, rally_id, step, 1, p1_skill)
        base["player"] = ""
        base["chosen_skill"] = ""
    else:
        base = pose_event(env, info, rally_id, step, player, p1_skill if player == 1 else p2_skill)
    base.update({
        "event_type": "terminal",
        "return_success": False,
        "winner": winner,
        "truncated": winner == "truncated",
        "termination_reason": reason,
    })
    return base


def initial_state_row(env, info: dict[str, Any], rally_id: str, seed: int, p1_skill: str, p2_skill: str) -> dict[str, Any]:
    initial = info.get("initial_state", {})
    p1_racket = np.asarray(initial.get("p1_racket", env.data.body("tennis_racket").xpos), dtype=float)
    p2_racket = np.asarray(initial.get("p2_racket", env.data.body("tennis_racket_opp").xpos), dtype=float)
    p2_mirror = np.asarray(
        initial.get(
            "p2_racket_mirrored",
            np.array([2 * TABLE_SHIFT - p2_racket[0], -p2_racket[1], p2_racket[2]], dtype=float),
        ),
        dtype=float,
    )
    mirror_err = p1_racket - p2_mirror
    ball = np.asarray(env.data.body("ball").xpos, dtype=float)
    ball_vel = np.asarray(env.data.qvel[-6:-3], dtype=float)
    if ball[0] < TABLE_SHIFT and ball_vel[0] > 0:
        ball_side = "toward_opp_from_p1_side"
    elif ball[0] > TABLE_SHIFT and ball_vel[0] < 0:
        ball_side = "toward_p1_from_opp_side"
    else:
        ball_side = "other"
    return {
        "rally_id": rally_id,
        "seed": seed,
        "player1_skill": p1_skill,
        "player2_skill": p2_skill,
        "p1_gantry_x": float(initial.get("p1_gantry", env.data.qpos[0:2])[0]),
        "p1_gantry_y": float(initial.get("p1_gantry", env.data.qpos[0:2])[1]),
        "p2_gantry_x": float(initial.get("p2_gantry", env.data.qpos[9:11])[0]),
        "p2_gantry_y": float(initial.get("p2_gantry", env.data.qpos[9:11])[1]),
        "p1_racket_x": float(p1_racket[0]),
        "p1_racket_y": float(p1_racket[1]),
        "p1_racket_z": float(p1_racket[2]),
        "p2_racket_x": float(p2_racket[0]),
        "p2_racket_y": float(p2_racket[1]),
        "p2_racket_z": float(p2_racket[2]),
        "p2_racket_mirror_x": float(p2_mirror[0]),
        "p2_racket_mirror_y": float(p2_mirror[1]),
        "p2_racket_mirror_z": float(p2_mirror[2]),
        "racket_mirror_error_x": float(mirror_err[0]),
        "racket_mirror_error_y": float(mirror_err[1]),
        "racket_mirror_error_z": float(mirror_err[2]),
        "racket_mirror_error_norm": float(np.linalg.norm(mirror_err)),
        "ball_x": float(ball[0]),
        "ball_y": float(ball[1]),
        "ball_z": float(ball[2]),
        "ball_vx": float(ball_vel[0]),
        "ball_vy": float(ball_vel[1]),
        "ball_vz": float(ball_vel[2]),
        "ball_mirror_side": ball_side,
    }


def run_fixed_pair(env, model, p1_skill: str, p2_skill: str, seed: int, episodes: int, max_steps: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    target_rows = []
    initial_rows = []
    event_rows = []
    rally_rows = []
    for episode in range(episodes):
        np.random.seed(seed * 10_000 + episode)
        rally_id = f"{seed}:{p1_skill}_vs_{p2_skill}:{episode}"
        env.set_skills(p1_skill, p2_skill)
        obs, info = env.reset()
        initial_rows.append(initial_state_row(env, info, rally_id, seed, p1_skill, p2_skill))
        prev_ball_x = float(obs[36])
        event_rows.append(pose_event(env, info, rally_id, 0, 1, p1_skill))
        event_rows.append(pose_event(env, info, rally_id, 0, 2, p2_skill))
        contacts_count = 0
        successes = 0
        winner = "truncated"
        termination_reason = "step_limit"
        done = False
        step = 0

        for step in range(1, max_steps + 1):
            ppo1 = build_ppo_obs(obs, info, player=1)
            ppo2 = build_ppo_obs(obs, info, player=2)
            a1, _ = model.predict(ppo1, deterministic=True)
            a2, _ = model.predict(ppo2, deterministic=True)
            action = np.zeros(18, dtype=np.float32)
            action[:9] = a1[:9]
            action[9:] = a2[:9]

            (obs, _reward, done, _truncated, info), lines = capture_step(env, action)
            for contact in parse_contact_lines(lines):
                contacts_count += 1
                successes += int(bool(contact["return_success"]))
                event_rows.append(contact_event(env, rally_id, step, contact, p1_skill, p2_skill))

            curr_ball_x = float(obs[36])
            if (prev_ball_x - TABLE_SHIFT) * (curr_ball_x - TABLE_SHIFT) < 0:
                event_rows.append(pose_event(env, info, rally_id, step, 1, p1_skill))
                event_rows.append(pose_event(env, info, rally_id, step, 2, p2_skill))
            prev_ball_x = curr_ball_x

            if done:
                winner = infer_winner(obs, info)
                termination_reason = info.get("termination_reason", "env_done")
                event_rows.append(terminal_event(env, info, rally_id, step, p1_skill, p2_skill, winner, termination_reason))
                break

        truncated = not done or winner == "truncated"
        for row in event_rows:
            if row["rally_id"] == rally_id:
                row["winner"] = winner
                row["truncated"] = truncated
                row["termination_reason"] = termination_reason
        rally_rows.append({
            "rally_id": rally_id,
            "seed": seed,
            "player1_skill": p1_skill,
            "player2_skill": p2_skill,
            "winner": winner,
            "truncated": truncated,
            "rally_length": step,
            "num_contacts": contacts_count,
            "num_successes": successes,
            "termination_reason": termination_reason,
        })
    return target_rows, initial_rows, event_rows, rally_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect 5-skill symmetry without changing environment behavior.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="data/diagnostics_skill_randomization/symmetry_probe")
    parser.add_argument("--ppo", default="logs/best_model_tracker1/best_model")
    parser.add_argument("--reset-mode", choices=["clean", "ready", "carryover"], default="ready")
    parser.add_argument("--gantry-speed-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from stable_baselines3 import PPO
    from nash_skills.env_wrapper import SkillEnv

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    target_rows = static_target_rows()
    print("Static skill landing targets:")
    for row in target_rows:
        print(
            f"P{row['player']} {row['skill_name']:12s} "
            f"relative=({row['relative_landing_x']:.3f}, {row['relative_landing_y']:.3f}) "
            f"world=({row['world_landing_x']:.3f}, {row['world_landing_y']:.3f})"
        )

    model = PPO.load(args.ppo)
    env = SkillEnv(proc_id=1, history=4, reset_mode=args.reset_mode, gantry_speed_scale=args.gantry_speed_scale)
    all_events = []
    all_rallies = []
    all_initial = []
    try:
        for p1_skill, p2_skill in [("right_short", "right_short"), ("left_short", "left_short")]:
            _targets, initial, events, rallies = run_fixed_pair(
                env, model, p1_skill, p2_skill, args.seed, args.episodes, args.steps
            )
            all_initial.extend(initial)
            all_events.extend(events)
            all_rallies.extend(rallies)
    finally:
        env.close()

    write_csv(out / "skill_target_positions.csv", target_rows, TARGET_FIELDS)
    write_csv(out / "initial_states.csv", all_initial, INITIAL_FIELDS)
    write_csv(out / "rally_events.csv", all_events, EVENT_FIELDS)
    write_csv(out / "rallies.csv", all_rallies, RALLY_FIELDS)
    (out / "metadata.json").write_text(json.dumps({
        "args": vars(args),
        "skill_pairs": [["right_short", "right_short"], ["left_short", "left_short"]],
        "note": "target_opp is stored in player-relative mirrored coordinates; target_pose_world_* mirrors it back to world coordinates.",
    }, indent=2))
    print(f"Wrote symmetry diagnostics to {out}")


if __name__ == "__main__":
    main()
