"""
Diagnose why fixed-skill episodes hit the evaluator step cap.

This does not change environment termination. It records terminal ball state for
truncated attempts so we can decide whether to add safer done conditions such as
low ball height, stuck ball velocity, out-of-bounds ball position, or no recent
net crossing.

Example:
    MUJOCO_GL=egl python nash_skills/v2/diagnose_truncation.py \
        --attempts 30 --steps 400 --warmup 100 \
        --output-csv skill_eval/truncation_diag.csv \
        --output-json skill_eval/truncation_diag.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from typing import Optional

import numpy as np
from stable_baselines3 import PPO

from nash_skills.env_wrapper import SkillEnv
from nash_skills.skills import SKILL_NAMES
from nash_skills.winner_inference import infer_terminal_winner


PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
HISTORY = 4
TABLE_SHIFT = 1.5
TABLE_HALF_LENGTH = 1.37
TABLE_HALF_WIDTH = 0.75
TABLE_Z = 0.56


def parse_skills(raw: Optional[str]) -> list:
    if raw is None:
        return list(SKILL_NAMES)
    names = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [name for name in names if name not in SKILL_NAMES]
    if unknown:
        raise ValueError(f"Unknown skill(s): {unknown}. Valid skills: {SKILL_NAMES}")
    return names


def build_matchups(skills: list) -> list:
    return [(ego, opp) for ego in skills for opp in skills]


def _build_obs1(obs, info):
    o = np.zeros(9 + 9 + 7 + 7 + 9 * HISTORY, dtype=np.float32)
    o[:9] = obs[:9]
    o[9:18] = obs[18:27]
    o[18:21] = info["diff_pos"]
    o[21:25] = info["diff_quat"]
    o[25:32] = info["target"]
    o[32:] = obs[42: 42 + HISTORY * 9]
    return o


def _build_obs2(obs, info):
    o = np.zeros(9 + 9 + 7 + 7 + 9 * HISTORY, dtype=np.float32)
    o[:9] = obs[9:18]
    o[9:18] = obs[27:36]
    o[18:21] = info["diff_pos_opp"]
    o[21:25] = info["diff_quat_opp"]
    o[25:32] = info["target_opp"]
    o[32:] = obs[42 + HISTORY * 9: 42 + 2 * HISTORY * 9]
    return o


def classify_truncation(ball_pos, ball_vel, steps_since_crossing):
    speed = float(np.linalg.norm(ball_vel))
    x, y, z = map(float, ball_pos)
    reasons = []

    if z < TABLE_Z - 0.05:
        reasons.append("low_ball")
    if speed < 0.05:
        reasons.append("stuck_slow")
    if (
        x < TABLE_SHIFT - TABLE_HALF_LENGTH - 0.5
        or x > TABLE_SHIFT + TABLE_HALF_LENGTH + 0.5
        or abs(y) > TABLE_HALF_WIDTH + 0.75
    ):
        reasons.append("far_out")
    if steps_since_crossing >= 200:
        reasons.append("no_recent_crossing")
    if not reasons:
        reasons.append("long_rally_or_unclear")

    return "+".join(reasons), speed


def run_diagnostic(
    ppo,
    attempts: int,
    max_steps: int,
    warmup_steps: int,
    skills: list,
    output_csv: str,
    output_json: str,
):
    rows = []
    summaries = {}

    for ego_skill, opp_skill in build_matchups(skills):
        print(f"[{ego_skill} vs {opp_skill}] ...", flush=True)
        env = SkillEnv(proc_id=1, history=HISTORY)
        env.set_skills(ego_skill, opp_skill)
        obs, info = env.reset()
        prev_ball_x = float(obs[36])

        for _ in range(warmup_steps):
            action1, _ = ppo.predict(_build_obs1(obs, info), deterministic=True)
            action2, _ = ppo.predict(_build_obs2(obs, info), deterministic=True)
            action = np.zeros(18, dtype=np.float32)
            action[:9] = action1[:9]
            action[9:] = action2[:9]
            with redirect_stdout(io.StringIO()):
                obs, _, done, _, info = env.step(action)
            if done:
                env.set_skills(ego_skill, opp_skill)
                obs, info = env.reset()
            prev_ball_x = float(obs[36])

        counts = Counter()
        speeds = []
        crossing_ages = []

        for attempt_idx in range(attempts):
            steps_in_ep = 0
            crossings = 0
            last_crossing_step = 0
            done = False

            while steps_in_ep < max_steps:
                action1, _ = ppo.predict(_build_obs1(obs, info), deterministic=True)
                action2, _ = ppo.predict(_build_obs2(obs, info), deterministic=True)
                action = np.zeros(18, dtype=np.float32)
                action[:9] = action1[:9]
                action[9:] = action2[:9]

                with redirect_stdout(io.StringIO()):
                    obs, _, done, _, info = env.step(action)

                steps_in_ep += 1
                curr_ball_x = float(obs[36])
                if (prev_ball_x - TABLE_SHIFT) * (curr_ball_x - TABLE_SHIFT) < 0:
                    crossings += 1
                    last_crossing_step = steps_in_ep
                prev_ball_x = curr_ball_x

                if done:
                    winner = infer_terminal_winner(obs, info, fallback="position") or "unknown"
                    rows.append({
                        "ego_skill": ego_skill,
                        "opp_skill": opp_skill,
                        "attempt": attempt_idx,
                        "outcome": "done",
                        "winner": winner,
                        "steps": steps_in_ep,
                        "crossings": crossings,
                        "steps_since_crossing": steps_in_ep - last_crossing_step,
                        "ball_x": float(obs[36]),
                        "ball_y": float(obs[37]),
                        "ball_z": float(obs[38]),
                        "ball_vx": float(obs[39]),
                        "ball_vy": float(obs[40]),
                        "ball_vz": float(obs[41]),
                        "ball_speed": float(np.linalg.norm(obs[39:42])),
                        "truncation_reason": "",
                    })
                    counts[f"done_{winner}"] += 1
                    env.set_skills(ego_skill, opp_skill)
                    obs, info = env.reset()
                    prev_ball_x = float(obs[36])
                    break

            if not done:
                ball_pos = obs[36:39]
                ball_vel = obs[39:42]
                steps_since_crossing = steps_in_ep - last_crossing_step
                reason, speed = classify_truncation(ball_pos, ball_vel, steps_since_crossing)
                rows.append({
                    "ego_skill": ego_skill,
                    "opp_skill": opp_skill,
                    "attempt": attempt_idx,
                    "outcome": "truncated",
                    "winner": "",
                    "steps": steps_in_ep,
                    "crossings": crossings,
                    "steps_since_crossing": steps_since_crossing,
                    "ball_x": float(ball_pos[0]),
                    "ball_y": float(ball_pos[1]),
                    "ball_z": float(ball_pos[2]),
                    "ball_vx": float(ball_vel[0]),
                    "ball_vy": float(ball_vel[1]),
                    "ball_vz": float(ball_vel[2]),
                    "ball_speed": speed,
                    "truncation_reason": reason,
                })
                counts["truncated"] += 1
                counts[f"reason_{reason}"] += 1
                speeds.append(speed)
                crossing_ages.append(steps_since_crossing)
                env.set_skills(ego_skill, opp_skill)
                obs, info = env.reset()
                prev_ball_x = float(obs[36])

        env.close()

        truncs = counts["truncated"]
        summaries[(ego_skill, opp_skill)] = {
            "ego_skill": ego_skill,
            "opp_skill": opp_skill,
            "attempts": attempts,
            "done": attempts - truncs,
            "truncated": truncs,
            "done_fraction": round((attempts - truncs) / attempts, 4) if attempts else None,
            "reason_counts": {
                key.replace("reason_", ""): val
                for key, val in counts.items()
                if key.startswith("reason_")
            },
            "truncated_speed_mean": round(float(np.mean(speeds)), 4) if speeds else None,
            "truncated_steps_since_crossing_mean": (
                round(float(np.mean(crossing_ages)), 2) if crossing_ages else None
            ),
        }
        print(
            f"  done={attempts - truncs} trunc={truncs} "
            f"done_frac={(attempts - truncs) / attempts:.2f} "
            f"reasons={summaries[(ego_skill, opp_skill)]['reason_counts']}",
            flush=True,
        )

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump({"summary": list(summaries.values()), "rows": rows}, f, indent=2)

    print(f"\nCSV saved to:  {output_csv}")
    print(f"JSON saved to: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose fixed-skill truncation causes.")
    parser.add_argument("--attempts", type=int, default=30,
                        help="Episode attempts per skill pair, including done and truncated")
    parser.add_argument("--steps", type=int, default=400,
                        help="Step cap per attempt")
    parser.add_argument("--warmup", type=int, default=100,
                        help="Warmup steps before counted attempts")
    parser.add_argument("--skills", default=None,
                        help="Comma-separated skill subset; default all skills")
    parser.add_argument("--output-csv", default="skill_eval/truncation_diag.csv")
    parser.add_argument("--output-json", default="skill_eval/truncation_diag.json")
    args = parser.parse_args()

    active_skills = parse_skills(args.skills)
    print(f"Loading PPO from {PPO_MODEL_PATH} ...")
    ppo = PPO.load(PPO_MODEL_PATH)
    print(
        f"Running truncation diagnostic over {len(active_skills) ** 2} pair(s), "
        f"attempts={args.attempts}, steps={args.steps} ...\n"
    )
    run_diagnostic(
        ppo=ppo,
        attempts=args.attempts,
        max_steps=args.steps,
        warmup_steps=args.warmup,
        skills=active_skills,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )
