#!/usr/bin/env python3
"""Resumable fixed-player skill diagnostic for the 5-skill setup."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from nash_skills.skills import SKILL_NAMES, N_SKILLS, SKILL_PROFILE_NAMES, get_skill, skill_index, world_target_xy

HISTORY = 4
EPISODE_FIELDS = [
    "key",
    "seed",
    "episode_index",
    "skill_profile",
    "player1_skill",
    "player2_skill",
    "winner",
    "termination_reason",
    "reached_step_limit",
    "physics_steps",
    "decision_state_count",
    "net_crossings",
    "successful_returns",
    "return_bucket",
    "raw_obs_ids",
    "state_ids",
    "player1_target_xy",
    "player2_target_xy",
    "validation_ok",
    "validation_errors",
]
SUMMARY_FIELDS = [
    "player1_skill",
    "player2_skill",
    "episode_count",
    "player1_wins",
    "player2_wins",
    "step_limit_count",
    "step_limit_rate",
    "step_limit_ci_low",
    "step_limit_ci_high",
    "player1_win_rate_all",
    "player1_win_ci_low",
    "player1_win_ci_high",
    "player2_win_rate_all",
    "player2_win_ci_low",
    "player2_win_ci_high",
    "completed_count",
    "completed_player1_win_rate",
    "completed_player1_win_ci_low",
    "completed_player1_win_ci_high",
    "physics_steps_mean",
    "physics_steps_median",
    "physics_steps_std",
    "physics_steps_max",
    "decision_states_mean",
    "decision_states_median",
    "decision_states_max",
    "net_crossings_mean",
    "successful_returns_mean",
    "zero_return_rate",
    "one_return_rate",
    "two_plus_return_rate",
]
CONTACT_FIELDS = [
    "key",
    "seed",
    "episode_index",
    "player1_skill",
    "player2_skill",
    "step",
    "player",
    "returning_skill",
    "x_land",
    "y_land",
    "expected_x",
    "expected_y",
    "error_dx",
    "error_dy",
    "error_dist",
    "winner",
    "truncated",
]


def norm_id(skill: str) -> float:
    return skill_index(skill) / (N_SKILLS - 1)


def target_xy(skill: str, skill_profile: str = "current") -> list[float]:
    side, x_target = get_skill(skill, profile=skill_profile)
    return [float(x_target), float(side * 0.38)]


def episode_key(seed: int, player1_skill: str, episode_index: int, player2_skill: str | None = None) -> str:
    if player2_skill is None:
        return f"{seed}:{player1_skill}:{episode_index}"
    return f"{seed}:{player1_skill}:{player2_skill}:{episode_index}"


def episode_rng_seed(seed: int, player1_skill: str, episode_index: int, player2_skill: str | None = None) -> int:
    player2_term = 0 if player2_skill is None else skill_index(player2_skill) * 101
    return (
        (seed + 1_000_003) * 1_000_003
        + skill_index(player1_skill) * 10_007
        + player2_term
        + episode_index
    ) % (2**32 - 1)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def build_ppo_obs(obs, info, player: int) -> np.ndarray:
    ppo_obs = np.zeros(9 + 9 + 7 + 7 + 9 * HISTORY, dtype=np.float32)
    if player == 1:
        ppo_obs[:9] = obs[:9]
        ppo_obs[9:18] = obs[18:27]
        ppo_obs[18:21] = info["diff_pos"]
        ppo_obs[21:25] = info["diff_quat"]
        ppo_obs[25:32] = info["target"]
        ppo_obs[32:] = obs[42: 42 + HISTORY * 9]
    else:
        ppo_obs[:9] = obs[9:18]
        ppo_obs[9:18] = obs[27:36]
        ppo_obs[18:21] = info["diff_pos_opp"]
        ppo_obs[21:25] = info["diff_quat_opp"]
        ppo_obs[25:32] = info["target_opp"]
        ppo_obs[32:] = obs[42 + HISTORY * 9: 42 + 2 * HISTORY * 9]
    return ppo_obs


def parse_json_cell(value: str, default=None):
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def load_episode_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dedupe_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["key"])
        if key in seen:
            raise ValueError(f"Duplicate episode key: {key}")
        seen[key] = row
    return [seen[k] for k in sorted(seen)]


def existing_keys(rows: Iterable[dict[str, Any]]) -> set[str]:
    keys = set()
    for row in rows:
        key = row.get("key")
        if key in keys:
            raise ValueError(f"Duplicate episode key in existing rows: {key}")
        keys.add(key)
    return keys


def validate_resume_rows(
    rows: Iterable[dict[str, Any]],
    fixed_skill: str,
    seeds: list[int],
    episodes_per_matchup: int,
    fixed_player: int = 2,
) -> None:
    allowed_seeds = set(seeds)
    for row in rows:
        key = row.get("key")
        fixed_field = f"player{fixed_player}_skill"
        if row.get(fixed_field) != fixed_skill:
            raise ValueError(
                f"Resume row {key} has {fixed_field}={row.get(fixed_field)}; "
                f"expected {fixed_skill}"
            )
        if row.get("player1_skill") not in SKILL_NAMES:
            raise ValueError(f"Resume row {key} has unknown player1_skill={row.get('player1_skill')}")
        if row.get("player2_skill") not in SKILL_NAMES:
            raise ValueError(f"Resume row {key} has unknown player2_skill={row.get('player2_skill')}")
        seed = int(row["seed"])
        episode_index = int(row["episode_index"])
        if seed not in allowed_seeds:
            raise ValueError(f"Resume row {key} has seed={seed}, outside requested seeds={seeds}")
        if episode_index < 0 or episode_index >= episodes_per_matchup:
            raise ValueError(
                f"Resume row {key} has episode_index={episode_index}, "
                f"outside requested range [0, {episodes_per_matchup})"
            )


def planned_keys(seeds: list[int], episodes_per_matchup: int) -> list[str]:
    return [
        episode_key(seed, skill, episode_index)
        for seed in seeds
        for skill in SKILL_NAMES
        for episode_index in range(episodes_per_matchup)
    ]


def rows_for_aggregation(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        item = dict(row)
        item["seed"] = int(item["seed"])
        item["episode_index"] = int(item["episode_index"])
        item["reached_step_limit"] = str(item["reached_step_limit"]).lower() in {"true", "1", "yes"}
        item["physics_steps"] = int(item["physics_steps"])
        item["decision_state_count"] = int(item["decision_state_count"])
        item["net_crossings"] = int(item.get("net_crossings") or item["decision_state_count"])
        item["successful_returns"] = int(item.get("successful_returns") or 0)
        item["validation_ok"] = str(item["validation_ok"]).lower() in {"true", "1", "yes"}
        normalized.append(item)
    return normalized


def aggregate_rows(rows: list[dict[str, Any]], fixed_skill: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {skill: [] for skill in SKILL_NAMES}
    for row in rows_for_aggregation(rows):
        groups.setdefault(row["player1_skill"], []).append(row)

    summary = []
    for skill in SKILL_NAMES:
        rs = groups.get(skill, [])
        n = len(rs)
        p1 = sum(r["winner"] == "player1" for r in rs)
        p2 = sum(r["winner"] == "player2" for r in rs)
        step_limit = sum(r["reached_step_limit"] for r in rs)
        completed = p1 + p2
        steps = [r["physics_steps"] for r in rs]
        decision_counts = [r["decision_state_count"] for r in rs]
        crossing_counts = [r["net_crossings"] for r in rs]
        return_counts = [r["successful_returns"] for r in rs]
        p1_ci = wilson_interval(p1, n)
        p2_ci = wilson_interval(p2, n)
        step_ci = wilson_interval(step_limit, n)
        completed_ci = wilson_interval(p1, completed)
        summary.append({
            "player1_skill": skill,
            "player2_skill": fixed_skill,
            "episode_count": n,
            "player1_wins": p1,
            "player2_wins": p2,
            "step_limit_count": step_limit,
            "step_limit_rate": step_limit / n if n else None,
            "step_limit_ci_low": step_ci[0],
            "step_limit_ci_high": step_ci[1],
            "player1_win_rate_all": p1 / n if n else None,
            "player1_win_ci_low": p1_ci[0],
            "player1_win_ci_high": p1_ci[1],
            "player2_win_rate_all": p2 / n if n else None,
            "player2_win_ci_low": p2_ci[0],
            "player2_win_ci_high": p2_ci[1],
            "completed_count": completed,
            "completed_player1_win_rate": p1 / completed if completed else None,
            "completed_player1_win_ci_low": completed_ci[0],
            "completed_player1_win_ci_high": completed_ci[1],
            "physics_steps_mean": statistics.mean(steps) if steps else None,
            "physics_steps_median": statistics.median(steps) if steps else None,
            "physics_steps_std": statistics.pstdev(steps) if len(steps) > 1 else 0.0,
            "physics_steps_max": max(steps) if steps else None,
            "decision_states_mean": statistics.mean(decision_counts) if decision_counts else None,
            "decision_states_median": statistics.median(decision_counts) if decision_counts else None,
            "decision_states_max": max(decision_counts) if decision_counts else None,
            "net_crossings_mean": statistics.mean(crossing_counts) if crossing_counts else None,
            "successful_returns_mean": statistics.mean(return_counts) if return_counts else None,
            "zero_return_rate": sum(v == 0 for v in return_counts) / n if n else None,
            "one_return_rate": sum(v == 1 for v in return_counts) / n if n else None,
            "two_plus_return_rate": sum(v >= 2 for v in return_counts) / n if n else None,
        })
    return summary


def validate_episode_state(env, player1_skill: str, player2_skill: str, raw_obs, state) -> list[str]:
    errors = []
    expected = [norm_id(player1_skill), norm_id(player2_skill)]
    if raw_obs is not None and not np.allclose(np.asarray(raw_obs)[-2:], expected, atol=1e-6):
        errors.append(f"raw ids {np.asarray(raw_obs)[-2:].tolist()} != {expected}")
    if state is not None and not np.allclose(np.asarray(state)[-2:], expected, atol=1e-6):
        errors.append(f"state ids {np.asarray(state)[-2:].tolist()} != {expected}")
    skill_profile = getattr(env, "skill_profile", "current")
    side1, x1 = get_skill(player1_skill, profile=skill_profile)
    side2, x2 = get_skill(player2_skill, profile=skill_profile)
    if float(env.side_target) != float(side1):
        errors.append(f"player1 side target {env.side_target} != {side1}")
    if float(env.side_target_opp) != float(side2):
        errors.append(f"player2 side target {env.side_target_opp} != {side2}")
    if float(env._x_target1) != float(x1):
        errors.append(f"player1 x target {env._x_target1} != {x1}")
    if float(env._x_target2) != float(x2):
        errors.append(f"player2 x target {env._x_target2} != {x2}")
    return errors


def capture_step(env, action):
    buffer = StringIO()
    with redirect_stdout(buffer):
        result = env.step(action)
    return result, buffer.getvalue().splitlines()


def parse_contacts(lines: list[str], step: int) -> list[dict[str, Any]]:
    contacts = []
    for line in lines:
        if "Returned successfully by ego" in line:
            player = "player1"
        elif "Returned successfully by opp" in line:
            player = "player2"
        else:
            continue
        parts = line.split()
        try:
            x_land = float(parts[-2])
            y_land = float(parts[-1])
        except (ValueError, IndexError):
            x_land = None
            y_land = None
        contacts.append({"step": step, "player": player, "x_land": x_land, "y_land": y_land})
    return contacts


def enrich_contacts(
    contacts: list[dict[str, Any]],
    row: dict[str, Any],
    player1_skill: str,
    player2_skill: str,
    skill_profile: str,
) -> list[dict[str, Any]]:
    enriched = []
    for contact in contacts:
        player_num = 1 if contact["player"] == "player1" else 2
        returning_skill = player1_skill if player_num == 1 else player2_skill
        expected_x, expected_y = world_target_xy(player_num, returning_skill, profile=skill_profile)
        x_land = contact.get("x_land")
        y_land = contact.get("y_land")
        if x_land is None or y_land is None:
            dx = dy = dist = ""
        else:
            dx = float(x_land) - expected_x
            dy = float(y_land) - expected_y
            dist = math.hypot(dx, dy)
        enriched.append({
            "key": row["key"],
            "seed": row["seed"],
            "episode_index": row["episode_index"],
            "player1_skill": player1_skill,
            "player2_skill": player2_skill,
            "step": contact["step"],
            "player": contact["player"],
            "returning_skill": returning_skill,
            "x_land": x_land,
            "y_land": y_land,
            "expected_x": expected_x,
            "expected_y": expected_y,
            "error_dx": dx,
            "error_dy": dy,
            "error_dist": dist,
            "winner": row["winner"],
            "truncated": row["winner"] == "truncated" or row["reached_step_limit"],
        })
    return enriched


def run_episode(env, model, seed: int, episode_index: int, player1_skill: str, player2_skill: str, steps: int):
    from nash_skills.v2.state_encoder import encode_ego
    from nash_skills.winner_inference import infer_terminal_winner

    env.set_skills(player1_skill, player2_skill)
    obs, info = env.reset()
    prev_ball_x = float(obs[36])
    decision_states = []
    trajectory_samples = []
    contacts = []
    validation_errors = validate_episode_state(env, player1_skill, player2_skill, obs, encode_ego(obs, info))
    last_raw = obs.copy()
    last_state = encode_ego(obs, info)
    termination_reason = "step_limit"
    winner = "truncated"
    done = False

    for step in range(1, steps + 1):
        ppo1 = build_ppo_obs(obs, info, player=1)
        ppo2 = build_ppo_obs(obs, info, player=2)
        a1, _ = model.predict(ppo1, deterministic=True)
        a2, _ = model.predict(ppo2, deterministic=True)
        action = np.zeros(18, dtype=np.float32)
        action[:9] = a1[:9]
        action[9:] = a2[:9]

        (obs, _reward, done, _truncated, info), lines = capture_step(env, action)
        contacts.extend(parse_contacts(lines, step))
        curr_ball_x = float(obs[36])

        if step % 50 == 0:
            trajectory_samples.append({
                "step": step,
                "ball_pos": obs[36:39].astype(float).tolist(),
                "ball_vel": obs[39:42].astype(float).tolist(),
                "player1_gantry": obs[0:2].astype(float).tolist(),
                "player2_gantry": obs[18:20].astype(float).tolist(),
            })

        if (prev_ball_x - 1.5) * (curr_ball_x - 1.5) < 0:
            raw = obs.copy()
            state = encode_ego(obs, info)
            last_raw = raw
            last_state = state
            decision_states.append({
                "step": step,
                "ball_pos": raw[36:39].astype(float).tolist(),
                "ball_vel": raw[39:42].astype(float).tolist(),
                "player1_gantry": raw[0:2].astype(float).tolist(),
                "player2_gantry": raw[18:20].astype(float).tolist(),
            })
            validation_errors.extend(validate_episode_state(env, player1_skill, player2_skill, raw, state))
        prev_ball_x = curr_ball_x

        if done:
            inferred = infer_terminal_winner(obs, info, fallback=None)
            winner = "player1" if inferred == "ego" else "player2" if inferred == "opp" else "truncated"
            termination_reason = info.get("termination_reason", "env_done")
            break

    validation_errors.extend(validate_episode_state(env, player1_skill, player2_skill, last_raw, last_state))
    key = episode_key(seed, player1_skill, episode_index, player2_skill)
    row = {
        "key": key,
        "seed": seed,
        "episode_index": episode_index,
        "skill_profile": getattr(env, "skill_profile", "current"),
        "player1_skill": player1_skill,
        "player2_skill": player2_skill,
        "winner": winner,
        "termination_reason": termination_reason,
        "reached_step_limit": not done,
        "physics_steps": step,
        "decision_state_count": len(decision_states),
        "net_crossings": len(decision_states),
        "successful_returns": len(contacts),
        "return_bucket": "0" if len(contacts) == 0 else "1" if len(contacts) == 1 else "2+",
        "raw_obs_ids": json.dumps(np.asarray(last_raw)[-2:].astype(float).tolist()),
        "state_ids": json.dumps(np.asarray(last_state)[-2:].astype(float).tolist()),
        "player1_target_xy": json.dumps(target_xy(player1_skill, getattr(env, "skill_profile", "current"))),
        "player2_target_xy": json.dumps(target_xy(player2_skill, getattr(env, "skill_profile", "current"))),
        "validation_ok": len(validation_errors) == 0,
        "validation_errors": json.dumps(validation_errors),
    }
    detail = {
        "key": key,
        "row": row,
        "decision_states": decision_states,
        "trajectory_samples": trajectory_samples,
        "contacts": enrich_contacts(
            contacts,
            row,
            player1_skill,
            player2_skill,
            getattr(env, "skill_profile", "current"),
        ),
    }
    return row, detail


def select_representatives(details: list[dict[str, Any]], per_skill: int) -> list[dict[str, Any]]:
    selected = []
    for skill in SKILL_NAMES:
        skill_details = [d for d in details if d["row"]["player1_skill"] == skill]
        completed = [d for d in skill_details if d["row"]["winner"] != "truncated"]
        truncated = [d for d in skill_details if d["row"]["winner"] == "truncated"]
        selected.extend(completed[:per_skill])
        selected.extend(sorted(truncated, key=lambda d: int(d["row"]["physics_steps"]), reverse=True)[:per_skill])
    seen = set()
    unique = []
    for detail in selected:
        if detail["key"] in seen:
            continue
        seen.add(detail["key"])
        unique.append(detail)
    return unique


def plot_outputs(
    out: Path,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    representatives: list[dict[str, Any]],
    steps: int,
    skill_profile: str = "current",
) -> list[str]:
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    labels = [row["player1_skill"] for row in summary]
    x = np.arange(len(summary))

    def save(name: str):
        path = plot_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path))

    p1 = np.array([int(row["player1_wins"]) for row in summary])
    p2 = np.array([int(row["player2_wins"]) for row in summary])
    trunc = np.array([int(row["step_limit_count"]) for row in summary])
    plt.figure(figsize=(8, 5))
    plt.bar(x, p1, label="player1")
    plt.bar(x, p2, bottom=p1, label="player2")
    plt.bar(x, trunc, bottom=p1 + p2, label="step limit")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("episodes")
    plt.title("Outcome distribution by player 1 skill")
    plt.legend()
    save("outcome_distribution.png")

    rates = [row["step_limit_rate"] or 0 for row in summary]
    lows = [row["step_limit_ci_low"] or 0 for row in summary]
    highs = [row["step_limit_ci_high"] or 0 for row in summary]
    yerr = [
        [max(0.0, rate - low) for rate, low in zip(rates, lows)],
        [max(0.0, high - rate) for rate, high in zip(rates, highs)],
    ]
    plt.figure(figsize=(8, 5))
    plt.bar(x, rates, yerr=yerr, capsize=3)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("step-limit rate")
    plt.title("Step-limit rate by player 1 skill")
    save("truncation_rate.png")

    normalized = rows_for_aggregation(rows)
    data_with_labels = [
        ([row["physics_steps"] for row in normalized if row["player1_skill"] == skill], skill)
        for skill in labels
    ]
    data = [values for values, _skill in data_with_labels if values]
    duration_labels = [skill for values, skill in data_with_labels if values]
    plt.figure(figsize=(8, 5))
    if data:
        try:
            plt.boxplot(data, tick_labels=duration_labels, showmeans=True)
        except TypeError:
            plt.boxplot(data, labels=duration_labels, showmeans=True)
    plt.axhline(steps, color="red", linestyle="--", label=f"{steps}-step cap")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("physics steps")
    plt.title("Rally-duration distribution")
    plt.legend()
    save("rally_duration_distribution.png")

    comp_rates = [0.0 if row["completed_player1_win_rate"] is None else row["completed_player1_win_rate"] for row in summary]
    comp_lows = [row["completed_player1_win_ci_low"] for row in summary]
    comp_highs = [row["completed_player1_win_ci_high"] for row in summary]
    comp_yerr = [
        [0.0 if low is None else max(0.0, rate - low) for rate, low in zip(comp_rates, comp_lows)],
        [0.0 if high is None else max(0.0, high - rate) for rate, high in zip(comp_rates, comp_highs)],
    ]
    plt.figure(figsize=(8, 5))
    plt.bar(x, comp_rates, yerr=comp_yerr, capsize=3)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("P1 win rate among completed")
    plt.title("Completed-rally player 1 win rate")
    save("completed_win_rate.png")

    plt.figure(figsize=(7, 5))
    for skill in SKILL_NAMES:
        tx, ty = target_xy(skill, skill_profile)
        plt.scatter([tx], [ty], s=60)
        plt.text(tx + 0.01, ty + 0.01, skill)
    plt.xlabel("target x")
    plt.ylabel("target y")
    plt.title("Canonical skill target coordinates")
    save("target_coordinates.png")

    plt.figure(figsize=(8, 5))
    plotted = False
    for detail in representatives:
        pts = np.array([p["ball_pos"] for p in detail["trajectory_samples"]], dtype=float)
        if len(pts) == 0:
            continue
        row = detail["row"]
        plt.plot(pts[:, 0], pts[:, 1], marker="o", alpha=0.7, label=f"{row['player1_skill']} {row['winner']}")
        plotted = True
    plt.axvline(1.5, color="gray", linestyle="--", label="net")
    plt.xlabel("ball x")
    plt.ylabel("ball y")
    plt.title("Representative long-rally trajectories")
    if plotted:
        plt.legend(fontsize=8)
    save("representative_long_rallies.png")
    return paths


def plot_landing_outputs(out: Path, contacts: list[dict[str, Any]], fixed_player: int, fixed_skill: str) -> list[str]:
    if not contacts:
        return []
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    fixed_player_name = f"player{fixed_player}"
    fixed_contacts = [
        c for c in contacts
        if c["player"] == fixed_player_name and c.get("x_land") not in (None, "") and c.get("y_land") not in (None, "")
    ]
    if fixed_contacts:
        fig, ax = plt.subplots(figsize=(7, 5))
        for skill in SKILL_NAMES:
            pts = np.array(
                [[float(c["x_land"]), float(c["y_land"])] for c in fixed_contacts if c["player2_skill" if fixed_player == 1 else "player1_skill"] == skill],
                dtype=float,
            )
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], alpha=0.6, s=20, label=f"opp {skill}")
        expected = np.array([[float(c["expected_x"]), float(c["expected_y"])] for c in fixed_contacts], dtype=float)
        actual = np.array([[float(c["x_land"]), float(c["y_land"])] for c in fixed_contacts], dtype=float)
        target = expected[0]
        ax.scatter([target[0]], [target[1]], marker="x", s=100, color="black", label=f"expected {fixed_skill}")
        sample_idx = np.linspace(0, len(actual) - 1, min(len(actual), 40), dtype=int)
        ax.quiver(
            expected[sample_idx, 0],
            expected[sample_idx, 1],
            actual[sample_idx, 0] - expected[sample_idx, 0],
            actual[sample_idx, 1] - expected[sample_idx, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
            alpha=0.35,
            color="gray",
        )
        ax.set_xlabel("world landing x")
        ax.set_ylabel("world landing y")
        ax.set_title(f"P{fixed_player} {fixed_skill}: expected vs actual landings")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = plot_dir / "fixed_player_landing_error_vectors.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 5))
    errors = [
        [float(c["error_dist"]) for c in fixed_contacts if c["player2_skill" if fixed_player == 1 else "player1_skill"] == skill and c["error_dist"] not in ("", None)]
        for skill in SKILL_NAMES
    ]
    labels = [skill for skill, values in zip(SKILL_NAMES, errors) if values]
    values = [values for values in errors if values]
    if values:
        try:
            ax.boxplot(values, tick_labels=labels, showmeans=True)
        except TypeError:
            ax.boxplot(values, labels=labels, showmeans=True)
    ax.set_ylabel("landing error distance")
    ax.set_title(f"P{fixed_player} {fixed_skill}: landing error by opponent skill")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    path = plot_dir / "fixed_player_landing_error_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def write_metadata(out: Path, args, rows: list[dict[str, Any]], failures: list[dict[str, Any]], elapsed: float) -> None:
    metadata = {
        "args": vars(args),
        "episode_count": len(rows),
        "validation_failures": len(failures),
        "all_player1_skills_present": sorted({row["player1_skill"] for row in rows}) == sorted(SKILL_NAMES),
        "player1_skills_seen": sorted({row["player1_skill"] for row in rows}),
        "player2_skills_seen": sorted({row["player2_skill"] for row in rows}),
        "skills": [
            {"name": skill, "index": skill_index(skill), "normalized_id": norm_id(skill), "target_xy": target_xy(skill, args.skill_profile)}
            for skill in SKILL_NAMES
        ],
        "elapsed_seconds": elapsed,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))


def write_report(out: Path, args, rows: list[dict[str, Any]], summary: list[dict[str, Any]], failures: list[dict[str, Any]], plots: list[str]) -> Path:
    populated = [row for row in summary if row.get("episode_count")]
    highest = sorted(populated, key=lambda row: row["step_limit_rate"] or 0, reverse=True)[:5]
    report = out / f"fixed_player{args.fixed_player}_diagnostic_report.md"
    lines = [
        f"# Fixed Player {args.fixed_player} Diagnostic Report",
        "",
        "## Files Inspected And Modified",
        "- Inspected: `nash_skills/skills.py`, `nash_skills/env_wrapper.py`, `nash_skills/winner_inference.py`, `diagnostic_fixed_skill.py`.",
        "- Modified: `diagnostic_fixed_skill.py`.",
        "",
        "## Skill Flow",
        f"Player {args.fixed_player} is fixed to the requested canonical skill for every episode. The other player is swept over `SKILL_NAMES`. `SkillEnv` applies physical side/depth targets while observations and encoded states store normalized skill IDs.",
        "",
        "## Command",
        f"`MUJOCO_GL=egl venv/bin/python diagnostic_fixed_skill.py --fixed-player {args.fixed_player} --fixed-skill {args.fixed_skill} --episodes-per-matchup {args.episodes_per_matchup} --steps {args.steps} --seeds {' '.join(map(str, args.seeds))} --reset-mode {args.reset_mode} --skill-profile {args.skill_profile} --gantry-speed-scale {args.gantry_speed_scale} --output-dir {args.output_dir}`",
        "",
        "## Validation Results",
        f"- Rows: {len(rows)}",
        f"- Validation failures: {len(failures)}",
        f"- Fixed player {args.fixed_player} skill: `{args.fixed_skill}`",
        "",
        "## Highest Step-Limit Matchups",
    ]
    for row in highest:
        rate = row["step_limit_rate"]
        lines.append(f"- {row['player1_skill']} vs {row['player2_skill']}: {rate:.3f} ({row['step_limit_count']}/{row['episode_count']})")
    if not highest:
        lines.append("- No populated matchup rows were available.")
    lines += [
        "",
        "## Bugs Found",
        "- No validation bugs found in completed checks." if not failures else "- Validation failures saved to `validation_failures.csv` and `validation_failures.json`.",
        "",
        "## Full Run",
        "- Large 100-episode multi-seed experiment was not run during validation." if len(rows) < len(planned_keys(args.seeds, args.episodes_per_matchup)) else "- Requested run completed.",
        "",
        "## Outputs",
        "- `episodes.csv`",
        "- `summary.csv`",
        "- `metadata.json`",
        "- `representative_trajectories.json`",
        "- `contacts.csv`",
        "- `validation_failures.csv`",
        "- `validation_failures.json`",
    ]
    lines.extend(f"- `{Path(path).relative_to(out)}`" for path in plots)
    report.write_text("\n".join(lines) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-player", type=int, choices=[1, 2], required=True)
    parser.add_argument("--fixed-skill", choices=SKILL_NAMES, required=True)
    parser.add_argument("--episodes-per-matchup", type=int, default=100)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--output-dir", default="skill_eval/p2_center_safe")
    parser.add_argument("--ppo", default="logs/best_model_tracker1/best_model")
    parser.add_argument("--representatives-per-skill", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--reset-mode", choices=["clean", "ready", "carryover"], default="ready")
    parser.add_argument("--skill-profile", choices=SKILL_PROFILE_NAMES, default="current")
    parser.add_argument("--gantry-speed-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes_per_matchup <= 0:
        raise ValueError("--episodes-per-matchup must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    episodes_path = out / "episodes.csv"

    existing = [] if args.no_resume else load_episode_rows(episodes_path)
    validate_resume_rows(existing, args.fixed_skill, args.seeds, args.episodes_per_matchup, args.fixed_player)
    seen = existing_keys(existing)

    from stable_baselines3 import PPO
    from nash_skills.env_wrapper import SkillEnv

    model = PPO.load(args.ppo)
    env = SkillEnv(
        proc_id=1,
        history=HISTORY,
        reset_mode=args.reset_mode,
        skill_profile=args.skill_profile,
        gantry_speed_scale=args.gantry_speed_scale,
    )
    new_rows = []
    details = []
    start = time.monotonic()
    try:
        for seed in args.seeds:
            np.random.seed(seed)
            for other_skill in SKILL_NAMES:
                if args.fixed_player == 1:
                    player1_skill = args.fixed_skill
                    player2_skill = other_skill
                else:
                    player1_skill = other_skill
                    player2_skill = args.fixed_skill
                print(f"Running seed={seed} {player1_skill} vs {player2_skill}", flush=True)
                for episode_index in range(args.episodes_per_matchup):
                    key = episode_key(seed, player1_skill, episode_index, player2_skill)
                    if key in seen:
                        continue
                    np.random.seed(episode_rng_seed(seed, player1_skill, episode_index, player2_skill))
                    row, detail = run_episode(env, model, seed, episode_index, player1_skill, player2_skill, args.steps)
                    new_rows.append(row)
                    details.append(detail)
                    seen.add(key)
                    print(
                        f"  {key} winner={row['winner']} steps={row['physics_steps']} "
                        f"decisions={row['decision_state_count']} validation={row['validation_ok']}",
                        flush=True,
                    )
    finally:
        env.close()

    rows = dedupe_rows([*existing, *new_rows])
    summary = aggregate_rows(rows, args.fixed_skill)
    failures = [row for row in rows if str(row["validation_ok"]).lower() not in {"true", "1", "yes"}]
    representatives = select_representatives(details, args.representatives_per_skill)
    contacts = [contact for detail in details for contact in detail["contacts"]]

    write_csv(episodes_path, rows, EPISODE_FIELDS)
    write_csv(out / "summary.csv", summary, SUMMARY_FIELDS)
    write_csv(out / "contacts.csv", contacts, CONTACT_FIELDS)
    write_csv(out / "validation_failures.csv", failures, EPISODE_FIELDS)
    (out / "validation_failures.json").write_text(json.dumps(failures, indent=2))
    (out / "representative_trajectories.json").write_text(json.dumps(representatives, indent=2))
    write_metadata(out, args, rows, failures, time.monotonic() - start)
    plots = plot_outputs(out, rows, summary, representatives, args.steps, args.skill_profile)
    plots.extend(plot_landing_outputs(out, contacts, args.fixed_player, args.fixed_skill))
    report = write_report(out, args, rows, summary, failures, plots)
    print(f"Wrote outputs to {out}", flush=True)
    print(f"Report: {report}", flush=True)


if __name__ == "__main__":
    main()
