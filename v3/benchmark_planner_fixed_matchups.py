#!/usr/bin/env python3
"""Benchmark legacy and common-pool planners on identical fixed-skill rallies."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nash_skills.skills import SKILL_NAMES, SKILL_PROFILE_NAMES, skill_index  # noqa: E402
from v3.skill_semantics import world_target_xy  # noqa: E402
from v3.eval_fixed_preposition import HISTORY, build_ppo_action  # noqa: E402
from v3.env_wrapper import V3PrePositionEnv  # noqa: E402


def episode_rng_seed(seed: int, skill1: str, episode_index: int, skill2: str) -> int:
    """Match the deterministic fixed-skill diagnostic reset convention."""
    return int(
        (
            (seed + 1_000_003) * 1_000_003
            + skill_index(skill1) * 10_007
            + skill_index(skill2) * 101
            + episode_index
        )
        % (2**32 - 1)
    )


def parse_matchups(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.all_matchups:
        return [(skill1, skill2) for skill1 in SKILL_NAMES for skill2 in SKILL_NAMES]
    return [tuple(pair) for pair in (args.matchup or [["left", "right"]])]


def normalize_winner(value: Any) -> str:
    text = str(value or "").lower()
    if text in {"ego", "player1", "p1", "1"}:
        return "player1"
    if text in {"opp", "player2", "p2", "2"}:
        return "player2"
    if text in {"truncated", "step_limit", "max_steps"}:
        return "truncated"
    return text or "unknown"


def target_landing_error(event: dict[str, Any], skill_profile: str) -> float | None:
    x_value = event.get("actual_landing_x", "")
    y_value = event.get("actual_landing_y", "")
    if x_value in ("", None) or y_value in ("", None):
        return None
    player = int(event["player"])
    skill = str(event["skill"])
    target_x, target_y = world_target_xy(player, skill, profile=skill_profile)
    return float(np.hypot(float(x_value) - target_x, float(y_value) - target_y))


def run_episode(
    *,
    model: Any,
    planner: str,
    skill1: str,
    skill2: str,
    seed: int,
    episode_index: int,
    reset_mode: str,
    skill_profile: str,
    max_steps: int,
    gantry_speed_scale: float,
) -> dict[str, Any]:
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
    events: list[dict[str, Any]] = []
    info: dict[str, Any] = {}
    done = False
    steps = 0
    try:
        obs, info = env.reset()
        for step in range(1, max_steps + 1):
            steps = step
            action = build_ppo_action(model, obs, info)
            obs, _reward, done, _truncated, info = env.step(action)
            if done:
                break
        events = list(env._planner_events)
    finally:
        env.close()

    contact_count = len(events)
    legal_count = sum(bool(event.get("legal_return")) for event in events)
    net_count = sum(bool(event.get("net_failure")) for event in events)
    out_count = sum(bool(event.get("out_of_bounds")) for event in events)
    landing_errors = [
        error for event in events
        if (error := target_landing_error(event, skill_profile)) is not None
    ]
    method_counts = defaultdict(int)
    for event in events:
        method_counts[str(event.get("solution_method", "none"))] += 1
    legality_fallback_count = sum(
        bool(event.get("legacy_legality_filter_fallback")) for event in events
    )
    net_candidate_counts = [
        float(event["legacy_legal_after_net_count"])
        for event in events
        if event.get("legacy_legal_after_net_count", "") not in ("", None)
    ]
    table_candidate_counts = [
        float(event["legacy_legal_after_table_count"])
        for event in events
        if event.get("legacy_legal_after_table_count", "") not in ("", None)
    ]

    winner = normalize_winner(info.get("winner"))
    if winner == "unknown" and events:
        winner = normalize_winner(events[-1].get("rally_winner"))
    return {
        "planner": planner,
        "seed": seed,
        "episode_index": episode_index,
        "skill1": skill1,
        "skill2": skill2,
        "winner": winner,
        "steps": steps,
        "rally_length": contact_count,
        "contact_occurred": bool(contact_count),
        "contact_count": contact_count,
        "legal_return_count": legal_count,
        "net_failure_count": net_count,
        "out_of_bounds_count": out_count,
        "missed_contact": not bool(contact_count),
        "actual_landing_error_mean": float(np.mean(landing_errors)) if landing_errors else "",
        "actual_landing_error_count": len(landing_errors),
        "explicit_selected_count": method_counts["explicit"],
        "grid_selected_count": method_counts["grid"],
        "legacy_legality_filter_fallback_count": legality_fallback_count,
        "legacy_legal_after_net_count_mean": float(np.mean(net_candidate_counts)) if net_candidate_counts else "",
        "legacy_legal_after_table_count_mean": float(np.mean(table_candidate_counts)) if table_candidate_counts else "",
        "termination_reason": str(info.get("termination_reason", "step_limit")),
        "truncated": not done,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["planner"], row["skill1"], row["skill2"])].append(row)
    output = []
    for (planner, skill1, skill2), group in sorted(groups.items()):
        episodes = len(group)
        contacts = [row for row in group if row["contact_count"] > 0]
        contact_events = sum(row["contact_count"] for row in group)
        legal_events = sum(row["legal_return_count"] for row in group)
        landing_errors = [
            float(row["actual_landing_error_mean"])
            for row in contacts
            if row["actual_landing_error_mean"] not in ("", None)
        ]
        p1_wins = sum(row["winner"] == "player1" for row in group)
        p2_wins = sum(row["winner"] == "player2" for row in group)
        output.append({
            "planner": planner,
            "skill1": skill1,
            "skill2": skill2,
            "episodes": episodes,
            "racket_contact_rate": len(contacts) / episodes if episodes else "",
            "contacts_per_episode": contact_events / episodes if episodes else "",
            "legal_return_rate": legal_events / contact_events if contact_events else "",
            "actual_landing_error_mean": float(np.mean(landing_errors)) if landing_errors else "",
            "average_rally_length": float(np.mean([row["rally_length"] for row in group])) if group else "",
            "miss_rate": sum(bool(row["missed_contact"]) for row in group) / episodes if episodes else "",
            "net_failure_rate": sum(row["net_failure_count"] for row in group) / contact_events if contact_events else "",
            "out_of_bounds_rate": sum(row["out_of_bounds_count"] for row in group) / contact_events if contact_events else "",
            "truncated_rate": sum(bool(row["truncated"]) for row in group) / episodes if episodes else "",
            "player1_win_rate": p1_wins / episodes if episodes else "",
            "player2_win_rate": p2_wins / episodes if episodes else "",
            "skill1_win_rate": p1_wins / episodes if episodes else "",
            "explicit_selected_count": sum(row["explicit_selected_count"] for row in group),
            "grid_selected_count": sum(row["grid_selected_count"] for row in group),
        })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["planner"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", choices=["legacy", "legacy_legal", "optimized", "both"], default="both")
    parser.add_argument(
        "--compare-legacy-legality",
        action="store_true",
        help="Compare original legacy normal selection against V3 legality-filtered legacy.",
    )
    parser.add_argument("--matchup", nargs=2, action="append", metavar=("P1_SKILL", "P2_SKILL"))
    parser.add_argument("--all-matchups", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--ppo", default="logs/best_model_tracker1/best_model")
    parser.add_argument("--reset-mode", choices=["clean", "ready", "carryover"], default="clean")
    parser.add_argument("--skill-profile", choices=SKILL_PROFILE_NAMES, default="current")
    parser.add_argument("--gantry-speed-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", default="skill_eval/v3/planner_fixed_matchup_benchmark")
    args = parser.parse_args(argv)

    for skill1, skill2 in parse_matchups(args):
        if skill1 not in SKILL_NAMES or skill2 not in SKILL_NAMES:
            parser.error(f"unknown matchup: {skill1} {skill2}")
    if args.compare_legacy_legality:
        planners = ["legacy", "legacy_legal"]
    else:
        planners = ["legacy", "optimized"] if args.planner == "both" else [args.planner]
    matchups = parse_matchups(args)

    from stable_baselines3 import PPO

    print(f"Loading PPO: {args.ppo}", flush=True)
    model = PPO.load(args.ppo)
    rows: list[dict[str, Any]] = []
    total = len(planners) * len(matchups) * len(args.seeds) * args.episodes
    completed = 0
    for planner in planners:
        for skill1, skill2 in matchups:
            for seed in args.seeds:
                for episode_index in range(args.episodes):
                    completed += 1
                    print(
                        f"[{completed}/{total}] planner={planner} seed={seed} "
                        f"{skill1} vs {skill2} episode={episode_index}",
                        flush=True,
                    )
                    rows.append(run_episode(
                        model=model,
                        planner=planner,
                        skill1=skill1,
                        skill2=skill2,
                        seed=seed,
                        episode_index=episode_index,
                        reset_mode=args.reset_mode,
                        skill_profile=args.skill_profile,
                        max_steps=args.steps,
                        gantry_speed_scale=args.gantry_speed_scale,
                    ))

    output_dir = Path(args.output_dir)
    episode_path = output_dir / "episodes.csv"
    summary_path = output_dir / "summary.csv"
    write_csv(episode_path, rows)
    summary = summarize(rows)
    write_csv(summary_path, summary)
    metadata = {
        "ppo": args.ppo,
        "planners": planners,
        "compare_legacy_legality": bool(args.compare_legacy_legality),
        "matchups": matchups,
        "seeds": args.seeds,
        "episodes_per_matchup": args.episodes,
        "reset_mode": args.reset_mode,
        "skill_profile": args.skill_profile,
        "outputs": {"episodes": str(episode_path), "summary": str(summary_path)},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
