#!/usr/bin/env python3
"""Run identical policy evaluations with legacy and optimized target planners."""

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

import nash_skills.env_wrapper as env_wrapper  # noqa: E402
import nash_skills.eval_matchup as baseline_eval  # noqa: E402
from nash_skills.skills import SKILL_NAMES  # noqa: E402
from v3.planner_env import PlannerSkillEnv  # noqa: E402
from v3.racket_target_planner import PlannerConfig  # noqa: E402


def _json_value(value: Any) -> str:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def write_diagnostics(rows: list[dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("planner,skill\n")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_value(value) for key, value in row.items()})


def _mean(rows: list[dict[str, Any]], field: str) -> float | str:
    values = []
    for row in rows:
        value = row.get(field, "")
        if value in ("", None):
            continue
        values.append(float(value))
    return "" if not values else float(np.mean(values))


def summarize_by_skill(rows: list[dict[str, Any]], planner: str) -> list[dict[str, Any]]:
    output = []
    for skill in SKILL_NAMES:
        selected = [row for row in rows if row.get("skill") == skill]
        attempts = len(selected)
        def rate(field: str) -> float:
            values = [row[field] for row in selected if row.get(field, "") not in ("", None)]
            return float(np.mean([bool(value) for value in values])) if values else float("nan")

        output.append({
            "planner": planner,
            "skill": skill,
            "attempts": attempts,
            "explicit_selected_rate": (
                float(np.mean([row.get("solution_method") == "explicit" for row in selected]))
                if selected else float("nan")
            ),
            "grid_selected_rate": (
                float(np.mean([row.get("solution_method") == "grid" for row in selected]))
                if selected else float("nan")
            ),
            "legal_return_rate": rate("legal_return"),
            "out_of_bounds_rate": rate("out_of_bounds"),
            "net_failure_rate": rate("net_failure"),
            "mean_predicted_actual_landing_error": _mean(selected, "predicted_actual_landing_error"),
            "mean_selected_landing_error": _mean(selected, "selected_landing_error"),
            "mean_racket_position_tracking_error": _mean(selected, "actual_racket_position_error"),
            "mean_racket_orientation_tracking_error_rad": _mean(selected, "actual_racket_orientation_error_rad"),
            "mean_planner_calls_per_return": _mean(selected, "planner_calls_per_return"),
            "mean_analytic_solve_time": _mean(selected, "analytic_solve_time"),
            "mean_fallback_count": _mean(selected, "fallback_count"),
            "mean_fallback_rate": _mean(selected, "fallback_rate"),
            "mean_planner_time": _mean(selected, "average_planner_time"),
            "legacy_legality_filter_fallback_rate": rate("legacy_legality_filter_fallback"),
            "mean_legacy_normal_candidate_count": _mean(selected, "legacy_normal_candidate_count"),
            "mean_legacy_legal_after_net_count": _mean(selected, "legacy_legal_after_net_count"),
            "mean_legacy_legal_after_table_count": _mean(selected, "legacy_legal_after_table_count"),
            "mean_selected_predicted_net_crossing_height": _mean(
                selected, "selected_predicted_net_crossing_height"
            ),
            "rally_win_rate_when_skill_selected": rate("player_won"),
        })
    return output


def _summary_lookup(rows: list[dict[str, Any]], planner: str, skill: str) -> dict[str, Any]:
    return next(row for row in rows if row["planner"] == planner and row["skill"] == skill)


def write_symmetry(rows: list[dict[str, Any]], path: str) -> None:
    comparisons = [
        ("left_vs_right", "left", "right"),
        ("left_short_vs_right_short", "left_short", "right_short"),
    ]
    fields = [
        "planner", "comparison", "skill_a", "skill_b", "legal_rate_a", "legal_rate_b",
        "legal_rate_abs_gap", "win_rate_a", "win_rate_b", "win_rate_abs_gap",
        "out_of_bounds_rate_a", "out_of_bounds_rate_b", "landing_error_a", "landing_error_b",
    ]
    output = []
    planners = sorted({row["planner"] for row in rows})
    for planner in planners:
        for name, skill_a, skill_b in comparisons:
            a = _summary_lookup(rows, planner, skill_a)
            b = _summary_lookup(rows, planner, skill_b)
            output.append({
                "planner": planner,
                "comparison": name,
                "skill_a": skill_a,
                "skill_b": skill_b,
                "legal_rate_a": a["legal_return_rate"],
                "legal_rate_b": b["legal_return_rate"],
                "legal_rate_abs_gap": abs(float(a["legal_return_rate"]) - float(b["legal_return_rate"])),
                "win_rate_a": a["rally_win_rate_when_skill_selected"],
                "win_rate_b": b["rally_win_rate_when_skill_selected"],
                "win_rate_abs_gap": abs(float(a["rally_win_rate_when_skill_selected"]) - float(b["rally_win_rate_when_skill_selected"])),
                "out_of_bounds_rate_a": a["out_of_bounds_rate"],
                "out_of_bounds_rate_b": b["out_of_bounds_rate"],
                "landing_error_a": a["mean_predicted_actual_landing_error"],
                "landing_error_b": b["mean_predicted_actual_landing_error"],
            })

        short = [_summary_lookup(rows, planner, skill) for skill in ("left_short", "right_short")]
        deep = [_summary_lookup(rows, planner, skill) for skill in ("left", "right")]
        for metric in ("legal_return_rate", "rally_win_rate_when_skill_selected"):
            short_values = [float(row[metric]) for row in short if not np.isnan(float(row[metric]))]
            deep_values = [float(row[metric]) for row in deep if not np.isnan(float(row[metric]))]
            output.append({
                "planner": planner,
                "comparison": f"short_vs_deep_{metric}",
                "skill_a": "left_short/right_short",
                "skill_b": "left/right",
                "legal_rate_a": float(np.mean(short_values)) if metric == "legal_return_rate" and short_values else "",
                "legal_rate_b": float(np.mean(deep_values)) if metric == "legal_return_rate" and deep_values else "",
                "legal_rate_abs_gap": "",
                "win_rate_a": float(np.mean(short_values)) if metric == "rally_win_rate_when_skill_selected" and short_values else "",
                "win_rate_b": float(np.mean(deep_values)) if metric == "rally_win_rate_when_skill_selected" and deep_values else "",
                "win_rate_abs_gap": "",
                "out_of_bounds_rate_a": "",
                "out_of_bounds_rate_b": "",
                "landing_error_a": "",
                "landing_error_b": "",
            })

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


def _build_baseline_argv(args: argparse.Namespace, planner: str) -> list[str]:
    prefix = [
        "eval_matchup.py",
        "--episodes", str(args.episodes),
        "--steps", str(args.steps),
        "--warmup", str(args.warmup),
        "--output-csv", args.output_dir + f"/matchup_{planner}.csv",
        "--output-json", args.output_dir + f"/matchup_{planner}.json",
        "--reset-mode", args.reset_mode,
        "--skill-profile", args.skill_profile,
        "--gantry-speed-scale", str(args.gantry_speed_scale),
    ]
    if args.max_total_steps is not None:
        prefix.extend(["--max-total-steps", str(args.max_total_steps)])
    if args.seed is not None:
        prefix.extend(["--seed", str(args.seed)])
    if args.matchup is not None:
        prefix.extend(["--matchup", *args.matchup])
    if args.v3_5skill:
        prefix.append("--v3-5skill")
    if args.v2_5skill:
        prefix.append("--v2-5skill")
    if args.v2:
        prefix.append("--v2")
    if args.arch != "simple":
        prefix.extend(["--arch", args.arch])
    if args.selection_mode != "argmax":
        prefix.extend(["--selection-mode", args.selection_mode])
    if args.temperature != 1.0:
        prefix.extend(["--temperature", str(args.temperature)])
    if args.epsilon != 0.0:
        prefix.extend(["--epsilon", str(args.epsilon)])
    if args.model_dir is not None:
        prefix.extend(["--model-dir", args.model_dir])
    return prefix


def run_one(args: argparse.Namespace, planner: str) -> list[dict[str, Any]]:
    config = PlannerConfig(
        planner=planner,
        contact_candidates=args.contact_candidates,
        normal_candidates=args.normal_candidates,
        net_safety_margin=args.net_safety_margin,
        net_safety_weight=args.net_safety_weight,
        net_safety_scale=args.net_safety_scale,
        table_x_safety_margin=args.table_x_safety_margin,
        table_y_safety_margin=args.table_y_safety_margin,
        landing_weight=args.landing_weight,
        reach_position_weight=args.reach_position_weight,
        reach_orientation_weight=args.reach_orientation_weight,
        translation_speed_weight=args.translation_speed_weight,
        rotation_speed_weight=args.rotation_speed_weight,
        paddle_center_weight=args.paddle_center_weight,
        reference_weight=args.reference_weight,
        landing_scale=args.landing_scale,
        reach_position_scale=args.reach_position_scale,
        paddle_center_scale=args.paddle_center_scale,
        reference_scale=args.reference_scale,
        flight_time_min=args.flight_time_min,
        flight_time_max=args.flight_time_max,
        flight_time_samples=args.flight_time_samples,
        explicit_landing_tolerance=args.explicit_landing_tolerance,
        reflection_tolerance=args.reflection_tolerance,
        max_position_change=args.max_position_change,
        max_orientation_change_rad=args.max_orientation_change_rad,
        max_translation_speed=args.max_translation_speed,
        max_rotation_speed_rad_s=args.max_rotation_speed_rad_s,
        hard_translation_speed=args.hard_translation_speed,
        hard_rotation_speed_rad_s=args.hard_rotation_speed_rad_s,
        contact_height_candidates=args.contact_height_candidates,
        contact_height_min=args.contact_height_min,
        contact_height_max=args.contact_height_max,
        normal_y_min=args.normal_y_min,
        normal_y_max=args.normal_y_max,
        normal_z_min=args.normal_z_min,
        normal_z_max=args.normal_z_max,
    )
    config.validate()
    PlannerSkillEnv.clear_diagnostics()
    original_skill_env = env_wrapper.SkillEnv

    def factory(*env_args, **env_kwargs):
        return PlannerSkillEnv(*env_args, planner_config=config, **env_kwargs)

    env_wrapper.SkillEnv = factory
    old_argv = sys.argv
    try:
        if args.seed is not None:
            np.random.seed(args.seed)
        sys.argv = _build_baseline_argv(args, planner)
        baseline_eval.main()
    finally:
        sys.argv = old_argv
        env_wrapper.SkillEnv = original_skill_env

    rows = PlannerSkillEnv.diagnostics()
    config_path = args.output_dir + f"/planner_{planner}_config.json"
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    diagnostics_path = args.output_dir + f"/planner_{planner}_attempts.csv"
    write_diagnostics(rows, diagnostics_path)
    print(f"Planner config saved to: {config_path}")
    print(f"Planner diagnostics saved to: {diagnostics_path}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare legacy and optimized V3 racket-target planners.")
    parser.add_argument("--planner", choices=["legacy", "legacy_legal", "optimized", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-total-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--matchup", nargs=2, default=None, metavar=("STRATEGY1", "STRATEGY2"))
    parser.add_argument("--reset-mode", choices=["clean", "ready", "carryover"], default="ready")
    parser.add_argument("--skill-profile", choices=["current", "moderate_aggressive", "aggressive"], default="aggressive")
    parser.add_argument("--gantry-speed-scale", type=float, default=1.0)
    parser.add_argument("--v3-5skill", action="store_true", default=True)
    parser.add_argument("--v2-5skill", action="store_true", default=False)
    parser.add_argument("--v2", action="store_true", default=False)
    parser.add_argument("--arch", choices=["simple", "factored"], default="simple")
    parser.add_argument("--selection-mode", choices=["argmax", "softmax", "epsilon_argmax", "epsilon_softmax"], default="argmax")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output-dir", default="skill_eval/v3/planner_ablation")

    parser.add_argument("--contact-candidates", type=int, default=21)
    parser.add_argument("--normal-candidates", type=int, default=121)
    parser.add_argument("--net-safety-margin", type=float, default=0.03)
    parser.add_argument("--table-x-safety-margin", type=float, default=0.03)
    parser.add_argument("--table-y-safety-margin", type=float, default=0.03)
    parser.add_argument("--landing-weight", type=float, default=4.0)
    parser.add_argument("--reach-position-weight", type=float, default=0.4)
    parser.add_argument("--reach-orientation-weight", type=float, default=0.2)
    parser.add_argument("--translation-speed-weight", type=float, default=0.6)
    parser.add_argument("--rotation-speed-weight", type=float, default=0.15)
    parser.add_argument("--net-safety-weight", type=float, default=0.10)
    parser.add_argument("--paddle-center-weight", type=float, default=0.15)
    parser.add_argument("--reference-weight", type=float, default=0.05)
    parser.add_argument("--landing-scale", type=float, default=1.37)
    parser.add_argument("--reach-position-scale", type=float, default=1.0)
    parser.add_argument("--paddle-center-scale", type=float, default=0.2)
    parser.add_argument("--reference-scale", type=float, default=0.5)
    parser.add_argument("--flight-time-min", type=float, default=0.02)
    parser.add_argument("--flight-time-max", type=float, default=3.0)
    parser.add_argument("--flight-time-samples", type=int, default=96)
    parser.add_argument("--explicit-landing-tolerance", type=float, default=0.01)
    parser.add_argument("--reflection-tolerance", type=float, default=1e-5)
    parser.add_argument("--max-position-change", type=float, default=1.5)
    parser.add_argument("--max-orientation-change-rad", type=float, default=1.5)
    parser.add_argument("--max-translation-speed", type=float, default=1.10)
    parser.add_argument("--max-rotation-speed-rad-s", type=float, default=0.60)
    parser.add_argument("--hard-translation-speed", type=float, default=3.0)
    parser.add_argument("--hard-rotation-speed-rad-s", type=float, default=3.0)
    parser.add_argument("--net-safety-scale", type=float, default=0.20)
    parser.add_argument("--contact-height-candidates", type=int, default=7)
    parser.add_argument("--contact-height-min", type=float, default=0.56)
    parser.add_argument("--contact-height-max", type=float, default=1.20)
    parser.add_argument("--normal-y-min", type=float, default=-1.0)
    parser.add_argument("--normal-y-max", type=float, default=1.0)
    parser.add_argument("--normal-z-min", type=float, default=-0.1)
    parser.add_argument("--normal-z-max", type=float, default=0.1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    planners = ["legacy", "optimized"] if args.planner == "both" else [args.planner]
    all_summary = []
    for planner in planners:
        print(f"\n=== Running planner={planner} ===")
        rows = run_one(args, planner)
        all_summary.extend(summarize_by_skill(rows, planner))

    summary_path = args.output_dir + "/planner_skill_comparison.csv"
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        fields = list(all_summary[0].keys()) if all_summary else ["planner", "skill"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_summary)
    symmetry_path = args.output_dir + "/planner_symmetry.csv"
    if all_summary:
        write_symmetry(all_summary, symmetry_path)
    print(f"Skill comparison saved to: {summary_path}")
    print(f"Symmetry comparison saved to: {symmetry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
