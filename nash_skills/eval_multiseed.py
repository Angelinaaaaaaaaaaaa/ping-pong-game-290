"""
Multi-seed evaluation for the 5-skill Nash pipeline.

Runs all five learned strategies (nash-p-hard, nash-p-br, nash-p-minimax,
nash-p-adaptive, ibr) against all six fixed-skill opponents across seeds 0..4,
then aggregates mean/std win rates.  30 matchups × n_seeds total runs.

DO NOT retrain models.  Only evaluation — no weight updates.

Usage (from project root):
    MUJOCO_GL=cgl venv/bin/python nash_skills/eval_multiseed.py
    MUJOCO_GL=cgl venv/bin/python nash_skills/eval_multiseed.py \\
        --seeds 0 1 2 3 4 \\
        --episodes 60 --steps 6000 \\
        --output-dir skill_eval/multiseed

Output files (all written to --output-dir):
    raw_results.csv        — one row per (seed, strategy, opponent)
    raw_results.json       — same data as JSON
    aggregated.csv         — mean/std win_rate per (strategy, opponent)
    aggregated.json        — same data as JSON
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import json
import random

import numpy as np
import torch

from nash_skills.eval_matchup import (
    run_matchup,
    save_csv,
    most_used_skill,
    dominant_skill_fraction,
)

# --------------------------------------------------------------------------- #
# Fixed evaluation spec — symmetric across all three strategies               #
# --------------------------------------------------------------------------- #

STRATEGIES = ["nash-p-hard", "nash-p-br", "nash-p-minimax", "nash-p-adaptive", "ibr"]

OPPONENTS = ["random", "left", "right", "left_short", "right_short", "center_safe"]

# All 30 matchups: 5 strategies × 6 opponents
MULTISEED_MATCHUPS = [
    (strategy, opp)
    for strategy in STRATEGIES
    for opp in OPPONENTS
]

DEFAULT_SEEDS          = [0, 1, 2, 3, 4]
DEFAULT_EPISODES       = 60
DEFAULT_MAX_STEPS      = 6000
DEFAULT_WARMUP         = 300
DEFAULT_OUTPUT_DIR     = "skill_eval/multiseed"

PPO_MODEL_PATH         = "logs/best_model_tracker1/best_model"
MODEL_P_5SK_PATH       = "models/model_p_5skill.pth"


# --------------------------------------------------------------------------- #
# Seeding helper                                                               #
# --------------------------------------------------------------------------- #

def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and Torch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Raw result row                                                               #
# --------------------------------------------------------------------------- #

def _result_to_row(seed: int, result) -> dict:
    """Convert a MatchupResult + seed into a flat CSV row."""
    return {
        "seed":                  seed,
        "strategy1":             result.strategy1,
        "strategy2":             result.strategy2,
        "episodes":              result.episodes,
        "truncated_episodes":    result.truncated_episodes,
        "ego_wins":              result.ego_wins,
        "opp_wins":              result.opp_wins,
        "win_rate":              round(result.win_rate, 4) if result.win_rate is not None else "",
        "avg_steps_per_episode": round(result.avg_steps_per_episode, 2)
                                 if result.avg_steps_per_episode is not None else "",
        "avg_rally_length":      round(result.avg_rally_length, 2)
                                 if result.avg_rally_length is not None else "",
        "most_used_skill":       most_used_skill(result) or "",
        "dominant_fraction":     round(dominant_skill_fraction(result), 4)
                                 if dominant_skill_fraction(result) is not None else "",
        "total_steps":           result.total_steps,
    }


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #

def aggregate(raw_rows: list) -> list:
    """
    Group raw rows by (strategy1, strategy2), compute mean and std of win_rate.

    Returns list of dicts with keys:
        strategy1, strategy2, n_seeds,
        win_rate_mean, win_rate_std,
        avg_steps_mean, avg_steps_std,
        avg_rally_mean, avg_rally_std
    """
    from collections import defaultdict

    groups: dict = defaultdict(list)
    for row in raw_rows:
        key = (row["strategy1"], row["strategy2"])
        groups[key].append(row)

    agg_rows = []
    for (s1, s2), rows in groups.items():
        win_rates   = [r["win_rate"]              for r in rows if r["win_rate"]              != ""]
        avg_steps   = [r["avg_steps_per_episode"] for r in rows if r["avg_steps_per_episode"] != ""]
        avg_rallies = [r["avg_rally_length"]      for r in rows if r["avg_rally_length"]      != ""]

        agg_rows.append({
            "strategy1":       s1,
            "strategy2":       s2,
            "n_seeds":         len(rows),
            "win_rate_mean":   round(float(np.mean(win_rates)),   4) if win_rates   else "",
            "win_rate_std":    round(float(np.std(win_rates)),    4) if win_rates   else "",
            "avg_steps_mean":  round(float(np.mean(avg_steps)),   2) if avg_steps   else "",
            "avg_steps_std":   round(float(np.std(avg_steps)),    2) if avg_steps   else "",
            "avg_rally_mean":  round(float(np.mean(avg_rallies)), 2) if avg_rallies else "",
            "avg_rally_std":   round(float(np.std(avg_rallies)),  2) if avg_rallies else "",
        })

    # Sort by strategy then opponent for readability
    agg_rows.sort(key=lambda r: (r["strategy1"], r["strategy2"]))
    return agg_rows


# --------------------------------------------------------------------------- #
# Save helpers                                                                 #
# --------------------------------------------------------------------------- #

def save_raw_csv(rows: list, path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_agg_csv(rows: list, path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(data, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed 5-skill evaluation: nash-p / nash-p-adaptive / ibr vs all opponents"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help=f"Random seeds to evaluate (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES,
        help=f"Completed episodes per seed per matchup (default: {DEFAULT_EPISODES})",
    )
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_MAX_STEPS,
        help=f"Max steps per episode before truncation (default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=f"One-time warmup steps per matchup (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for all output files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--tau", type=float, default=0.2,
        help="Softmax temperature for nash-p-adaptive (default: 0.2)",
    )
    parser.add_argument(
        "--confidence-margin", type=float, default=0.05, dest="confidence_margin",
        help="Gap threshold for nash-p-adaptive deterministic cutoff (default: 0.05)",
    )
    parser.add_argument(
        "--v2", action="store_true", default=False,
        help="Use 4-skill v2 model_p_v2.pth (76-dim) instead of model_p_5skill.pth (116-dim)",
    )
    parser.add_argument(
        "--v2-5skill", action="store_true", default=False, dest="v2_5skill",
        help="Use 5-skill v2 model_p_5skill_v2.pth (76-dim, discounted returns, all 5 skills)",
    )
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from model_arch import SimpleModel

    print("Loading models ...")
    ppo = PPO.load(PPO_MODEL_PATH)

    if args.v2_5skill:
        from nash_skills.v2.state_encoder import STATE_DIM as V2_STATE_DIM
        from nash_skills.v2.state_encoder import encode_ego, encode_opp
        model_p_path = "models/model_p_5skill_v2.pth"
        model_p = SimpleModel(V2_STATE_DIM, [64, 32, 16], 1, last_layer_activation=None)
        try:
            model_p.load_state_dict(torch.load(model_p_path, weights_only=True))
        except TypeError:
            model_p.load_state_dict(torch.load(model_p_path))

        def state_encoder_fn(obs, info, player):
            return encode_ego(obs, info) if player == 1 else encode_opp(obs, info)

    elif args.v2:
        from nash_skills.v2.state_encoder import STATE_DIM as V2_STATE_DIM
        from nash_skills.v2.state_encoder import encode_ego, encode_opp
        model_p_path = "models/model_p_v2.pth"
        model_p = SimpleModel(V2_STATE_DIM, [64, 32, 16], 1, last_layer_activation=None)
        try:
            model_p.load_state_dict(torch.load(model_p_path, weights_only=True))
        except TypeError:
            model_p.load_state_dict(torch.load(model_p_path))

        def state_encoder_fn(obs, info, player):
            return encode_ego(obs, info) if player == 1 else encode_opp(obs, info)

    else:
        model_p_path = MODEL_P_5SK_PATH
        model_p = SimpleModel(116, [64, 32, 16], 1, last_layer_activation=None)
        try:
            model_p.load_state_dict(torch.load(model_p_path, weights_only=True))
        except TypeError:
            model_p.load_state_dict(torch.load(model_p_path))
        state_encoder_fn = None

    model_p.eval()

    # Q-value models — required for ibr strategy
    needs_q = any(s == "ibr" for s, _ in MULTISEED_MATCHUPS) or any(
        s == "ibr" for _, s in MULTISEED_MATCHUPS
    )
    model1 = model2 = None
    if needs_q:
        if args.v2_5skill:
            from nash_skills.v2.state_encoder import STATE_DIM as V2_STATE_DIM
            _sdim = V2_STATE_DIM
            _q1_path, _q2_path = "models/model1_5skill_v2.pth", "models/model2_5skill_v2.pth"
        elif args.v2:
            from nash_skills.v2.state_encoder import STATE_DIM as V2_STATE_DIM
            _sdim = V2_STATE_DIM
            _q1_path, _q2_path = "models/model1_v2.pth", "models/model2_v2.pth"
        else:
            _sdim = 116
            _q1_path, _q2_path = "models/model1_5skill.pth", "models/model2_5skill.pth"
        model1 = SimpleModel(_sdim, [64, 32, 16], 1)
        model2 = SimpleModel(_sdim, [64, 32, 16], 1)
        for _m, _path in [(model1, _q1_path), (model2, _q2_path)]:
            try:
                _m.load_state_dict(torch.load(_path, weights_only=True))
            except TypeError:
                _m.load_state_dict(torch.load(_path))
        model1.eval()
        model2.eval()
        print(f"  Q-models:  {_q1_path}, {_q2_path}")

    print(f"  PPO:       {PPO_MODEL_PATH}")
    print(f"  Potential: {model_p_path}")
    print()

    n_matchups = len(MULTISEED_MATCHUPS)
    n_seeds    = len(args.seeds)
    total_runs = n_matchups * n_seeds
    print(f"Plan: {n_matchups} matchups × {n_seeds} seeds = {total_runs} runs")
    print(f"      {args.episodes} completed episodes each, max_steps={args.steps}")
    print()

    raw_rows: list = []
    run_idx = 0

    for seed in args.seeds:
        print(f"=== Seed {seed} ===")
        set_global_seed(seed)

        for strategy1, strategy2 in MULTISEED_MATCHUPS:
            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] {strategy1} vs {strategy2} ...", flush=True)

            result = run_matchup(
                strategy1=strategy1,
                strategy2=strategy2,
                ppo=ppo,
                model_p=model_p,
                n_episodes=args.episodes,
                max_steps_per_episode=args.steps,
                warmup_steps=args.warmup,
                state_encoder_fn=state_encoder_fn,
                tau=args.tau,
                confidence_margin=args.confidence_margin,
                model1=model1,
                model2=model2,
            )

            wr  = f"{result.win_rate:.0%}" if result.win_rate is not None else "---"
            arl = f"{result.avg_rally_length:.1f}" if result.avg_rally_length is not None else "---"
            print(f"    done={result.episodes}  trunc={result.truncated_episodes}"
                  f"  win_rate={wr}  avg_rally={arl}", flush=True)

            raw_rows.append(_result_to_row(seed, result))

    # Write raw results
    out = args.output_dir
    raw_csv_path  = os.path.join(out, "raw_results.csv")
    raw_json_path = os.path.join(out, "raw_results.json")
    agg_csv_path  = os.path.join(out, "aggregated.csv")
    agg_json_path = os.path.join(out, "aggregated.json")

    save_raw_csv(raw_rows, raw_csv_path)
    save_json(raw_rows, raw_json_path)
    print(f"\nRaw results saved to: {raw_csv_path}")

    agg_rows = aggregate(raw_rows)
    save_agg_csv(agg_rows, agg_csv_path)
    save_json({"aggregated": agg_rows}, agg_json_path)
    print(f"Aggregated saved to:  {agg_csv_path}")

    # Print aggregated table
    print()
    header = f"{'Strategy':<20} {'vs':<14} {'Mean WR':>8} {'Std WR':>8} {'Seeds':>6}"
    sep    = "-" * len(header)
    print(f"\n{'AGGREGATED RESULTS (mean ± std win rate)':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)
    for row in agg_rows:
        wr_mean = f"{row['win_rate_mean']:.0%}" if row["win_rate_mean"] != "" else "---"
        wr_std  = f"{row['win_rate_std']:.0%}"  if row["win_rate_std"]  != "" else "---"
        print(f"{row['strategy1']:<20} {row['strategy2']:<14} {wr_mean:>8} {wr_std:>8} {row['n_seeds']:>6}")
    print(sep)


if __name__ == "__main__":
    main()
