"""
Diagnose which source of winner inference is firing for each rally.

For each rally in the pickle, re-run infer_terminal_winner on the LAST raw_obs
and find which of these tiers decided the winner:
  (1) explicit info  — none (info not stored in pickle)
  (2) racket bounds  — none (info not stored)
  (3) ball velocity sign
  (4) fallback "position"
  (5) inconclusive (would return None with fallback=None)

Show win-rate matrix separately for "high-confidence" rallies (decided by
velocity, the most reliable available signal without info) vs "fallback-only"
rallies. If the same-skill 95% bias exists only in the fallback subset, the
problem is the position fallback, not the env's true physics.

Run:
    PYTHONPATH=. python nash_skills/v2/diag_winner_sources.py \\
        --rallies data/rallies_v2_1000.pkl
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pickle as pkl
from collections import defaultdict

import numpy as np

from nash_skills.skills import SKILL_NAMES, N_SKILLS

BALL_X_IDX = 36
BALL_VEL_X_IDX = 39
TABLE_SHIFT = 1.5
VELOCITY_EPS = 1e-6


def classify_winner_source(last_obs):
    """
    Return ('source_tag', inferred_winner_str_or_None) for the last obs.

    Tags:
      'velocity'  : ball still moving — velocity sign decides
      'fallback'  : ball stationary — only position fallback can decide
      'inconc'    : velocity is exactly zero AND ball at table_shift exactly (rare)
    """
    arr = np.asarray(last_obs)
    if len(arr) <= BALL_VEL_X_IDX:
        return "inconc", None

    ball_vel_x = float(arr[BALL_VEL_X_IDX])
    if abs(ball_vel_x) > VELOCITY_EPS:
        winner = "ego" if ball_vel_x > 0 else "opp"
        return "velocity", winner

    ball_x = float(arr[BALL_X_IDX])
    if ball_x > TABLE_SHIFT:
        return "fallback", "ego"
    if ball_x < TABLE_SHIFT:
        return "fallback", "opp"
    return "inconc", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rallies", required=True)
    args = ap.parse_args()

    print(f"Loading {args.rallies} ...")
    with open(args.rallies, "rb") as f:
        rallies = pkl.load(f)
    print(f"  Loaded {len(rallies)} rallies\n")

    # Per rally classification
    tally_by_source = defaultdict(lambda: {"ego": 0, "opp": 0, "inconc": 0})
    by_pair_source = defaultdict(lambda: defaultdict(lambda: {"ego": 0, "opp": 0}))

    for r in rallies:
        if r["winner"] == 0:
            continue  # skip truncated
        if not r["raw_obs"]:
            continue
        last_obs = r["raw_obs"][-1]
        source, winner = classify_winner_source(last_obs)
        if winner is None:
            tally_by_source[source]["inconc"] += 1
            continue
        # Use the rally's stored winner (which was determined at collection time)
        stored_winner = "ego" if r["winner"] == 1 else "opp"
        tally_by_source[source][stored_winner] += 1
        by_pair_source[source][(r["skill1"], r["skill2"])][stored_winner] += 1

    # Print source counts
    print("=" * 78)
    print("  WINNER-SOURCE BREAKDOWN")
    print("=" * 78)
    print(f"  {'source':<12} {'ego':>8} {'opp':>8} {'inconc':>8}  {'total':>8}")
    print("  " + "-" * 56)
    for src in ("velocity", "fallback", "inconc"):
        t = tally_by_source[src]
        total = t["ego"] + t["opp"] + t["inconc"]
        print(f"  {src:<12} {t['ego']:>8d} {t['opp']:>8d} {t['inconc']:>8d}  {total:>8d}")

    # Per-skill-pair win rate, separated by source
    for src in ("velocity", "fallback"):
        print(f"\n" + "=" * 78)
        print(f"  EGO WIN RATE — only rallies decided by '{src}' source")
        print("=" * 78)
        skill_w = 14
        col_w = 12
        header = " " * skill_w + "".join(f"{s[:9]:>{col_w}}" for s in SKILL_NAMES)
        print(header)
        print("-" * len(header))
        for i, s1 in enumerate(SKILL_NAMES):
            row = f"{s1:<{skill_w}}"
            for j, s2 in enumerate(SKILL_NAMES):
                t = by_pair_source[src][(s1, s2)]
                total = t["ego"] + t["opp"]
                if total == 0:
                    row += f"{'---':>{col_w}}"
                else:
                    wr = t["ego"] / total
                    row += f"{wr:>{col_w-3}.0%} ({total:>3d})"
            print(row)
        print("  (each cell: win_rate (n_done))")

    # Specific same-skill diagonal analysis
    print(f"\n" + "=" * 78)
    print(f"  SAME-SKILL DIAGONAL  (ego == opp skill)")
    print(f"  If bias is from 'fallback', diagonal should be ≈ 50% for 'velocity' source.")
    print("=" * 78)
    print(f"  {'skill':<14} {'src':<10} {'ego':>6} {'opp':>6} {'win_rate':>10}")
    for s in SKILL_NAMES:
        for src in ("velocity", "fallback"):
            t = by_pair_source[src][(s, s)]
            total = t["ego"] + t["opp"]
            wr_str = f"{t['ego']/total:.0%}" if total > 0 else "---"
            print(f"  {s:<14} {src:<10} {t['ego']:>6d} {t['opp']:>6d} {wr_str:>10}")


if __name__ == "__main__":
    main()
