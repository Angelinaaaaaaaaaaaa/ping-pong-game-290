"""
Check whether 'left'/'left_short' (or 'right'/'right_short') are actually
distinguishable in the collected data, i.e. does PPO produce different
physical trajectories for them?

Compare the terminal ball state (position + velocity) aggregated across all
rallies of each skill, regardless of opponent. If left and left_short produce
near-identical terminal ball distributions, PPO can't actually execute the
"short" variant — the two skills are functionally duplicates.

Run:
    PYTHONPATH=. python nash_skills/v2/diag_skill_redundancy.py \\
        --rallies data/rallies_v2_1000.pkl
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pickle as pkl

import numpy as np

from nash_skills.skills import SKILL_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rallies", required=True)
    args = ap.parse_args()

    print(f"Loading {args.rallies} ...")
    with open(args.rallies, "rb") as f:
        rallies = pkl.load(f)
    print(f"  Loaded {len(rallies)} rallies\n")

    # For each ego skill, aggregate terminal ball state across all rallies
    print("=" * 78)
    print("  TERMINAL BALL STATE  (per ego skill, aggregated across all opp)")
    print("=" * 78)
    print(f"  {'ego_skill':<14} {'n':>6} {'ball_x':>16} {'ball_y':>16} {'ball_vx':>16}")
    print("  " + "-" * 70)

    per_skill_stats = {}
    for s in SKILL_NAMES:
        # Use the LAST raw_obs of each rally as the terminal state
        terminals = [r["raw_obs"][-1] for r in rallies if r["skill1"] == s and r["raw_obs"]]
        if not terminals:
            print(f"  {s:<14} (no data)")
            continue
        terminals = np.stack(terminals)
        bx = terminals[:, 36]
        by = terminals[:, 37]
        bvx = terminals[:, 39]
        per_skill_stats[s] = (bx, by, bvx)
        print(f"  {s:<14} {len(bx):>6d}  "
              f"{bx.mean():>+8.3f} ± {bx.std():>5.3f}   "
              f"{by.mean():>+8.3f} ± {by.std():>5.3f}   "
              f"{bvx.mean():>+8.3f} ± {bvx.std():>5.3f}")

    # ────────────────────────────────────────────────────────────
    # Pairwise comparison: which pairs of skills are nearly identical?
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  PAIRWISE TERMINAL-STATE DISTANCE")
    print("  (|mean_diff| in ball_x / ball_vx between two ego skills)")
    print("  Small distance → skills produce same physical trajectory → REDUNDANT")
    print("=" * 78)
    print(f"  {'skill_A':<14} {'skill_B':<14} {'|Δ ball_x|':>14} {'|Δ ball_vx|':>14}")
    print("  " + "-" * 60)

    skills = list(per_skill_stats.keys())
    for i, sa in enumerate(skills):
        for sb in skills[i+1:]:
            bxa, _, bvxa = per_skill_stats[sa]
            bxb, _, bvxb = per_skill_stats[sb]
            dx = abs(bxa.mean() - bxb.mean())
            dvx = abs(bvxa.mean() - bvxb.mean())
            marker = ""
            if dx < 0.10 and dvx < 0.50:
                marker = "  ← NEAR-IDENTICAL"
            elif dx < 0.20:
                marker = "  ← similar"
            print(f"  {sa:<14} {sb:<14} {dx:>14.3f} {dvx:>14.3f}{marker}")

    # ────────────────────────────────────────────────────────────
    # Same-state check: do same-skill rallies have same trajectory across opp?
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  DESIGN CHECK: skills.py declared targets vs observed terminal ball_x")
    print("=" * 78)
    print(f"  {'skill':<14} {'declared x_target':>20} {'observed terminal_x':>22}")
    print("  " + "-" * 60)
    declared = {
        "left":         2.19,
        "left_short":   1.75,
        "center_safe":  1.85,
        "right_short":  1.75,
        "right":        2.19,
    }
    for s in SKILL_NAMES:
        if s not in per_skill_stats:
            continue
        bx, _, _ = per_skill_stats[s]
        d = declared.get(s, float("nan"))
        diff = bx.mean() - d
        marker = "  ⚠️ PPO undershoots/overshoots" if abs(diff) > 0.15 else "  ✓ close"
        print(f"  {s:<14} {d:>20.2f} {bx.mean():>22.3f}  (Δ={diff:+.2f}){marker}")


if __name__ == "__main__":
    main()
