"""
Plot the empirical 5x5 skill-matchup win rate from collected rally data.

This is the RAW DATA truth: no model, no encoder, no training — just count
ego_wins / opp_wins per (skill1, skill2) pair across all collected rallies.

Useful for verifying meeting feedback: is the data biased / are some skills
inherently weak / does the PPO controller have hidden asymmetries?

Run:
    PYTHONPATH=. python nash_skills/v2/plot_skill_matchup.py \\
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rallies", required=True)
    ap.add_argument("--out-csv", default=None, help="Optional CSV output path")
    args = ap.parse_args()

    print(f"Loading {args.rallies} ...")
    with open(args.rallies, "rb") as f:
        rallies = pkl.load(f)
    print(f"  Loaded {len(rallies)} rallies\n")

    # Tally: (skill1, skill2) -> {ego_wins, opp_wins, trunc, total}
    tally = defaultdict(lambda: {"ego": 0, "opp": 0, "trunc": 0})
    for r in rallies:
        s1, s2 = r["skill1"], r["skill2"]
        w = r["winner"]
        if w == 1:   tally[(s1, s2)]["ego"]   += 1
        elif w == 2: tally[(s1, s2)]["opp"]   += 1
        else:        tally[(s1, s2)]["trunc"] += 1

    # Build win-rate matrix (ego wins / (ego+opp), excluding trunc)
    wr_matrix = np.full((N_SKILLS, N_SKILLS), np.nan, dtype=np.float32)
    done_matrix = np.zeros((N_SKILLS, N_SKILLS), dtype=np.int32)
    trunc_matrix = np.zeros((N_SKILLS, N_SKILLS), dtype=np.int32)
    total_matrix = np.zeros((N_SKILLS, N_SKILLS), dtype=np.int32)

    for i, s1 in enumerate(SKILL_NAMES):
        for j, s2 in enumerate(SKILL_NAMES):
            t = tally[(s1, s2)]
            done = t["ego"] + t["opp"]
            done_matrix[i, j] = done
            trunc_matrix[i, j] = t["trunc"]
            total_matrix[i, j] = done + t["trunc"]
            if done > 0:
                wr_matrix[i, j] = t["ego"] / done

    # ────────────────────────────────────────────────────────────
    # Print win-rate table (ego's perspective)
    # ────────────────────────────────────────────────────────────
    print("=" * 78)
    print("  EGO WIN RATE   (excluding truncated rallies)")
    print("  rows = ego skill (skill1)   cols = opp skill (skill2)")
    print("=" * 78)
    col_w = 12
    skill_w = 14
    header = " " * skill_w + "".join(f"{s[:9]:>{col_w}}" for s in SKILL_NAMES)
    print(header)
    print("-" * len(header))
    for i, s1 in enumerate(SKILL_NAMES):
        row = f"{s1:<{skill_w}}"
        for j in range(N_SKILLS):
            if np.isnan(wr_matrix[i, j]):
                row += f"{'---':>{col_w}}"
            else:
                row += f"{wr_matrix[i, j]:>{col_w}.0%}"
        print(row)

    # ────────────────────────────────────────────────────────────
    # Print done rallies count
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  DONE RALLIES PER CELL  (ego+opp wins; trunc not counted)")
    print("=" * 78)
    print(header)
    print("-" * len(header))
    for i, s1 in enumerate(SKILL_NAMES):
        row = f"{s1:<{skill_w}}"
        for j in range(N_SKILLS):
            row += f"{done_matrix[i, j]:>{col_w}d}"
        print(row)

    # ────────────────────────────────────────────────────────────
    # Print truncation rate per cell
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  TRUNCATION RATE  (truncated / total per cell)")
    print("=" * 78)
    print(header)
    print("-" * len(header))
    for i, s1 in enumerate(SKILL_NAMES):
        row = f"{s1:<{skill_w}}"
        for j in range(N_SKILLS):
            trunc_pct = trunc_matrix[i, j] / max(total_matrix[i, j], 1)
            row += f"{trunc_pct:>{col_w}.0%}"
        print(row)

    # ────────────────────────────────────────────────────────────
    # Row / column marginals
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  PER-SKILL MARGINAL WIN RATE")
    print("  - 'as ego' = avg win rate when ego plays this skill (across all opp choices)")
    print("  - 'as opp' = avg win rate against this opp skill (i.e. 1 - other's WR)")
    print("=" * 78)
    print(f"  {'skill':<14} {'as ego':>10} {'as opp':>10}  {'diff (ego-opp)':>16}")
    for k, s in enumerate(SKILL_NAMES):
        as_ego = np.nanmean(wr_matrix[k, :])
        as_opp = 1.0 - np.nanmean(wr_matrix[:, k])
        diff = as_ego - as_opp
        marker = "  ⭐ dominant" if diff > 0.10 else ("  ⚠️ weak" if diff < -0.10 else "")
        print(f"  {s:<14} {as_ego:>10.1%} {as_opp:>10.1%}  {diff:>+16.1%}{marker}")

    # Symmetry check
    print("\n" + "=" * 78)
    print("  SYMMETRY CHECK   (entry vs swap: wr(i,j) + wr(j,i) should ≈ 1.0)")
    print("=" * 78)
    asymm = []
    for i in range(N_SKILLS):
        for j in range(i, N_SKILLS):
            if np.isnan(wr_matrix[i, j]) or np.isnan(wr_matrix[j, i]):
                continue
            s = wr_matrix[i, j] + wr_matrix[j, i]
            asymm.append(abs(s - 1.0))
            if abs(s - 1.0) > 0.20:
                print(f"  ({SKILL_NAMES[i]:<12}, {SKILL_NAMES[j]:<12}):  "
                      f"wr_fwd={wr_matrix[i,j]:.1%}  wr_rev={wr_matrix[j,i]:.1%}  "
                      f"sum={s:.2f}  ← asymmetric ({(s-1.0)*100:+.0f}pp)")
    if asymm:
        print(f"\n  Mean |sum - 1.0| across all pairs: {np.mean(asymm):.3f}  "
              f"(0 = perfectly symmetric)")

    # Save CSV if requested
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) if os.path.dirname(args.out_csv) else ".",
                    exist_ok=True)
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ego_skill", "opp_skill", "ego_wins", "opp_wins", "trunc",
                        "win_rate", "trunc_rate"])
            for i, s1 in enumerate(SKILL_NAMES):
                for j, s2 in enumerate(SKILL_NAMES):
                    t = tally[(s1, s2)]
                    total = total_matrix[i, j]
                    w.writerow([
                        s1, s2, t["ego"], t["opp"], t["trunc"],
                        f"{wr_matrix[i,j]:.4f}" if not np.isnan(wr_matrix[i, j]) else "",
                        f"{trunc_matrix[i,j] / max(total, 1):.4f}",
                    ])
        print(f"\nSaved CSV to {args.out_csv}")


if __name__ == "__main__":
    main()
