"""
Diagnostic: fix opp_skill, look at Q1(s, ego_skill, opp_fixed) patterns.

For each opp_skill ∈ {left, left_short, center_safe, right_short, right}:
  - Compute Q1(s, k_ego, opp_fixed) for all k_ego across N sampled states.
  - Report distribution of argmax_ego across those states.
  - Report mean & std of Q1 per ego_skill.

A sensible pattern would show ego best-responding to opp's choice (e.g. when
opp plays "left" the model prefers a different ego skill). A degenerate
pattern (e.g. ego always picks "left" regardless of opp) is evidence that
collect_data produced biased or under-informative training data.

Run:
    PYTHONPATH=. python nash_skills/v2/diag_q_fixed_opp.py \\
        --model1 models/model1_5skill_v3_1000.pth \\
        --rallies data/rallies_v2_1000.pkl
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pickle as pkl

import numpy as np
import torch

from model_arch import SimpleModel
from nash_skills.skills import SKILL_NAMES, N_SKILLS


def auto_load(path: str):
    state_dict = torch.load(path, weights_only=True, map_location="cpu")
    first_w = next(k for k in state_dict
                   if k.endswith(".weight") and state_dict[k].ndim == 2)
    state_dim = state_dict[first_w].shape[1]
    m = SimpleModel(state_dim, [64, 32, 16], 1)
    m.load_state_dict(state_dict)
    m.eval()
    return m, state_dim


def q_all_ego_for_fixed_opp(model, state_np, opp_idx):
    """Return (N_SKILLS,) array of Q1 values for ego ∈ 0..N_SKILLS-1, opp fixed."""
    batch = []
    for i in range(N_SKILLS):
        s = state_np.copy()
        s[-2] = i / (N_SKILLS - 1)
        s[-1] = opp_idx / (N_SKILLS - 1)
        batch.append(s)
    x = torch.tensor(np.array(batch), dtype=torch.float32)
    with torch.no_grad():
        return model(x).squeeze(1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model1", default="models/model1_5skill_v3.pth")
    ap.add_argument("--rallies", default="data/rallies_v2_1000.pkl")
    ap.add_argument("--n-states", type=int, default=200)
    args = ap.parse_args()

    print(f"Loading model1: {args.model1}")
    m1, dim = auto_load(args.model1)
    print(f"  state_dim = {dim}")

    print(f"Loading rallies: {args.rallies}")
    with open(args.rallies, "rb") as f:
        rallies = pkl.load(f)
    all_states = [s for r in rallies for s in r["states"]]
    print(f"  total states = {len(all_states)}")

    rng = np.random.default_rng(0)
    chosen = rng.choice(len(all_states),
                        size=min(args.n_states, len(all_states)),
                        replace=False)
    states = [all_states[i] for i in chosen]

    print("\n" + "=" * 78)
    print("  FIXED-OPP Q DIAGNOSTIC")
    print("=" * 78)
    print(f"  Across {len(states)} sampled rally states, for each fixed opp_skill,")
    print(f"  we report:")
    print(f"    (a) argmax_ego distribution: which ego skill maxes Q1 most often")
    print(f"    (b) mean Q1 per ego_skill\n")

    # For each fixed opp skill
    for opp_idx, opp_name in enumerate(SKILL_NAMES):
        print(f"\n──── opp fixed to '{opp_name}' (idx={opp_idx}) ────")

        # Compute Q1 for all states, all ego
        q_per_state = np.zeros((len(states), N_SKILLS), dtype=np.float32)
        for i, s in enumerate(states):
            q_per_state[i] = q_all_ego_for_fixed_opp(m1, s, opp_idx)

        # (a) argmax distribution
        argmax = q_per_state.argmax(axis=1)
        counts = np.bincount(argmax, minlength=N_SKILLS)
        print("  argmax_ego distribution:")
        for k, name in enumerate(SKILL_NAMES):
            pct = counts[k] / len(states) * 100
            bar = "#" * int(pct / 2)
            star = "  ← opp" if k == opp_idx else ""
            print(f"    ego={name:<12}: {counts[k]:4d}/{len(states)} ({pct:5.1f}%) {bar}{star}")

        # (b) mean & std Q per ego
        print("  mean Q1 per ego (across all sampled states):")
        for k, name in enumerate(SKILL_NAMES):
            mu = q_per_state[:, k].mean()
            sd = q_per_state[:, k].std()
            print(f"    ego={name:<12}: mean={mu:+.4f}  std={sd:.4f}")

    # ────────────────────────────────────────────────────────────
    # Cross-table summary: argmax counts as 5x5 matrix
    print("\n" + "=" * 78)
    print("  SUMMARY: argmax distribution as 5x5 table")
    print("  rows = opp_skill (fixed), cols = ego_skill (argmax pct)")
    print("=" * 78)

    table = np.zeros((N_SKILLS, N_SKILLS), dtype=np.float32)
    for opp_idx in range(N_SKILLS):
        q_per_state = np.zeros((len(states), N_SKILLS), dtype=np.float32)
        for i, s in enumerate(states):
            q_per_state[i] = q_all_ego_for_fixed_opp(m1, s, opp_idx)
        counts = np.bincount(q_per_state.argmax(axis=1), minlength=N_SKILLS)
        table[opp_idx] = counts / len(states)

    col_w = 10
    oe = 'opp\\ego'
    print(f"  {oe:<12}" + "".join(f"{s[:8]:>{col_w}}" for s in SKILL_NAMES))
    print("  " + "-" * (12 + col_w * N_SKILLS))
    for opp_idx, opp_name in enumerate(SKILL_NAMES):
        row = f"  {opp_name:<12}"
        for ego_idx in range(N_SKILLS):
            pct = table[opp_idx, ego_idx] * 100
            row += f"{pct:>{col_w-2}.1f}%"
            row += "  "
        print(row)

    print("\n=== INTERPRETATION ===")
    print(" - If diagonal (opp==ego) is high → ego mirrors opp's skill")
    print(" - If a single COLUMN is high → ego always picks the same skill")
    print("   (degenerate; suggests under-trained Q or biased data)")
    print(" - Sensible Nash pattern: off-diagonal high (ego counters opp)")
    print(" - PPO/data bias often produces a 'always pick X' column")

    # ────────────────────────────────────────────────────────────
    # Automated pattern-quality verdict
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  PATTERN QUALITY METRICS")
    print("=" * 78)

    # 1. Max row-pair L1 distance — how different are any two opp rows?
    max_row_l1 = 0.0
    worst_pair = (None, None)
    for i in range(N_SKILLS):
        for j in range(i + 1, N_SKILLS):
            d = float(np.abs(table[i] - table[j]).sum() / 2)
            if d > max_row_l1:
                max_row_l1 = d
                worst_pair = (SKILL_NAMES[i], SKILL_NAMES[j])

    # 2. Max column spread — how much does a column vary across opp rows?
    col_max = table.max(axis=0)
    col_min = table.min(axis=0)
    col_spreads = col_max - col_min
    max_col_spread = float(col_spreads.max())
    dominant_col_idx = int(col_spreads.argmax())

    # 3. Off-diagonal vs diagonal mass
    diag_mass = float(np.diag(table).sum() / N_SKILLS)
    offdiag_mass = float((table.sum() - np.diag(table).sum()) / (N_SKILLS * (N_SKILLS - 1)))

    print(f"  Max row-pair L1 distance : {max_row_l1:.3f}  "
          f"(rows: {worst_pair[0]} ↔ {worst_pair[1]})")
    print(f"    →  > 0.30 = strong signal | 0.10-0.30 = weak | < 0.10 = ignored")
    print(f"  Max column spread        : {max_col_spread:.3f}  "
          f"(column: {SKILL_NAMES[dominant_col_idx]})")
    print(f"    →  > 0.30 = some col varies a lot | < 0.10 = all cols flat")
    print(f"  Mean diagonal cell       : {diag_mass:.3f}  (ego == opp)")
    print(f"  Mean off-diagonal cell   : {offdiag_mass:.3f}  (ego ≠ opp)")
    print(f"    →  diag >> offdiag means ego mirrors opp")
    print(f"    →  off >> diag means ego counters opp (Nash-style)")

    # Final verdict
    print(f"\n  VERDICT:")
    if max_row_l1 < 0.10 and max_col_spread < 0.10:
        print(f"    ❌ Q model IGNORES opp_skill (column-bias).")
        print(f"       All 5 opp choices produce nearly identical ego argmax distributions.")
        print(f"       Likely cause: state encoder lacks opp position info, OR data bias.")
    elif max_row_l1 > 0.30:
        print(f"    ✅ Q model USES opp_skill (clear conditional pattern).")
        if diag_mass > offdiag_mass + 0.05:
            print(f"       Pattern: ego mirrors opp (diagonal dominant).")
        elif offdiag_mass > diag_mass + 0.05:
            print(f"       Pattern: ego counters opp (off-diagonal dominant; Nash-like).")
    else:
        print(f"    ⚠️ Q model has WEAK conditional signal (L1={max_row_l1:.2f}).")
        print(f"       Some opp-dependence visible but far from sharp.")


if __name__ == "__main__":
    main()
