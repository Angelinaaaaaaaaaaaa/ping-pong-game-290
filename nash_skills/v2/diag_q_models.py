"""
Sanity-check Q-value models trained by train_q_model_5skill_v3*.py.

Loads model1 (ego Q) and model2 (opp Q), evaluates them on real rally states,
and reports:

  1. Output range — are the Q values non-degenerate (not all ~0)?
  2. Skill discrimination — does Q change when you flip skills?
  3. Zero-sum sanity — is Q1(s, k1, k2) ≈ -Q2(s, k1, k2)?
  4. Empirical target match — does mean Q match the empirical return baseline?
  5. Per-skill bias — is Q systematically higher for some skills regardless of state?

Run:
    PYTHONPATH=. python nash_skills/v2/diag_q_models.py \
        --model1 models/model1_5skill_v3.pth \
        --model2 models/model2_5skill_v3.pth \
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
    first_weight_key = next(k for k in state_dict
                             if k.endswith(".weight") and state_dict[k].ndim == 2)
    state_dim = state_dict[first_weight_key].shape[1]
    m = SimpleModel(state_dim, [64, 32, 16], 1)
    m.load_state_dict(state_dict)
    m.eval()
    return m, state_dim


def q_all_pairs(model, state_np, state_dim):
    """Return (N_SKILLS, N_SKILLS) array of Q values for all (k1, k2) pairs."""
    out = np.zeros((N_SKILLS, N_SKILLS), dtype=np.float32)
    batch = []
    for i in range(N_SKILLS):
        for j in range(N_SKILLS):
            s = state_np.copy()
            s[-2] = i / (N_SKILLS - 1)
            s[-1] = j / (N_SKILLS - 1)
            batch.append(s)
    x = torch.tensor(np.array(batch), dtype=torch.float32)
    with torch.no_grad():
        vals = model(x).squeeze(1).numpy()
    return vals.reshape(N_SKILLS, N_SKILLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model1", default="models/model1_5skill_v3.pth")
    ap.add_argument("--model2", default="models/model2_5skill_v3.pth")
    ap.add_argument("--rallies", default="data/rallies_v2_1000.pkl")
    ap.add_argument("--n-states", type=int, default=200)
    args = ap.parse_args()

    print(f"Loading model1: {args.model1}")
    m1, dim1 = auto_load(args.model1)
    print(f"  state_dim = {dim1}")
    print(f"Loading model2: {args.model2}")
    m2, dim2 = auto_load(args.model2)
    print(f"  state_dim = {dim2}")
    assert dim1 == dim2, "model1 and model2 have different state_dim"

    print(f"\nLoading rallies: {args.rallies}")
    with open(args.rallies, "rb") as f:
        rallies = pkl.load(f)
    all_states = [s for r in rallies for s in r["states"]]
    print(f"  total states = {len(all_states)}")

    rng = np.random.default_rng(0)
    chosen = rng.choice(len(all_states), size=min(args.n_states, len(all_states)), replace=False)
    states = [all_states[i] for i in chosen]

    # Compute Q1 and Q2 over all skill pairs on each sampled state
    q1_table = np.stack([q_all_pairs(m1, s, dim1) for s in states])   # (N, 5, 5)
    q2_table = np.stack([q_all_pairs(m2, s, dim2) for s in states])

    print("\n" + "=" * 72)
    print("  Q-VALUE DIAGNOSTIC SUMMARY")
    print("=" * 72)

    # 1. Output range
    print("\n[1] Output range")
    print(f"  Q1: min={q1_table.min():+.4f}  max={q1_table.max():+.4f}  "
          f"mean={q1_table.mean():+.4f}  std={q1_table.std():.4f}")
    print(f"  Q2: min={q2_table.min():+.4f}  max={q2_table.max():+.4f}  "
          f"mean={q2_table.mean():+.4f}  std={q2_table.std():.4f}")
    if q1_table.std() < 0.01:
        print("  ⚠️  Q1 std < 0.01 — value function nearly constant (degenerate)")

    # 2. Skill discrimination — how much does Q change when we change skills?
    #    For each state, range over (k1, k2) pairs.
    range_per_state_q1 = q1_table.max(axis=(1, 2)) - q1_table.min(axis=(1, 2))
    range_per_state_q2 = q2_table.max(axis=(1, 2)) - q2_table.min(axis=(1, 2))
    print("\n[2] Skill discrimination (per-state range over 25 skill pairs)")
    print(f"  Q1 range: mean={range_per_state_q1.mean():.4f}  median={np.median(range_per_state_q1):.4f}")
    print(f"  Q2 range: mean={range_per_state_q2.mean():.4f}  median={np.median(range_per_state_q2):.4f}")
    if range_per_state_q1.mean() < 0.02:
        print("  ⚠️  Q1 barely changes across skill pairs — skill signal lost")

    # 3. Zero-sum check
    print("\n[3] Zero-sum sanity: corr(Q1, -Q2) should be high")
    flat1 = q1_table.flatten()
    flat2 = q2_table.flatten()
    corr = np.corrcoef(flat1, -flat2)[0, 1]
    print(f"  Pearson r(Q1, -Q2) = {corr:+.4f}")
    if corr < 0.5:
        print("  ⚠️  Low correlation — Q1 and Q2 disagree about who wins")

    sum_q = q1_table + q2_table
    print(f"  (Q1 + Q2): mean={sum_q.mean():+.4f}  std={sum_q.std():.4f}  "
          f"(should be ~0 for true zero-sum)")

    # 4. Empirical return baseline
    print("\n[4] Empirical baseline")
    n_ego_wins = sum(1 for r in rallies if r.get("winner") == 1)
    n_opp_wins = sum(1 for r in rallies if r.get("winner") == 2)
    n_trunc    = sum(1 for r in rallies if r.get("winner") == 0)
    total = len(rallies)
    print(f"  Rallies: ego_wins={n_ego_wins} ({n_ego_wins/total:.1%})  "
          f"opp_wins={n_opp_wins} ({n_opp_wins/total:.1%})  "
          f"truncated={n_trunc} ({n_trunc/total:.1%})")
    if n_ego_wins / total < 0.4 or n_ego_wins / total > 0.6:
        print(f"  ⚠️  Ego/opp asymmetry in data: ego wins {n_ego_wins/total:.1%} of rallies "
              "(should be ~50% under symmetric self-play)")

    # 5. Per-skill marginal bias (averaging out opp_skill)
    print("\n[5] Per-skill marginal Q (mean over states & opp_skill)")
    print(f"  {'skill':<14} {'Q1 (ego picks)':>16} {'Q2 (opp picks)':>16}")
    # For Q1: avg over (states, opp_skill) at each ego_skill
    q1_per_ego = q1_table.mean(axis=(0, 2))   # (N_SKILLS,)
    # For Q2: avg over (states, ego_skill) at each opp_skill
    q2_per_opp = q2_table.mean(axis=(0, 1))   # (N_SKILLS,)
    for i, name in enumerate(SKILL_NAMES):
        print(f"  {name:<14} {q1_per_ego[i]:>+16.4f} {q2_per_opp[i]:>+16.4f}")
    spread_q1 = q1_per_ego.max() - q1_per_ego.min()
    print(f"  Q1 marginal spread = {spread_q1:.4f}  "
          f"({'flat — skill irrelevant' if spread_q1 < 0.02 else 'OK'})")

    # 6. Sample Q1 surface (one state)
    print(f"\n[6] Sample Q1 surface (state[0])")
    surf = q1_table[0]
    skill_w = 12
    hd = 'ego \\\\ opp'
    header = f"  {hd:<14}" + "".join(f"{s[:8]:>{skill_w}}" for s in SKILL_NAMES)
    print(header)
    for i, name in enumerate(SKILL_NAMES):
        row = f"  {name:<14}"
        for j in range(N_SKILLS):
            row += f"{surf[i, j]:>+{skill_w}.4f}"
        print(row)

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
