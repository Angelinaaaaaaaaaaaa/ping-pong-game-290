"""Solve the attacker/defender payoff matrix and score policies against it.

Two quantities, both on the win-rate scale so they need no normalisation:

  LP equilibrium   the exact Nash mixed strategy of the matrix game, by linear
                   programming. For a symmetric zero-sum game its value is 0.5.

  Nash regret      how much a best-responding opponent gains against a policy:
                       regret(p) = max_j (opponent's win rate playing pure j) - 0.5
                   Zero exactly when p is an equilibrium. This matches the
                   quantity alpha-RACER reports (max_theta V(theta, theta*_-i)
                   minus V(theta*)), with one difference in our favour: our
                   action set is 15 pure strategies, so the max is enumerated
                   rather than approached by gradient ascent, and the regret is
                   exact rather than a lower bound.

    python scripts/solve_equilibrium.py --matrix logs/AD/payoff.npz \
        --policy logs/AD/selfplay_ad.json --policy logs/AD/selfplay_ad_det.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.optimize import linprog

from nash_skills.skills import SKILL_NAMES, N_SKILLS
from nash_skills.defender import DEFENDER_NAMES, N_DEFENDER, split_joint


def solve_zero_sum(A):
    """Maximin mixed strategy for the row player of a zero-sum game.

    Maximise v subject to (A^T p)_j >= v for every column j, with p a
    distribution. Variables are [p_0 .. p_{n-1}, v]; linprog minimises, so the
    objective is -v and the column constraints are negated into <= form.
    """
    n = A.shape[0]
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_ub = np.hstack([-A.T, np.ones((n, 1))])      # v - (A^T p)_j <= 0
    b_ub = np.zeros(n)
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = [1.0]
    bounds = [(0.0, 1.0)] * n + [(None, None)]
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                method="highs")
    if not r.success:
        raise RuntimeError(f"LP failed: {r.message}")
    return r.x[:n], float(r.x[-1])


def nash_regret(A, p):
    """Best-response gain against a policy, on the win-rate scale.

    A[i][j] is the row player's win rate for row i against column j, so a row
    player using p gives the column player playing pure j a win rate of
    1 - (p^T A)_j. Note the orientation: `A @ p` would index rows, not columns,
    and silently returns the row player's own payoff instead -- which made the
    LP solution look like it had regret 0.47 instead of 0.
    """
    opp = 1.0 - p @ A
    j = int(np.argmax(opp))
    return float(opp[j] - 0.5), j


def fold(p):
    """Joint distribution -> (over skills, over defender positions)."""
    s = [sum(p[d * N_SKILLS + k] for d in range(N_DEFENDER)) for k in range(N_SKILLS)]
    d = [sum(p[d * N_SKILLS + k] for k in range(N_SKILLS)) for d in range(N_DEFENDER)]
    return np.array(s), np.array(d)


def joint_name(i):
    k, d = split_joint(i, N_SKILLS)
    return f"{SKILL_NAMES[k]}+{DEFENDER_NAMES[d][4:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="logs/AD/payoff.npz")
    ap.add_argument("--policy", action="append", default=[],
                    help="eval JSON whose skill_pct gives the policy's mixture")
    args = ap.parse_args()

    z = np.load(args.matrix)
    A = z["A"]
    n = A.shape[0]
    asym = float(np.max(np.abs(A + A.T - 1.0)))
    print(f"matrix {n}x{n}, {int(z['rallies'])} rallies per cell, "
          f"antisymmetry error {asym:.3g}\n")

    p_star, v = solve_zero_sum(A)
    r_star, _ = nash_regret(A, p_star)
    print(f"LP equilibrium: value {v:.4f}   (0.5 for a symmetric game)")
    print(f"  its own Nash regret {r_star:+.4f}   <- a self-check; non-zero means"
          " the matrix is too noisy")
    s, d = fold(p_star)
    print("  over skills   : " + "  ".join(f"{SKILL_NAMES[i]} {s[i]:.3f}" for i in range(N_SKILLS)))
    print("  over positions: " + "  ".join(f"{DEFENDER_NAMES[i][4:]} {d[i]:.3f}" for i in range(N_DEFENDER)))
    top = np.argsort(-p_star)[:5]
    print("  largest pure weights: " + "  ".join(f"{joint_name(i)} {p_star[i]:.3f}" for i in top if p_star[i] > 1e-4))

    print(f"\n{'policy':<26}{'Nash regret':>13}{'best response':>26}{'TV from LP eq':>15}")
    for f in args.policy:
        j = json.load(open(f))
        pct = np.mean([o["skill_pct"] for o in j["opponents"]], axis=0)
        pct = np.asarray(pct, dtype=float)
        pct = pct / pct.sum()
        reg, br = nash_regret(A, pct)
        tv = 0.5 * np.abs(pct - p_star).sum()
        label = f.split("/")[-1].replace(".json", "")
        print(f"{label:<26}{reg:>+13.4f}{joint_name(br):>26}{tv:>15.3f}")

    print("\nuniform reference:")
    u = np.full(n, 1.0 / n)
    reg, br = nash_regret(A, u)
    print(f"{'uniform over 15':<26}{reg:>+13.4f}{joint_name(br):>26}"
          f"{0.5 * np.abs(u - p_star).sum():>15.3f}")


if __name__ == "__main__":
    main()
