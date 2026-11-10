"""
Why does the paper's online phase (Eq. 12) lose to some fixed strategies? On held-out decision
states, compare the learned zero-sum game -- player 1 gets M = (Q1 - Q2) / 2 over the 15 x 15
profiles of fixed (target, preparation) continuation policies -- with the profile Eq. 12 plays
(the argmax of Phi):

  pure_saddle          M has a pure equilibrium (max_i min_j M == min_j max_i M)
  support1/2           number of pairs in each player's equilibrium mix (linear program)
  randomization_gain   equilibrium value minus the best pure guarantee, per player
  phi_gain1/2          what a unilateral switch away from the Eq. 12 profile gains in M
  phi_nashconv         their sum: 0 iff the Eq. 12 profile is an equilibrium of the learned game

plus how often each live choice (the attacker's target, the defender's preparation) is made by
Eq. 12 and by the learned game's equilibrium. Every potential game has a pure equilibrium (the
potential's maximizer); where the learned game has none, no potential can represent it and the
argmax profile is open to a fixed counter-strategy.

    PYTHONPATH=. python scripts/diagnose_potential_equilibrium.py --rallies runs_paper/aggressive/rallies_10x.pkl \\
        --model-dir runs_paper/paper_alpha/aggressive/phi/gamma0.99_s0 --seed 0 --states 600 --output out.json
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import pickle

import numpy as np

SADDLE_TOL = 1e-6
ATTACKER_P1_SLOT = 70  # joint state [70:72]: attacker one-hot, slot 70 = player 1 attacks


def pure_security(M):
    """(player 1's best pure guarantee, the lowest cap player 2 can enforce with a pure choice)."""
    M = np.asarray(M, dtype=float)
    return float(M.min(axis=1).max()), float(M.max(axis=0).min())


def profile_gains(M, i, j):
    """Unilateral improvements from profile (i, j) in the zero-sum game M (player 1 maximizes)."""
    M = np.asarray(M, dtype=float)
    return float(M[:, j].max() - M[i, j]), float(M[i, j] - M[i, :].min())


def analyze_state(q1, q2, phi, support_tol=1e-3):
    from nash_skills.v2.matrix_game import solve_zero_sum
    M = (np.asarray(q1, dtype=float) - np.asarray(q2, dtype=float)) / 2.0
    lower, upper = pure_security(M)
    p, q, value = solve_zero_sum(M)
    i, j = np.unravel_index(int(np.argmax(phi)), np.shape(phi))
    g1, g2 = profile_gains(M, i, j)
    return {"pure_saddle": bool(upper - lower <= SADDLE_TOL), "support1": int((p > support_tol).sum()),
            "support2": int((q > support_tol).sum()), "value": float(value),
            "randomization_gain1": float(value - lower), "randomization_gain2": float(upper - value),
            "phi_i": int(i), "phi_j": int(j), "phi_gain1": g1, "phi_gain2": g2, "phi_nashconv": g1 + g2,
            "p": p.tolist(), "q": q.tolist()}


def summarize(rows):
    def mean(key):
        return float(np.mean([r[key] for r in rows]))
    return {"n_states": len(rows), "pure_saddle_share": mean("pure_saddle"),
            "support1_mean": mean("support1"), "support2_mean": mean("support2"),
            "randomization_gain1_mean": mean("randomization_gain1"),
            "randomization_gain2_mean": mean("randomization_gain2"),
            "phi_gain1_mean": mean("phi_gain1"), "phi_gain2_mean": mean("phi_gain2"),
            "phi_nashconv_mean": mean("phi_nashconv"),
            "phi_nashconv_median": float(np.median([r["phi_nashconv"] for r in rows]))}


def live_choices(state, row):
    """The live components at this state: the attacker's target and the defender's preparation,
    as played by Eq. 12 and as weighted by the learned game's equilibrium mixes."""
    from nash_skills.defend_positions import DEFEND_POSITION_NAMES
    from nash_skills.skills import SKILL_NAMES
    shape = (len(SKILL_NAMES), len(DEFEND_POSITION_NAMES))
    p1_attacks = state[ATTACKER_P1_SLOT] > 0.5
    att, dfd = (row["phi_i"], row["phi_j"]) if p1_attacks else (row["phi_j"], row["phi_i"])
    mix_att, mix_dfd = (row["p"], row["q"]) if p1_attacks else (row["q"], row["p"])
    return {"eq12_target": int(np.unravel_index(att, shape)[0]), "eq12_prep": int(np.unravel_index(dfd, shape)[1]),
            "eq_target": np.reshape(mix_att, shape).sum(axis=1), "eq_prep": np.reshape(mix_dfd, shape).sum(axis=0)}


def choice_summary(live):
    from nash_skills.defend_positions import DEFEND_POSITION_NAMES
    from nash_skills.skills import SKILL_NAMES
    n = len(live)

    def freq(key, names):
        counts = np.bincount([c[key] for c in live], minlength=len(names)) / n
        return dict(zip(names, map(float, counts)))
    return {"eq12_attacker_target": freq("eq12_target", SKILL_NAMES),
            "eq12_defender_preparation": freq("eq12_prep", DEFEND_POSITION_NAMES),
            "equilibrium_attacker_target": dict(zip(SKILL_NAMES, map(float, np.mean([c["eq_target"] for c in live], axis=0)))),
            "equilibrium_defender_preparation": dict(zip(DEFEND_POSITION_NAMES,
                                                         map(float, np.mean([c["eq_prep"] for c in live], axis=0))))}


def load_heldout_states(rally_path, seed, holdout_frac):
    """Decision states of the held-out fixed-pair rallies, split exactly as the trainer did."""
    from nash_skills.v2.train_q_model_5skill_v3_attack_defend import split_rallies
    with open(rally_path, "rb") as f:
        rallies = pickle.load(f)
    rallies = [r for r in rallies if r.get("theta_mode") != "mixed"]
    _train, holdout = split_rallies(rallies, holdout_frac, seed)
    return [s for r in holdout for s in r["states"]]


def main():
    from eval_table1_attack_defend import _load_models
    from nash_skills.table_rules import joint_value_table
    from nash_skills.v2.train_q_model_5skill_v3_attack_defend import DEFAULT_HOLDOUT_FRAC
    ap = argparse.ArgumentParser()
    ap.add_argument("--rallies", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seed", type=int, default=0, help="the training seed (fixes the held-out split)")
    ap.add_argument("--states", type=int, default=600)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    models = _load_models(args.model_dir)
    states = load_heldout_states(args.rallies, args.seed, DEFAULT_HOLDOUT_FRAC)
    rng = np.random.default_rng(args.seed)
    rows, live = [], []
    for k in rng.choice(len(states), size=min(args.states, len(states)), replace=False):
        x = np.asarray(states[k], dtype=np.float32)
        row = analyze_state(*(joint_value_table(x, models[m]) for m in ("model1", "model2", "model_p")))
        rows.append(row)
        live.append(live_choices(x, row))
    summary = {**summarize(rows), **choice_summary(live)}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"args": vars(args), "summary": summary}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
