"""
Does the learned potential move the same way as the deviating player's own value?

For held-out states and random pure base pairs (a1, d1, a2, d2), change ONE half of ONE
player's choice (attack target or preparation position) and compare
    dPhi = Phi(x, new) - Phi(x, old)   with   dQ_i = Q_i(x, new) - Q_i(x, old).
An alpha-potential should satisfy dPhi ~ dQ_i, so at least the signs should match.
Reported by deviating player, by changed half, and by the deviator's role at that state
(attacker = it receives the ball this exchange; read from the state's attacker one-hot).

    PYTHONPATH=. python scripts/phi_sign_agreement.py --rallies R.pkl --model-dir PHI_DIR --seed 0
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json

import numpy as np
import torch

from nash_skills.defend_positions import N_DEFEND_POSITIONS
from nash_skills.skills import N_SKILLS
from nash_skills.v2.joint_action import joint_action_index, write_joint_pair_by_index

ATTACKER_SLOT = 70  # [70:72] one-hot: index 0 -> player 1 attacks


def sign_agreement(dphi, dq, eps=1e-3):
    dphi, dq = np.asarray(dphi, float), np.asarray(dq, float)
    keep = np.abs(dq) > eps
    n = int(keep.sum())
    agree = float(np.mean(np.sign(dphi[keep]) == np.sign(dq[keep]))) if n else float("nan")
    corr = float(np.corrcoef(dphi[keep], dq[keep])[0, 1]) if n > 2 else float("nan")
    return {"agree": agree, "n": n, "corr": corr, "mean_abs_dq": float(np.abs(dq).mean()),
            "mean_abs_dphi": float(np.abs(dphi).mean())}


def deviation_diffs(model_p, model1, model2, x, player, half, rng):
    """One random pure base pair and one single-half deviation of `player` per state."""
    n = len(x)
    a = rng.integers(N_SKILLS, size=(n, 2))
    d = rng.integers(N_DEFEND_POSITIONS, size=(n, 2))
    a2, d2 = a.copy(), d.copy()
    k = player - 1
    if half == "attack":
        a2[:, k] = (a[:, k] + rng.integers(1, N_SKILLS, size=n)) % N_SKILLS
    else:
        d2[:, k] = (d[:, k] + rng.integers(1, N_DEFEND_POSITIONS, size=n)) % N_DEFEND_POSITIONS
    old = torch.stack([write_joint_pair_by_index(x[i:i + 1], joint_action_index(a[i, 0], d[i, 0]),
                                                 joint_action_index(a[i, 1], d[i, 1]))[0] for i in range(n)])
    new = torch.stack([write_joint_pair_by_index(x[i:i + 1], joint_action_index(a2[i, 0], d2[i, 0]),
                                                 joint_action_index(a2[i, 1], d2[i, 1]))[0] for i in range(n)])
    q = model1 if player == 1 else model2
    with torch.no_grad():
        dphi = (model_p(new) - model_p(old)).reshape(-1).numpy()
        dq = (q(new) - q(old)).reshape(-1).numpy()
    return dphi, dq


def main():
    from model_arch import SimpleModel
    from nash_skills.v2.state_encoder_attack_defend import STATE_DIM_ATTACK_DEFEND as D
    from nash_skills.v2.train_q_model_5skill_v3_attack_defend import build_dataset, split_rallies
    import pickle
    ap = argparse.ArgumentParser()
    ap.add_argument("--rallies", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-states", type=int, default=4000)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    with open(os.path.join(args.model_dir, "heldout_attack_defend.json")) as f:
        report = json.load(f)
    models = {"p": SimpleModel(D, [64, 32, 16], 1, last_layer_activation=None),
              "1": SimpleModel(D, [64, 32, 16], 1), "2": SimpleModel(D, [64, 32, 16], 1)}
    for key, name in (("p", "model_p"), ("1", "model1"), ("2", "model2")):
        models[key].load_state_dict(torch.load(os.path.join(args.model_dir, f"{name}_attack_defend.pth"),
                                               map_location="cpu"))
        models[key].eval()
    with open(args.rallies, "rb") as f:
        rallies = pickle.load(f)
    _train, hold = split_rallies(rallies, 0.2, report["seed"])
    x, _, _ = build_dataset(hold, gamma=report["gamma"])
    rng = np.random.default_rng(args.seed)
    x = x[rng.permutation(len(x))[:args.n_states]]
    p1_attacks = x[:, ATTACKER_SLOT] > 0.5
    out = {}
    for player in (1, 2):
        attacker = p1_attacks if player == 1 else ~p1_attacks
        for half in ("attack", "defend"):
            dphi, dq = deviation_diffs(models["p"], models["1"], models["2"], x, player, half, rng)
            for role, mask in (("attacker", attacker), ("defender", ~attacker), ("all", torch.ones_like(attacker))):
                m = mask.numpy()
                out[f"P{player}/{half}/{role}"] = sign_agreement(dphi[m], dq[m])
    for k, v in out.items():
        print(f"{k:<22} sign agreement {v['agree']:.3f}  corr {v['corr']:+.3f}  n={v['n']:<5} "
              f"mean|dQ| {v['mean_abs_dq']:.3f}  mean|dPhi| {v['mean_abs_dphi']:.3f}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
