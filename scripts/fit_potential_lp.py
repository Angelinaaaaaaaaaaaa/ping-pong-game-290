"""
Kexin's V3 way of fitting Phi (v3/generate_potential_labels.py), on the Design B joint game: solve paper Eq. 11
exactly at every training state as a linear program over the 15 x 15 joint profiles (nash_skills.v2.
potential_lp_joint), using the learned value networks Q1, Q2 of an existing model directory, then train the Phi
network to reproduce those per-state potentials by squared error. Writes a model directory that the evaluators
accept: the same Q1, Q2 and settings, with the new Phi and its held-out residuals.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/fit_potential_lp.py --rallies runs_v3/data/rallies.pkl \\
        --model-dir runs_v3/paper/aggressive/phi/gamma0.99_s0 --seed 0 --out-dir runs_v3/paper_lp/aggressive/phi/gamma0.99_s0
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import multiprocessing as mp
import pickle
import shutil
import time

import numpy as np
import torch

from nash_skills.defend_positions import N_DEFEND_POSITIONS
from nash_skills.skills import N_SKILLS
from nash_skills.v2.joint_action import N_JOINT_ACTIONS

_EGO_ATTACK, _OPP_ATTACK = 74, 74 + N_SKILLS
_EGO_DEFEND, _OPP_DEFEND = 74 + 2 * N_SKILLS, 74 + 2 * N_SKILLS + N_DEFEND_POSITIONS
_CHOICES = slice(74, _OPP_DEFEND + N_DEFEND_POSITIONS)


def with_profiles(x, i, j):
    """Copy of the states x (B, D) with player 1 on joint pair i[b] and player 2 on j[b] in row b."""
    out = x.clone()
    rows = torch.arange(len(x))
    out[:, _CHOICES] = 0.0
    out[rows, _EGO_ATTACK + i // N_DEFEND_POSITIONS] = 1.0
    out[rows, _OPP_ATTACK + j // N_DEFEND_POSITIONS] = 1.0
    out[rows, _EGO_DEFEND + i % N_DEFEND_POSITIONS] = 1.0
    out[rows, _OPP_DEFEND + j % N_DEFEND_POSITIONS] = 1.0
    return out


def value_tables(model1, model2, x, chunk=256):
    """Q1, Q2 over all 15 x 15 joint profiles at every state: two (n_states, 15, 15) arrays."""
    n = N_JOINT_ACTIONS
    i = torch.arange(n).repeat_interleave(n)
    j = torch.arange(n).repeat(n)
    q1, q2 = [], []
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            block = x[start:start + chunk]
            rows = with_profiles(block.repeat_interleave(n * n, dim=0), i.repeat(len(block)), j.repeat(len(block)))
            q1.append(model1(rows).reshape(len(block), n, n).numpy())
            q2.append(model2(rows).reshape(len(block), n, n).numpy())
    return np.concatenate(q1), np.concatenate(q2)


def _solve(args):
    from nash_skills.v2.potential_lp_joint import solve_potential_lp
    return solve_potential_lp(*args)


def lp_labels(q1, q2, workers=8):
    """Per-state LP potentials (n_states, 15, 15) with Phi[0, 0] = 0, and each state's smallest alpha."""
    pairs = list(zip(q1, q2))
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            out = pool.map(_solve, pairs, chunksize=256)
    else:
        out = [_solve(p) for p in pairs]
    return np.stack([phi for phi, _a in out]), np.array([a for _p, a in out])


def regress_phi(model1, x_train, phi_train, x_val, phi_val, epochs, lr, batch, profiles_per_state, seed):
    """Phi network fit to the per-state LP potentials by squared error; the epoch with the lowest validation
    error is kept. Input normalisation is copied from Q1 and frozen, as in the alpha-potential trainer."""
    from model_arch import SimpleModel
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SimpleModel(x_train.shape[1], [64, 32, 16], 1, last_layer_activation=None)
    model.batch_norm.running_mean = model1.batch_norm.running_mean.clone()
    model.batch_norm.running_var = model1.batch_norm.running_var.clone()
    model.batch_norm.momentum = 0.0
    model.eval()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = N_JOINT_ACTIONS
    vs, vi, vj = (torch.as_tensor(a) for a in (rng.integers(len(x_val), size=20000), rng.integers(n, size=20000),
                                                rng.integers(n, size=20000)))
    x_v, y_v = with_profiles(x_val[vs], vi, vj), torch.as_tensor(phi_val[vs.numpy(), vi.numpy(), vj.numpy()],
                                                                   dtype=torch.float32).unsqueeze(1)
    phi_t = torch.as_tensor(phi_train, dtype=torch.float32)
    best = {"mse": float("inf"), "epoch": 0, "state": None}
    steps = max(1, len(x_train) * profiles_per_state // batch)
    for epoch in range(1, epochs + 1):
        for _ in range(steps):
            s, i, j = (torch.as_tensor(rng.integers(m, size=batch)) for m in (len(x_train), n, n))
            loss = ((model(with_profiles(x_train[s], i, j)) - phi_t[s, i, j].unsqueeze(1)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            val = float(((model(x_v) - y_v) ** 2).mean())
        if val < best["mse"]:
            best = {"mse": val, "epoch": epoch, "state": {k: v.clone() for k, v in model.state_dict().items()}}
        print(f"  epoch {epoch}/{epochs}: train {float(loss):.4f}  validation {val:.4f}", flush=True)
    model.load_state_dict(best["state"])
    return model, best["epoch"], best["mse"]


def split_states(rally_path, seed, gamma, horizon):
    """Train / validation / held-out states, split by rally exactly as the alpha-potential trainer does."""
    from nash_skills.v2 import train_q_model_5skill_v3_attack_defend as tq
    with open(rally_path, "rb") as f:
        rallies = [r for r in pickle.load(f) if r.get("theta_mode") != "mixed"]
    train, hold = tq.split_rallies(rallies, tq.DEFAULT_HOLDOUT_FRAC, seed)
    train, val = tq.split_rallies(train, tq.DEFAULT_VAL_FRAC, seed + 1)
    return tuple(tq.build_dataset(r, gamma=gamma, horizon=horizon)[0] for r in (train, val, hold))


def main():
    import eval_table1_attack_defend as t1
    from nash_skills.v2 import train_q_model_5skill_v3_attack_defend as tq
    ap = argparse.ArgumentParser()
    ap.add_argument("--rallies", required=True)
    ap.add_argument("--model-dir", required=True, help="trained model directory whose Q1, Q2 define the game")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, required=True, help="the source model's training seed (same split)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--profiles-per-state", type=int, default=16)
    args = ap.parse_args()
    with open(os.path.join(args.model_dir, "heldout_attack_defend.json")) as f:
        source = json.load(f)
    models = t1._load_models(args.model_dir)
    start = time.time()
    x_train, x_val, x_hold = split_states(args.rallies, args.seed, source["gamma"], source.get("horizon"))
    print(f"states: {len(x_train)} train, {len(x_val)} validation, {len(x_hold)} held-out", flush=True)
    labels = {}
    for name, x in (("train", x_train), ("val", x_val), ("hold", x_hold)):
        q1, q2 = value_tables(models["model1"], models["model2"], x)
        labels[name] = lp_labels(q1, q2, args.workers)
        print(f"  {name}: {len(x)} per-state LPs solved ({time.time() - start:.0f}s)", flush=True)
    model_p, epoch, val_mse = regress_phi(models["model1"], x_train, labels["train"][0], x_val, labels["val"][0],
                                          args.epochs, args.lr, args.batch, args.profiles_per_state, args.seed)
    rng = np.random.default_rng(args.seed)
    residuals = tq.unilateral_residuals(model_p, models["model1"], models["model2"], x_hold, tq.build_constraints(),
                                        rng, n_constraints=tq.HELDOUT_N_CONSTRAINTS, batch_size=tq.BATCH_SIZE)
    os.makedirs(args.out_dir, exist_ok=True)
    for name in ("model1_attack_defend.pth", "model2_attack_defend.pth"):
        shutil.copy(os.path.join(args.model_dir, name), os.path.join(args.out_dir, name))
    torch.save(model_p.state_dict(), os.path.join(args.out_dir, "model_p_attack_defend.pth"))
    report = {**source, "phi_fit": "per_state_lp_regression", "phi_source_dir": args.model_dir,
              "phi_regression_epoch": epoch, "phi_regression_val_mse": val_mse,
              "heldout_residuals": tq.residual_stats(residuals),
              "heldout_residuals_source_phi": source.get("heldout_residuals"),
              "lp_alpha_heldout": tq.residual_stats(labels["hold"][1]),
              "lp_alpha_train": tq.residual_stats(labels["train"][1])}
    with open(os.path.join(args.out_dir, "heldout_attack_defend.json"), "w") as f:
        json.dump(report, f, indent=2)
    h, a = report["heldout_residuals"], report["lp_alpha_heldout"]
    print(f"held-out residuals of the LP-fitted Phi: p95 {h['p95_abs']:.3f}, mean {h['mean_abs']:.3f} "
          f"(source Phi p95 {source['heldout_residuals']['p95_abs']:.3f}); per-state LP alpha on held-out states: "
          f"median {a['median_abs']:.3f}, p95 {a['p95_abs']:.3f} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
