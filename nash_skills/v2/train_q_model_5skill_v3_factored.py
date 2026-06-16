"""
Train 5-skill Q-value + potential models using the v3 same-state per-sample
counterfactual potential training, with the FactoredModel estimator (§3.6
architecture ablation, v3 variant).

This is a one-for-one mirror of nash_skills/v2/train_q_model_5skill_v3.py
with the only change being SimpleModel -> FactoredModel. The Phi training
recipe stays identical to v3:

  - Q models trained on discounted-return targets (same as v2)
  - Phi training samples ONE base batch, clones into counterfactual
    skill assignments, and matches per-sample Q deltas (NOT minibatch means)
  - Same optimizer (Adam, lr=0.001), cosine LR scheduler, 1500 epochs

Outputs (with _5skill_v3_factored suffix so they never collide with the
SimpleModel or v2-factored versions):

    models/model1_5skill_v3_factored.pth
    models/model2_5skill_v3_factored.pth
    models/model_p_5skill_v3_factored.pth
    logs/train_q_5skill_v3_factored.csv
    logs/train_p_5skill_v3_factored.csv

Run (from project root):
    venv/bin/python nash_skills/v2/train_q_model_5skill_v3_factored.py
    venv/bin/python nash_skills/v2/train_q_model_5skill_v3_factored.py --epochs 2000
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import csv
import random

import numpy as np
import pickle as pkl
import torch
import torch.nn as nn

from model_arch import FactoredModel
from nash_skills.skills import N_SKILLS
from nash_skills.v2.labeling import compute_returns, check_balance, GAMMA

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
RALLY_PATH       = "data/rallies_5skill_v2.pkl"
N_EPOCHS         = 1500
LR               = 0.001
BATCH_SIZE       = 512
POTENTIAL_PAIRS  = 20
CHECKPOINT_EVERY = 100

# Skill columns occupy the last 2 dims of every state vector (ego, opp).
SKILL_DIM = 2

MODEL1_PATH  = "models/model1_5skill_v3_factored.pth"
MODEL2_PATH  = "models/model2_5skill_v3_factored.pth"
MODEL_P_PATH = "models/model_p_5skill_v3_factored.pth"
LOG_Q_PATH   = "logs/train_q_5skill_v3_factored.csv"
LOG_P_PATH   = "logs/train_p_5skill_v3_factored.csv"
# --------------------------------------------------------------------------- #


def build_dataset(rallies: list):
    """Convert rally list into (X, Y1, Y2) tensors using discounted returns."""
    x_list, y1_list, y2_list = [], [], []
    for entry in rallies:
        states = entry["states"]
        winner = entry.get("winner", 0)
        if len(states) == 0:
            continue
        g1, g2 = compute_returns(states, gamma=GAMMA, winner=winner)
        for state, v1, v2 in zip(states, g1, g2):
            x_list.append(state)
            y1_list.append([v1])
            y2_list.append([v2])

    x  = torch.tensor(np.array(x_list),  dtype=torch.float32)
    y1 = torch.tensor(np.array(y1_list), dtype=torch.float32)
    y2 = torch.tensor(np.array(y2_list), dtype=torch.float32)
    return x, y1, y2


def apply_skill_pair(x: torch.Tensor, ego_skill: int, opp_skill: int) -> torch.Tensor:
    """Clone a base batch and overwrite the 2 skill dimensions."""
    x_cf = x.clone()
    x_cf[:, -2] = ego_skill / (N_SKILLS - 1)
    x_cf[:, -1] = opp_skill / (N_SKILLS - 1)
    return x_cf


def sample_base_batch(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Sample one shared base batch of states for same-state counterfactuals."""
    idx = random.sample(range(len(x)), min(batch_size, len(x)))
    return x[idx]


def train(rally_path: str, n_epochs: int, lr: float) -> None:
    print(f"Loading {rally_path} ...")
    rallies = pkl.load(open(rally_path, "rb"))
    print(f"  {len(rallies)} rallies loaded")

    is_ok, ratio = check_balance(rallies, threshold=5.0)
    print(f"  Balance: max/min ratio={ratio:.2f}  {'OK' if is_ok else 'WARNING: imbalanced'}")

    x, y1, y2 = build_dataset(rallies)
    print(f"  Dataset: X={x.shape}, Y1={y1.shape}")

    n_nonzero = (y1.abs() > 1e-6).sum().item()
    print(f"  Non-zero Y1: {n_nonzero}/{len(y1)} = {100*n_nonzero/len(y1):.1f}%"
          f"  (sparse pipeline had ~4%)")

    total_dim = x.shape[1]
    state_dim = total_dim - SKILL_DIM
    print(f"  GAMMA={GAMMA}, total_dim={total_dim}, state_dim={state_dim},"
          f" skill_dim={SKILL_DIM}, N_SKILLS={N_SKILLS}")
    print(f"  Estimator: FactoredModel (state {state_dim}->32, skill {SKILL_DIM}->8, fusion [64,32,16]->1)")

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    criterion = nn.MSELoss()

    # ------------------------------------------------------------------ #
    # Train Q-value models                                                #
    # ------------------------------------------------------------------ #
    model1 = FactoredModel(state_dim=state_dim, skill_dim=SKILL_DIM)
    model2 = FactoredModel(state_dim=state_dim, skill_dim=SKILL_DIM)
    opt1 = torch.optim.Adam(model1.parameters(), lr=lr)
    opt2 = torch.optim.Adam(model2.parameters(), lr=lr)

    print(f"\nTraining Q-value models ({n_epochs} epochs) ...")
    with open(LOG_Q_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss1", "loss2"])
        for epoch in range(n_epochs):
            opt1.zero_grad()
            opt2.zero_grad()
            loss1 = criterion(model1(x), y1)
            loss2 = criterion(model2(x), y2)
            loss1.backward()
            loss2.backward()
            opt1.step()
            opt2.step()
            w.writerow([epoch + 1, loss1.item(), loss2.item()])
            if (epoch + 1) % CHECKPOINT_EVERY == 0:
                print(f"  Q-epoch {epoch+1}/{n_epochs}  "
                      f"loss1={loss1.item():.5f}  loss2={loss2.item():.5f}")

    torch.save(model1.state_dict(), MODEL1_PATH)
    torch.save(model2.state_dict(), MODEL2_PATH)
    print(f"  Saved {MODEL1_PATH} and {MODEL2_PATH}")

    # ------------------------------------------------------------------ #
    # Train potential model (v3 same-state per-sample counterfactuals)    #
    # ------------------------------------------------------------------ #
    model1.eval()
    model2.eval()

    model_p = FactoredModel(state_dim=state_dim, skill_dim=SKILL_DIM,
                            last_layer_activation=None)
    model_p.batch_norm.running_mean = model1.batch_norm.running_mean.clone()
    model_p.batch_norm.running_var  = model1.batch_norm.running_var.clone()
    model_p.batch_norm.momentum     = 0.0
    # eval() mode is critical: BN must use frozen stats so the skill column
    # difference survives across counterfactual skill clones.
    model_p.eval()

    opt_p = torch.optim.Adam(model_p.parameters(), lr=lr)
    scheduler_p = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_p, T_max=n_epochs, eta_min=1e-5
    )
    min_loss = float("inf")

    # Enumerate all unilateral-deviation constraints (no skill-pair filtering;
    # the same-state per-sample trick makes every (i, j) -> (i', j) reachable).
    constraints = []
    for i in range(N_SKILLS):
        for j in range(N_SKILLS):
            for i2 in range(N_SKILLS):
                if i2 != i:
                    constraints.append(("p1", i, j, i2, j))
            for j2 in range(N_SKILLS):
                if j2 != j:
                    constraints.append(("p2", i, j, i, j2))

    print(f"\nTraining potential model ({n_epochs} epochs, {len(constraints)} constraints) ...")
    with open(LOG_P_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "total_loss", "saved"])

        for epoch in range(n_epochs):
            total_loss = 0.0
            sampled = random.sample(constraints, min(len(constraints), POTENTIAL_PAIRS))

            opt_p.zero_grad()

            for c in sampled:
                x_base = sample_base_batch(x, BATCH_SIZE)

                if c[0] == "p1":
                    _, i, j, i2, _ = c
                    x_ij  = apply_skill_pair(x_base, i,  j)
                    x_i2j = apply_skill_pair(x_base, i2, j)
                    with torch.no_grad():
                        target = model1(x_ij) - model1(x_i2j)
                    loss = criterion(
                        model_p(x_ij) - model_p(x_i2j),
                        target,
                    )
                else:
                    _, i, j, _, j2 = c
                    x_ij  = apply_skill_pair(x_base, i, j)
                    x_ij2 = apply_skill_pair(x_base, i, j2)
                    with torch.no_grad():
                        target = model2(x_ij) - model2(x_ij2)
                    loss = criterion(
                        model_p(x_ij) - model_p(x_ij2),
                        target,
                    )

                loss.backward()
                total_loss += loss.item()

            opt_p.step()
            scheduler_p.step()

            saved = 0
            if total_loss < min_loss:
                min_loss = total_loss
                torch.save(model_p.state_dict(), MODEL_P_PATH)
                saved = 1
                if (epoch + 1) % CHECKPOINT_EVERY == 0 or epoch < 10:
                    print(f"  [saved] Phi-epoch {epoch+1}  loss={total_loss:.5f}")

            w.writerow([epoch + 1, total_loss, saved])
            if (epoch + 1) % CHECKPOINT_EVERY == 0 and not saved:
                print(f"  Phi-epoch {epoch+1}/{n_epochs}  total_loss={total_loss:.5f}")

    print(f"\nSaved {MODEL_P_PATH}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train 5-skill v3 Q-value and per-sample potential models "
                    "with the FactoredModel estimator (§3.6 ablation)."
    )
    parser.add_argument("--rallies", default=RALLY_PATH,
                        help=f"Rally pickle path (default: {RALLY_PATH})")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS,
                        help=f"Training epochs (default: {N_EPOCHS})")
    parser.add_argument("--lr", type=float, default=LR,
                        help=f"Learning rate (default: {LR})")
    args = parser.parse_args()

    train(args.rallies, args.epochs, args.lr)
