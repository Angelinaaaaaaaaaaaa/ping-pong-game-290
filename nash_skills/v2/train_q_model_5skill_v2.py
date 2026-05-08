"""
Train Q-value + potential models for the 5-skill pipeline using all v2 improvements:
  - discounted-return labels (gamma=0.7) instead of sparse {-1,0,+1}
  - correct gradient accumulation (single opt.step() per epoch)
  - LR=0.001 for both Q and potential (no dying-ReLU collapse)
  - balanced dataset: 50 rallies × 25 skill pairs = 1250 rallies
  - mini-batch potential training (no full-dataset tensor clones)
  - periodic checkpoints every 100 epochs

Reads:  data/rallies_5skill_v2.pkl   (collected by nash_skills/v2/collect_data.py
                                       with all 5 SKILL_NAMES)
Writes: models/model1_5skill_v2.pth
        models/model2_5skill_v2.pth
        models/model_p_5skill_v2.pth
        logs/train_q_5skill_v2.csv
        logs/train_p_5skill_v2.csv

This script is a thin wrapper around nash_skills/v2/train_models.py that
overrides the default file paths to use the 5-skill v2 dataset.

Run (from project root):
    venv/bin/python nash_skills/v2/train_q_model_5skill_v2.py
    venv/bin/python nash_skills/v2/train_q_model_5skill_v2.py --epochs 2000 --lr 0.0005
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

from model_arch import SimpleModel
from nash_skills.skills import SKILL_NAMES, N_SKILLS, skill_index
from nash_skills.v2.labeling import compute_returns, summarise_balance, check_balance, GAMMA
from nash_skills.v2.state_encoder import STATE_DIM

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
RALLY_PATH       = "data/rallies_5skill_v2.pkl"
N_EPOCHS         = 1500
LR               = 0.001
BATCH_SIZE       = 512
POTENTIAL_PAIRS  = 20       # constraint pairs sampled per potential epoch
CHECKPOINT_EVERY = 100

MODEL1_PATH  = "models/model1_5skill_v2.pth"
MODEL2_PATH  = "models/model2_5skill_v2.pth"
MODEL_P_PATH = "models/model_p_5skill_v2.pth"
LOG_Q_PATH   = "logs/train_q_5skill_v2.csv"
LOG_P_PATH   = "logs/train_p_5skill_v2.csv"
# --------------------------------------------------------------------------- #


def build_dataset(rallies: list):
    """Convert rally list into (X, Y1, Y2) tensors using discounted returns."""
    X_list  = []
    Y1_list = []
    Y2_list = []

    for entry in rallies:
        states = entry["states"]
        winner = entry.get("winner", 0)
        if len(states) == 0:
            continue

        g1, g2 = compute_returns(states, gamma=GAMMA, winner=winner)
        for state, v1, v2 in zip(states, g1, g2):
            X_list.append(state)
            Y1_list.append([v1])
            Y2_list.append([v2])

    X  = torch.tensor(np.array(X_list),  dtype=torch.float32)
    Y1 = torch.tensor(np.array(Y1_list), dtype=torch.float32)
    Y2 = torch.tensor(np.array(Y2_list), dtype=torch.float32)
    return X, Y1, Y2


def train(rally_path: str, n_epochs: int, lr: float) -> None:

    # ------------------------------------------------------------------ #
    # Load                                                                 #
    # ------------------------------------------------------------------ #
    print(f"Loading {rally_path} ...")
    rallies = pkl.load(open(rally_path, "rb"))
    print(f"  {len(rallies)} rallies loaded")

    is_ok, ratio = check_balance(rallies, threshold=5.0)
    print(f"  Balance: max/min ratio={ratio:.2f}  {'OK' if is_ok else 'WARNING: imbalanced'}")

    X, Y1, Y2 = build_dataset(rallies)
    print(f"  Dataset: X={X.shape}, Y1={Y1.shape}")

    n_nonzero = (Y1.abs() > 1e-6).sum().item()
    print(f"  Non-zero Y1: {n_nonzero}/{len(Y1)} = {100*n_nonzero/len(Y1):.1f}%"
          f"  (sparse pipeline had ~4%)")
    print(f"  GAMMA={GAMMA}, STATE_DIM={STATE_DIM}, N_SKILLS={N_SKILLS}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    criterion = nn.MSELoss()

    # ------------------------------------------------------------------ #
    # Train Q-value models                                                 #
    # ------------------------------------------------------------------ #
    model1 = SimpleModel(STATE_DIM, [64, 32, 16], 1)
    model2 = SimpleModel(STATE_DIM, [64, 32, 16], 1)
    opt1 = torch.optim.Adam(model1.parameters(), lr=lr)
    opt2 = torch.optim.Adam(model2.parameters(), lr=lr)

    print(f"\nTraining Q-value models ({n_epochs} epochs) ...")
    with open(LOG_Q_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss1", "loss2"])
        for epoch in range(n_epochs):
            opt1.zero_grad()
            opt2.zero_grad()
            loss1 = criterion(model1(X), Y1)
            loss2 = criterion(model2(X), Y2)
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
    # Train potential model                                                #
    # ------------------------------------------------------------------ #
    model1.eval()
    model2.eval()

    model_p = SimpleModel(STATE_DIM, [64, 32, 16], 1, last_layer_activation=None)
    model_p.batch_norm.running_mean = model1.batch_norm.running_mean.clone()
    model_p.batch_norm.running_var  = model1.batch_norm.running_var.clone()
    model_p.batch_norm.momentum     = 0.0

    opt_p = torch.optim.Adam(model_p.parameters(), lr=lr)
    scheduler_p = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_p, T_max=n_epochs, eta_min=1e-5
    )
    min_loss = float("inf")

    # Build per-skill-pair row index lists (no full X clones in memory)
    skill_pair_indices: dict = {}
    offset = 0
    for entry in rallies:
        n   = len(entry["states"])
        key = (skill_index(entry["skill1"]), skill_index(entry["skill2"]))
        skill_pair_indices.setdefault(key, []).extend(range(offset, offset + n))
        offset += n

    # All valid unilateral-deviation constraint pairs
    constraints = []
    for i in range(N_SKILLS):
        for j in range(N_SKILLS):
            for i2 in range(N_SKILLS):
                if i2 != i and (i, j) in skill_pair_indices and (i2, j) in skill_pair_indices:
                    constraints.append(("p1", i, j, i2, j))
            for j2 in range(N_SKILLS):
                if j2 != j and (i, j) in skill_pair_indices and (i, j2) in skill_pair_indices:
                    constraints.append(("p2", i, j, i, j2))

    print(f"\nTraining potential model ({n_epochs} epochs, {len(constraints)} constraints) ...")
    with open(LOG_P_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "total_loss", "saved"])

        for epoch in range(n_epochs):
            total_loss = 0.0
            sampled    = random.sample(constraints, min(len(constraints), POTENTIAL_PAIRS))

            opt_p.zero_grad()                        # single zero per epoch

            for c in sampled:
                if c[0] == "p1":
                    _, i, j, i2, _ = c
                    idx_ij  = random.sample(skill_pair_indices[(i,  j)],
                                            min(BATCH_SIZE, len(skill_pair_indices[(i,  j)])))
                    idx_i2j = random.sample(skill_pair_indices[(i2, j)],
                                            min(BATCH_SIZE, len(skill_pair_indices[(i2, j)])))
                    Xij_s  = X[idx_ij].clone();  Xij_s[:,  -2] = i  / (N_SKILLS - 1)
                    Xi2j_s = X[idx_i2j].clone(); Xi2j_s[:, -2] = i2 / (N_SKILLS - 1)
                    with torch.no_grad():
                        target = model1(Xij_s).mean() - model1(Xi2j_s).mean()
                    loss = (model_p(Xij_s).mean() - model_p(Xi2j_s).mean() - target) ** 2

                else:  # p2
                    _, i, j, _, j2 = c
                    idx_ij  = random.sample(skill_pair_indices[(i, j)],
                                            min(BATCH_SIZE, len(skill_pair_indices[(i, j)])))
                    idx_ij2 = random.sample(skill_pair_indices[(i, j2)],
                                            min(BATCH_SIZE, len(skill_pair_indices[(i, j2)])))
                    Xij_s  = X[idx_ij].clone();  Xij_s[:,  -1] = j  / (N_SKILLS - 1)
                    Xij2_s = X[idx_ij2].clone(); Xij2_s[:, -1] = j2 / (N_SKILLS - 1)
                    with torch.no_grad():
                        target = model2(Xij_s).mean() - model2(Xij2_s).mean()
                    loss = (model_p(Xij_s).mean() - model_p(Xij2_s).mean() - target) ** 2

                loss.backward()                      # accumulate gradients
                total_loss += loss.item()

            opt_p.step()                             # single step per epoch
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


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train 5-skill v2 Q-value and potential models (discounted returns)"
    )
    parser.add_argument("--rallies", default=RALLY_PATH,
                        help=f"Rally pickle path (default: {RALLY_PATH})")
    parser.add_argument("--epochs",  type=int,   default=N_EPOCHS,
                        help=f"Training epochs (default: {N_EPOCHS})")
    parser.add_argument("--lr",      type=float, default=LR,
                        help=f"Learning rate (default: {LR})")
    args = parser.parse_args()

    train(args.rallies, args.epochs, args.lr)
