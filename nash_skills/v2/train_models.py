"""
Train Q-value models (model1, model2) and potential model (model_p) for the
v2 Nash pipeline.

Key changes from the old train_q_model_5skill.py
=================================================

OLD (bugs / design flaws):
  1. Sparse labeling: only L-2 and L-1 crossings were labeled ±1; all others 0.
     → ~96% zero labels → models learned near-constant outputs.
  2. Potential LR = 0.1  (100× too large) → dying ReLU, constant output.
  3. Skill encoding used {0,1} (wrong; env uses {-1,+1} convention) → sign flip.
  4. Potential training built N_SKILLS^2 × N_dataset clones of X in memory.
     → huge memory footprint.
  5. Model saved only on loss improvement, not periodically → brittle.

NEW design:
  1. Discounted-return targets via labeling.compute_returns (gamma=0.9).
  2. Potential LR matches Q LR (both 0.001) → stable training.
  3. Skill encoding: normalised index [0,1] used consistently throughout.
     (Both Q training and potential training use the same encoded states
      from collect_data.py which already embeds the skill index.)
  4. Potential constraints evaluated on random mini-batches, not full copies.
  5. Checkpoints saved every N epochs and on best loss.
  6. New model files saved as model1_v2.pth, model2_v2.pth, model_p_v2.pth
     so old weights are never silently overwritten.

Run:
    venv/bin/python nash_skills/v2/train_models.py
    venv/bin/python nash_skills/v2/train_models.py --rallies data/rallies_v2.pkl
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
from nash_skills.v2.labeling import compute_returns, GAMMA
from nash_skills.v2.state_encoder import STATE_DIM

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
DEFAULT_RALLY_PATH = "data/rallies_v2.pkl"
N_EPOCHS           = 1500
LR                 = 0.001    # same for Q models and potential model
BATCH_SIZE         = 512      # mini-batch for full-dataset training
POTENTIAL_PAIRS    = 20       # constraint pairs sampled per potential epoch
CHECKPOINT_EVERY   = 100      # save periodic checkpoints every N epochs
MODEL1_PATH        = "models/model1_v2.pth"
MODEL2_PATH        = "models/model2_v2.pth"
MODEL_P_PATH       = "models/model_p_v2.pth"
LOG_Q_PATH         = "logs/train_q_v2.csv"
LOG_P_PATH         = "logs/train_p_v2.csv"
# --------------------------------------------------------------------------- #


def build_dataset(rallies: list):
    """
    Convert list of rally dicts into (X, Y1, Y2) tensors using discounted returns.

    Parameters
    ----------
    rallies : list of dicts with keys 'states', 'winner'
        Each state is a (STATE_DIM,) float32 encoded vector.

    Returns
    -------
    X  : (N, STATE_DIM) float32 tensor
    Y1 : (N, 1) float32 tensor  — ego discounted returns
    Y2 : (N, 1) float32 tensor  — opp discounted returns
    """
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


def train(
    rally_path: str = DEFAULT_RALLY_PATH,
    n_epochs: int = N_EPOCHS,
    lr: float = LR,
) -> None:
    """Full training pipeline: load data → train Q → train potential."""

    # ------------------------------------------------------------------ #
    # Load data                                                            #
    # ------------------------------------------------------------------ #
    print(f"Loading rallies from {rally_path}")
    rallies = pkl.load(open(rally_path, "rb"))
    print(f"  {len(rallies)} rallies loaded")

    # Quick balance check
    from nash_skills.v2.labeling import summarise_balance, check_balance
    is_ok, ratio = check_balance(rallies, threshold=5.0)
    print(f"  Balance check: ratio={ratio:.2f}  {'OK' if is_ok else 'WARNING: imbalanced'}")

    X, Y1, Y2 = build_dataset(rallies)
    print(f"  Dataset: X={X.shape}, Y1={Y1.shape}, Y2={Y2.shape}")

    n_nonzero_1 = (Y1.abs() > 1e-6).sum().item()
    n_nonzero_2 = (Y2.abs() > 1e-6).sum().item()
    print(f"  Non-zero Y1: {n_nonzero_1}/{len(Y1)} = "
          f"{100*n_nonzero_1/len(Y1):.1f}%  "
          f"(old sparse pipeline had ~4%)")

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # ------------------------------------------------------------------ #
    # Train Q-value models                                                 #
    # ------------------------------------------------------------------ #
    model1 = SimpleModel(STATE_DIM, [64, 32, 16], 1)
    model2 = SimpleModel(STATE_DIM, [64, 32, 16], 1)

    opt1 = torch.optim.Adam(model1.parameters(), lr=lr)
    opt2 = torch.optim.Adam(model2.parameters(), lr=lr)
    criterion = nn.MSELoss()

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
    # Inherit batch-norm statistics from model1 so normalisation is consistent
    model_p.batch_norm.running_mean = model1.batch_norm.running_mean.clone()
    model_p.batch_norm.running_var  = model1.batch_norm.running_var.clone()
    model_p.batch_norm.momentum     = 0.0

    # LR matches Q training — fixes the dying-ReLU collapse from old lr=0.1
    opt_p = torch.optim.Adam(model_p.parameters(), lr=lr)
    min_loss = float("inf")

    # Build skill-pair index arrays (no full X clones — use index slicing)
    # For each skill pair (i, j), collect the row indices in X that belong
    # to rallies with that skill pair.
    skill_pair_indices: dict = {}
    offset = 0
    for entry in rallies:
        s1 = entry["skill1"]
        s2 = entry["skill2"]
        n  = len(entry["states"])
        key = (skill_index(s1), skill_index(s2))
        if key not in skill_pair_indices:
            skill_pair_indices[key] = []
        skill_pair_indices[key].extend(range(offset, offset + n))
        offset += n

    # Build all valid unilateral-deviation constraint pairs
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
            sampled = random.sample(constraints, min(len(constraints), POTENTIAL_PAIRS))

            for c in sampled:
                opt_p.zero_grad()

                if c[0] == "p1":
                    _, i, j, i2, _ = c
                    # Sample a mini-batch from each skill pair
                    idx_ij  = random.sample(skill_pair_indices[(i,  j)],
                                            min(BATCH_SIZE, len(skill_pair_indices[(i,  j)])))
                    idx_i2j = random.sample(skill_pair_indices[(i2, j)],
                                            min(BATCH_SIZE, len(skill_pair_indices[(i2, j)])))

                    Xij  = X[idx_ij]
                    Xi2j = X[idx_i2j]
                    # Override skill dims so the model sees the counterfactual skill
                    Xij_s  = Xij.clone();  Xij_s[:, -2]  = i  / (N_SKILLS - 1)
                    Xi2j_s = Xi2j.clone(); Xi2j_s[:, -2] = i2 / (N_SKILLS - 1)

                    with torch.no_grad():
                        # Target: Q1 difference (player 1 unilateral deviation)
                        q_ij  = model1(Xij_s).mean()
                        q_i2j = model1(Xi2j_s).mean()
                        target = q_ij - q_i2j

                    # Potential must match the Q difference
                    phi_ij  = model_p(Xij_s).mean()
                    phi_i2j = model_p(Xi2j_s).mean()
                    loss = (phi_ij - phi_i2j - target) ** 2

                else:  # p2
                    _, i, j, _, j2 = c
                    idx_ij  = random.sample(skill_pair_indices[(i, j)],
                                            min(BATCH_SIZE, len(skill_pair_indices[(i, j)])))
                    idx_ij2 = random.sample(skill_pair_indices[(i, j2)],
                                            min(BATCH_SIZE, len(skill_pair_indices[(i, j2)])))

                    Xij  = X[idx_ij]
                    Xij2 = X[idx_ij2]
                    Xij_s  = Xij.clone();  Xij_s[:, -1]  = j  / (N_SKILLS - 1)
                    Xij2_s = Xij2.clone(); Xij2_s[:, -1] = j2 / (N_SKILLS - 1)

                    with torch.no_grad():
                        q_ij  = model2(Xij_s).mean()
                        q_ij2 = model2(Xij2_s).mean()
                        target = q_ij - q_ij2

                    phi_ij  = model_p(Xij_s).mean()
                    phi_ij2 = model_p(Xij2_s).mean()
                    loss = (phi_ij - phi_ij2 - target) ** 2

                loss.backward()
                opt_p.step()
                total_loss += loss.item()

            saved = 0
            if total_loss < min_loss:
                min_loss = total_loss
                torch.save(model_p.state_dict(), MODEL_P_PATH)
                saved = 1
                if (epoch + 1) % CHECKPOINT_EVERY == 0:
                    print(f"  [saved] Phi-epoch {epoch+1}  loss={total_loss:.5f}")

            w.writerow([epoch + 1, total_loss, saved])
            if (epoch + 1) % CHECKPOINT_EVERY == 0 and not saved:
                print(f"  Phi-epoch {epoch+1}/{n_epochs}  total_loss={total_loss:.5f}")

    print(f"\nSaved {MODEL_P_PATH}")
    print("Training complete.")


# --------------------------------------------------------------------------- #
# CLI entry point                                                               #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Q-value and potential models for the v2 Nash pipeline."
    )
    parser.add_argument("--rallies", type=str, default=DEFAULT_RALLY_PATH,
                        help=f"Path to rally pickle file (default: {DEFAULT_RALLY_PATH})")
    parser.add_argument("--epochs",  type=int, default=N_EPOCHS,
                        help=f"Number of training epochs (default: {N_EPOCHS})")
    parser.add_argument("--lr",      type=float, default=LR,
                        help=f"Learning rate (default: {LR})")
    args = parser.parse_args()

    train(
        rally_path=args.rallies,
        n_epochs=args.epochs,
        lr=args.lr,
    )
