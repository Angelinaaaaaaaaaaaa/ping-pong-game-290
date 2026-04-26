"""
Train Q-value models (model1, model2) and potential model (model_p) for the
2-skill v2 Nash pipeline.

Key changes from the original train_q_model.py
===============================================

OLD (bugs / design flaws):
  1. Sparse labeling: only the last 2 crossings per rally got ±1 labels;
     all earlier crossings got 0. → ~96% zero labels → models learned
     near-constant outputs.
  2. Potential LR = 0.1 (100× too large) → dying ReLU → constant output.
  3. Skill encoding used {0, 1} instead of the env's {-1, +1} convention
     → sign-flipped potential training targets.
  4. Skills were randomised mid-rally → no stable (skill1, skill2) label.
  5. 'done' / winner not recorded per rally → labeling couldn't tell wins
     from truncations.

NEW design:
  1. Discounted-return targets via labeling.compute_returns (gamma=0.9).
     Every crossing in a won rally receives a nonzero label.
  2. Potential LR = 0.001 (same as Q models) → stable training.
  3. Skill encoding: {-1, +1} matching the env's side_target convention.
     X01[:,-2] = -1.0  (left ego)
     X10[:,-1] = -1.0  (left opp)
     X00[:,-2] = -1.0  (left ego)
     X00[:,-1] = -1.0  (left opp)
  4. One stable (skill1, skill2) label per rally (set by collect_data_v2.py).
  5. winner field recorded per rally; truncated rallies → all-zero returns.
  6. New model files: model1_76dim.pth, model2_76dim.pth, model_p_76dim.pth —
     uses _76dim suffix; trained on 76-dim encoded state (gantry + joint angles
     + ball pos/vel + skill indicators) from collect_data_v2.py.

Run:
    venv/bin/python train_q_model_v2.py
    venv/bin/python train_q_model_v2.py \
        --rallies data/rallies_v2_2skill.pkl --epochs 1500
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import csv

import numpy as np
import pickle as pkl
import torch
import torch.nn as nn

from model_arch import SimpleModel
from nash_skills.v2.labeling import compute_returns, GAMMA

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
DEFAULT_RALLY_PATH = "data/rallies_v2_2skill.pkl"
N_EPOCHS           = 1500
LR                 = 0.001    # same for Q models and potential model
# Use _76dim suffix: these models take 76-dim encoded state (gantry+joints+ball+skill).
# This avoids clobbering the 116-dim raw-obs models (*_2skill_v2.pth) and the
# 76-dim 5-skill models (*_v2.pth) written by nash_skills/v2/train_models.py.
MODEL1_PATH        = "models/model1_76dim.pth"
MODEL2_PATH        = "models/model2_76dim.pth"
MODEL_P_PATH       = "models/model_p_76dim.pth"
LOG_Q_PATH         = "logs/train_q_76dim.csv"
LOG_P_PATH         = "logs/train_p_76dim.csv"
CHECKPOINT_EVERY   = 100
# --------------------------------------------------------------------------- #


def build_dataset_2skill(rallies: list, gamma: float = GAMMA):
    """
    Convert 2-skill rally dicts into (X, Y1, Y2) tensors using discounted returns.

    Parameters
    ----------
    rallies : list of dicts with keys 'states', 'winner'
        Each state is a numpy array (any dim — may be 116-dim raw obs or
        76-dim encoded, depending on the collector used).
    gamma   : discount factor (default 0.9)

    Returns
    -------
    X  : (N,) torch.Tensor  [float32, shape (N, state_dim)]
    Y1 : (N,) torch.Tensor  [float32, shape (N,)]   — ego discounted returns
    Y2 : (N,) torch.Tensor  [float32, shape (N,)]   — opp discounted returns
    """
    X_list  = []
    Y1_list = []
    Y2_list = []

    for entry in rallies:
        states = entry["states"]
        winner = entry.get("winner", 0)

        if len(states) == 0:
            continue

        g1, g2 = compute_returns(states, gamma=gamma, winner=winner)

        for state, v1, v2 in zip(states, g1, g2):
            X_list.append(state)
            Y1_list.append(v1)
            Y2_list.append(v2)

    X  = torch.tensor(np.array(X_list),  dtype=torch.float32)
    Y1 = torch.tensor(np.array(Y1_list), dtype=torch.float32)
    Y2 = torch.tensor(np.array(Y2_list), dtype=torch.float32)
    return X, Y1, Y2


def train(
    rally_path: str = DEFAULT_RALLY_PATH,
    n_epochs: int = N_EPOCHS,
    lr: float = LR,
) -> None:
    """Full 2-skill training pipeline: load data → train Q → train potential."""

    # ------------------------------------------------------------------ #
    # Load data                                                            #
    # ------------------------------------------------------------------ #
    print(f"Loading rallies from {rally_path}")
    rallies = pkl.load(open(rally_path, "rb"))
    print(f"  {len(rallies)} rallies loaded")

    X, Y1_flat, Y2_flat = build_dataset_2skill(rallies, gamma=GAMMA)
    # Reshape for MSELoss: (N, 1)
    Y1 = Y1_flat.unsqueeze(1)
    Y2 = Y2_flat.unsqueeze(1)
    print(f"  Dataset: X={X.shape}, Y1={Y1.shape}, Y2={Y2.shape}")

    n_nonzero = (Y1.abs() > 1e-6).sum().item()
    print(f"  Non-zero Y1: {n_nonzero}/{len(Y1)} = "
          f"{100*n_nonzero/len(Y1):.1f}%  "
          f"(old sparse pipeline had ~4%)")

    state_dim = X.shape[1]

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    # ------------------------------------------------------------------ #
    # Train Q-value models                                                 #
    # ------------------------------------------------------------------ #
    model1 = SimpleModel(state_dim, [64, 32, 16], 1)
    model2 = SimpleModel(state_dim, [64, 32, 16], 1)

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)
    criterion  = nn.MSELoss()

    print(f"\nTraining Q-value models ({n_epochs} epochs) ...")
    with open(LOG_Q_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss1", "loss2"])

        for epoch in range(n_epochs):
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss1 = criterion(model1(X), Y1)
            loss2 = criterion(model2(X), Y2)
            loss1.backward()
            loss2.backward()
            optimizer1.step()
            optimizer2.step()
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

    model_p = SimpleModel(state_dim, [64, 32, 16], 1, last_layer_activation=None)
    # Inherit batch-norm statistics from model1 for consistent normalisation
    model_p.batch_norm.running_mean = model1.batch_norm.running_mean.clone()
    model_p.batch_norm.running_var  = model1.batch_norm.running_var.clone()
    model_p.batch_norm.momentum     = 0.0
    # CRITICAL: switch to eval mode so BN uses the frozen running stats above.
    # In train() mode BatchNorm normalises each batch independently, which maps
    # the constant skill-encoding column (±1) to 0 in both X11 and X01 batches,
    # making phi(X11) == phi(X01) exactly → gradient = 0 → weights never update.
    # In eval() mode both batches share the same fixed BN transform, so the ±1
    # skill difference survives and the gradient is nonzero.
    model_p.eval()

    # LR = 0.001 — fixes the dying-ReLU collapse from old lr=0.1
    optimizer_p = torch.optim.Adam(model_p.parameters(), lr=0.001)

    # Build four skill-pair variants of the dataset.
    # 2-skill: left=-1.0, right=+1.0  (matching env's side_target convention)
    # Encoding: obs[-2] = ego skill, obs[-1] = opp skill
    X11 = X.clone(); X11[:,-2] = 1.;  X11[:,-1] = 1.   # right ego, right opp
    X01 = X.clone(); X01[:,-2] = -1.; X01[:,-1] = 1.   # left ego,  right opp
    X10 = X.clone(); X10[:,-2] = 1.;  X10[:,-1] = -1.  # right ego, left opp
    X00 = X.clone(); X00[:,-2] = -1.; X00[:,-1] = -1.  # left ego,  left opp

    min_loss = float("inf")

    # Constraint pairs for 2-skill potential:
    # Player 1 deviations: (right,right)↔(left,right) and (right,left)↔(left,left)
    # Player 2 deviations: (right,right)↔(right,left) and (left,right)↔(left,left)
    print(f"\nTraining potential model ({n_epochs} epochs) ...")
    with open(LOG_P_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "total_loss", "saved"])

        for epoch in range(n_epochs):
            total_loss = 0.0

            with torch.no_grad():
                # Player 1 unilateral deviation constraints
                target_p1_right = model1(X11).mean() - model1(X01).mean()  # right vs left (opp=right)
                target_p1_left  = model1(X10).mean() - model1(X00).mean()  # right vs left (opp=left)
                # Player 2 unilateral deviation constraints
                target_p2_right = model2(X11).mean() - model2(X10).mean()  # right vs left (ego=right)
                target_p2_left  = model2(X01).mean() - model2(X00).mean()  # right vs left (ego=left)

            # Constraint 1: player 1 deviation with opp=right
            optimizer_p.zero_grad()
            loss_p = criterion(
                model_p(X11).mean() - model_p(X01).mean(),
                target_p1_right,
            )
            loss_p.backward()
            optimizer_p.step()
            total_loss += loss_p.item()

            # Constraint 2: player 1 deviation with opp=left
            optimizer_p.zero_grad()
            loss_p = criterion(
                model_p(X10).mean() - model_p(X00).mean(),
                target_p1_left,
            )
            loss_p.backward()
            optimizer_p.step()
            total_loss += loss_p.item()

            # Constraint 3: player 2 deviation with ego=right
            optimizer_p.zero_grad()
            loss_p = criterion(
                model_p(X11).mean() - model_p(X10).mean(),
                target_p2_right,
            )
            loss_p.backward()
            optimizer_p.step()
            total_loss += loss_p.item()

            # Constraint 4: player 2 deviation with ego=left
            optimizer_p.zero_grad()
            loss_p = criterion(
                model_p(X01).mean() - model_p(X00).mean(),
                target_p2_left,
            )
            loss_p.backward()
            optimizer_p.step()
            total_loss += loss_p.item()

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
        description="Train 2-skill Q-value and potential models for the v2 Nash pipeline."
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
