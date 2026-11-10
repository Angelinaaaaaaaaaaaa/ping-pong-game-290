"""
Train Q-value + potential models for the 5-skill pipeline, identical to
train_q_model_5skill_v3.py EXCEPT it consumes the 84-dim one-hot skill
encoding (nash_skills.v2.state_encoder_one_hot.STATE_DIM_ONE_HOT) instead
of the legacy 76-dim scalar encoding.

This is a NEW, separate script -- it does not modify
train_q_model_5skill_v3.py, and its default output paths
(models/*_one_hot.pth, logs/*_one_hot.csv) never collide with the legacy
checkpoints. Only `apply_skill_pair` differs in actual logic (see its
docstring below); everything else (build_dataset, the Q-training loop,
the potential model's same-state counterfactual-constraint training) is
unchanged from v3.

Input data must come from collect_data.py --encoder one_hot (84-dim
states) -- NOT the legacy 76-dim pickles. The legacy trainer's
assumptions about where skill identity lives (a single scalar at [-2]/
[-1]) do not apply here; this script's apply_skill_pair instead zeroes
and sets the 5-dim one-hot block at [74:79] (ego) / [79:84] (opp).

Run:
    venv/bin/python nash_skills/v2/train_q_model_5skill_v3_one_hot.py
    venv/bin/python nash_skills/v2/train_q_model_5skill_v3_one_hot.py --epochs 2000
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
from nash_skills.skills import N_SKILLS
from nash_skills.v2.labeling import compute_returns, check_balance, GAMMA
from nash_skills.v2.state_encoder_one_hot import STATE_DIM_ONE_HOT as STATE_DIM

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
RALLY_PATH       = "data/rallies_5skill_v2_one_hot.pkl"
N_EPOCHS         = 1500
LR               = 0.001
BATCH_SIZE       = 512
POTENTIAL_PAIRS  = 20
CHECKPOINT_EVERY = 100

MODEL1_PATH  = "models/model1_5skill_v3_one_hot.pth"
MODEL2_PATH  = "models/model2_5skill_v3_one_hot.pth"
MODEL_P_PATH = "models/model_p_5skill_v3_one_hot.pth"
LOG_Q_PATH   = "logs/train_q_5skill_v3_one_hot.csv"
LOG_P_PATH   = "logs/train_p_5skill_v3_one_hot.csv"

# ego one-hot block [74:79], opp one-hot block [79:84] -- see
# nash_skills.v2.state_encoder_one_hot's module docstring for the layout.
_EGO_ONE_HOT = slice(74, 74 + N_SKILLS)
_OPP_ONE_HOT = slice(74 + N_SKILLS, 74 + 2 * N_SKILLS)
# --------------------------------------------------------------------------- #


def build_dataset(rallies: list):
    """Convert rally list into (X, Y1, Y2) tensors using discounted returns.
    Identical to the legacy v3 trainer -- generic over state dimension."""
    x_list = []
    y1_list = []
    y2_list = []

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

    x = torch.tensor(np.array(x_list), dtype=torch.float32)
    y1 = torch.tensor(np.array(y1_list), dtype=torch.float32)
    y2 = torch.tensor(np.array(y2_list), dtype=torch.float32)
    return x, y1, y2


def apply_skill_pair(x: torch.Tensor, ego_skill: int, opp_skill: int) -> torch.Tensor:
    """
    Clone a base batch and overwrite the one-hot skill-identity blocks.

    Differs from the legacy v3 trainer's apply_skill_pair, which writes a
    single scalar to x[:, -2] / x[:, -1] -- correct for the 76-dim scalar
    encoding, but wrong here: skill identity in the 84-dim one-hot
    encoding is a 5-dim one-hot block at [74:79] (ego) / [79:84] (opp),
    not a scalar. This zeroes each full block before setting the chosen
    index, so the result is always a clean one-hot vector regardless of
    what was in those dims before (the counterfactual-constraint training
    loop clones real encoded states, which already have a valid one-hot
    block, but zeroing first makes this function correct even on
    arbitrary input, matching its unit tests).
    """
    x_cf = x.clone()
    x_cf[:, _EGO_ONE_HOT] = 0.0
    x_cf[:, _OPP_ONE_HOT] = 0.0
    x_cf[:, 74 + ego_skill] = 1.0
    x_cf[:, 74 + N_SKILLS + opp_skill] = 1.0
    return x_cf


def sample_base_batch(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Sample one shared base batch of states for same-state counterfactuals."""
    idx = random.sample(range(len(x)), min(batch_size, len(x)))
    return x[idx]


def train(rally_path: str, n_epochs: int, lr: float) -> None:
    print(f"Loading {rally_path} ...")
    with open(rally_path, "rb") as f:
        rallies = pkl.load(f)
    print(f"  {len(rallies)} rallies loaded")

    bad_encoder = [r.get("encoder") for r in rallies if r.get("encoder") != "one_hot"]
    if bad_encoder:
        raise ValueError(
            f"{len(bad_encoder)} rallies were not collected with encoder='one_hot' "
            f"(found: {set(bad_encoder)}). Recollect with "
            f"'collect_data.py --encoder one_hot' -- a legacy 76-dim pickle cannot "
            f"be reused here, the info-dict fields needed to re-encode were never saved."
        )

    print("  Flip augmentation: disabled (uses original rallies only)")

    is_ok, ratio = check_balance(rallies, threshold=5.0)
    print(f"  Balance: max/min ratio={ratio:.2f}  {'OK' if is_ok else 'WARNING: imbalanced'}")

    x, y1, y2 = build_dataset(rallies)
    print(f"  Dataset: X={x.shape}, Y1={y1.shape}")
    if x.shape[1] != STATE_DIM:
        raise ValueError(f"Expected {STATE_DIM}-dim states, got {x.shape[1]}")

    n_nonzero = (y1.abs() > 1e-6).sum().item()
    print(f"  Non-zero Y1: {n_nonzero}/{len(y1)} = {100*n_nonzero/len(y1):.1f}%")
    print(f"  GAMMA={GAMMA}, STATE_DIM={STATE_DIM} (one-hot), N_SKILLS={N_SKILLS}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

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
    # Train potential model                                                #
    # ------------------------------------------------------------------ #
    model1.eval()
    model2.eval()

    model_p = SimpleModel(STATE_DIM, [64, 32, 16], 1, last_layer_activation=None)
    model_p.batch_norm.running_mean = model1.batch_norm.running_mean.clone()
    model_p.batch_norm.running_var = model1.batch_norm.running_var.clone()
    model_p.batch_norm.momentum = 0.0
    model_p.eval()

    opt_p = torch.optim.Adam(model_p.parameters(), lr=lr)
    scheduler_p = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_p, T_max=n_epochs, eta_min=1e-5
    )
    min_loss = float("inf")

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
                    x_ij = apply_skill_pair(x_base, i, j)
                    x_i2j = apply_skill_pair(x_base, i2, j)
                    with torch.no_grad():
                        target = model1(x_ij) - model1(x_i2j)
                    loss = criterion(
                        model_p(x_ij) - model_p(x_i2j),
                        target,
                    )
                else:
                    _, i, j, _, j2 = c
                    x_ij = apply_skill_pair(x_base, i, j)
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
        description="Train 5-skill v3 Q-value and per-sample potential models (one-hot skill encoding)"
    )
    parser.add_argument("--rallies", default=RALLY_PATH,
                        help=f"Rally pickle path (default: {RALLY_PATH})")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS,
                        help=f"Training epochs (default: {N_EPOCHS})")
    parser.add_argument("--lr", type=float, default=LR,
                        help=f"Learning rate (default: {LR})")
    args = parser.parse_args()

    train(args.rallies, args.epochs, args.lr)
