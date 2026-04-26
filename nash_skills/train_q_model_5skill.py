"""
Train Q-value estimators and Nash potential function for the 5-skill pipeline.

Reads:  data/rallies_5skill.pkl
Writes: models/model1_5skill.pth
        models/model2_5skill.pth
        models/model_p_5skill.pth

The observation vector already contains (side_target, side_target_opp) in the
last two positions. For the 5-skill case we replace those two scalars with
(skill_idx1 / 4, skill_idx2 / 4) so the model sees a normalised skill index
instead of the raw ±1 side values that were used in the 2-skill baseline.

This keeps the model architecture identical (SimpleModel) and the training
loop identical to the baseline in train_q_model.py.

Run:
    python nash_skills/train_q_model_5skill.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import random
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn

from model_arch import SimpleModel
from nash_skills.skills import SKILL_NAMES, N_SKILLS, skill_index

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
RALLY_PATH  = "data/rallies_5skill.pkl"
N_EPOCHS    = 1500
LR          = 0.001
LOG_PATH_Q  = "logs/train_q_5skill.csv"
LOG_PATH_P  = "logs/train_p_5skill.csv"
# --------------------------------------------------------------------------- #

rallies = pkl.load(open(RALLY_PATH, "rb"))
print(f"Loaded {len(rallies)} rallies")

X  = []
Y1 = []
Y2 = []

for entry in rallies:
    skill1 = entry['skill1']
    skill2 = entry['skill2']
    states = entry['states']

    # Normalised skill indices in [0, 1]
    idx1_norm = skill_index(skill1) / (N_SKILLS - 1)
    idx2_norm = skill_index(skill2) / (N_SKILLS - 1)

    L = len(states)
    if L == 0:
        continue

    # Determine outcome from last state's ball velocity
    last_ball_vel = states[-1][39:42]
    last_ball_pos = states[-1][36:39]
    v_table = -np.sqrt(last_ball_vel[2] ** 2 + 2 * 9.81 * max(last_ball_pos[2] - 0.56, 0))
    t = (last_ball_vel[2] - v_table) / 9.81
    x_table = last_ball_pos[0] + last_ball_vel[0] * t
    y_table = last_ball_pos[1] + last_ball_vel[1] * t

    # Trim last crossing if ball went out of bounds (same logic as baseline)
    if last_ball_vel[0] > 0:
        if not (1.5 < x_table < 2.87 and -0.76 < y_table < 0.76):
            states = states[:-1]
    else:
        if not (0.13 < x_table < 1.5 and -0.76 < y_table < 0.76):
            states = states[:-1]

    L = len(states)
    for i, state in enumerate(states):
        s = state.copy()
        # Replace last two entries with normalised skill indices
        s[-2] = idx1_norm
        s[-1] = idx2_norm

        ball_vel = state[39:42]
        player_no = 1 if ball_vel[0] > 0 else 0

        X.append(s)
        if i < L - 2:
            Y1.append([0])
            Y2.append([0])
        elif i == L - 2:
            if player_no == 0:
                Y1.append([1])
                Y2.append([0])
            else:
                Y1.append([0])
                Y2.append([1])
        else:
            if player_no == 0:
                Y1.append([-1])
                Y2.append([0])
            else:
                Y1.append([0])
                Y2.append([-1])

X  = torch.tensor(np.array(X),  dtype=torch.float32)
Y1 = torch.tensor(np.array(Y1), dtype=torch.float32)
Y2 = torch.tensor(np.array(Y2), dtype=torch.float32)
print(f"Dataset: X={X.shape}, Y1={Y1.shape}, Y2={Y2.shape}")

# --------------------------------------------------------------------------- #
# Train Q-value models                                                         #
# --------------------------------------------------------------------------- #
model1 = SimpleModel(X.shape[1], [64, 32, 16], 1)
model2 = SimpleModel(X.shape[1], [64, 32, 16], 1)

opt1 = torch.optim.Adam(model1.parameters(), lr=LR)
opt2 = torch.optim.Adam(model2.parameters(), lr=LR)
criterion = nn.MSELoss()

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

with open(LOG_PATH_Q, "w", newline="") as q_csv:
    q_writer = csv.writer(q_csv)
    q_writer.writerow(["epoch", "loss1", "loss2"])

    for epoch in range(N_EPOCHS):
        opt1.zero_grad()
        opt2.zero_grad()
        loss1 = criterion(model1(X), Y1)
        loss2 = criterion(model2(X), Y2)
        loss1.backward()
        loss2.backward()
        opt1.step()
        opt2.step()
        q_writer.writerow([epoch + 1, loss1.item(), loss2.item()])
        if (epoch + 1) % 100 == 0:
            print(f"Q-epoch {epoch+1}/{N_EPOCHS}  loss1={loss1.item():.5f}  loss2={loss2.item():.5f}")

print(f"Q-value losses saved to {LOG_PATH_Q}")
torch.save(model1.state_dict(), "models/model1_5skill.pth")
torch.save(model2.state_dict(), "models/model2_5skill.pth")
print("Saved model1_5skill.pth and model2_5skill.pth")

# --------------------------------------------------------------------------- #
# Train potential function                                                     #
# --------------------------------------------------------------------------- #
model1.eval()
model2.eval()

model_p = SimpleModel(X.shape[1], [64, 32, 16], 1, last_layer_activation=None)
# Inherit batch-norm statistics from model1 (same as baseline)
model_p.batch_norm.running_mean = model1.batch_norm.running_mean.clone()
model_p.batch_norm.running_var  = model1.batch_norm.running_var.clone()
model_p.batch_norm.momentum = 0.0

opt_p = torch.optim.Adam(model_p.parameters(), lr=0.1)
min_loss = float("inf")

# Build all N_SKILLS^2 skill-pair tensors
skill_tensors = {}
for i, s1 in enumerate(SKILL_NAMES):
    for j, s2 in enumerate(SKILL_NAMES):
        Xij = X.clone()
        Xij[:, -2] = i / (N_SKILLS - 1)
        Xij[:, -1] = j / (N_SKILLS - 1)
        skill_tensors[(i, j)] = Xij

# Potential constraints: for each player, fixing the other player's skill,
# Φ(s, θ1, θ2) - Φ(s, θ1', θ2)  ≈  Q1(s, θ1, θ2) - Q1(s, θ1', θ2)
# (same for player 2)
pairs = []
for i in range(N_SKILLS):
    for j in range(N_SKILLS):
        for i2 in range(N_SKILLS):
            if i2 != i:
                # Player 1 deviates: (i,j) vs (i2,j)
                pairs.append(('p1', i, j, i2, j))
        for j2 in range(N_SKILLS):
            if j2 != j:
                # Player 2 deviates: (i,j) vs (i,j2)
                pairs.append(('p2', i, j, i, j2))

with open(LOG_PATH_P, "w", newline="") as p_csv:
    p_writer = csv.writer(p_csv)
    p_writer.writerow(["epoch", "potential_residual", "saved"])

    for epoch in range(N_EPOCHS):
        total_loss = 0.0
        sampled = random.sample(pairs, min(len(pairs), 20))  # subsample for speed
        for entry in sampled:
            opt_p.zero_grad()
            if entry[0] == 'p1':
                _, i, j, i2, _ = entry
                Xij  = skill_tensors[(i,  j)]
                Xi2j = skill_tensors[(i2, j)]
                with torch.no_grad():
                    target = model1(Xij) - model1(Xi2j)
                loss = criterion(model_p(Xij) - model_p(Xi2j), target)
            else:
                _, i, j, _, j2 = entry
                Xij  = skill_tensors[(i, j)]
                Xij2 = skill_tensors[(i, j2)]
                with torch.no_grad():
                    target = model2(Xij) - model2(Xij2)
                loss = criterion(model_p(Xij) - model_p(Xij2), target)
            loss.backward()
            opt_p.step()
            total_loss += loss.item()

        saved = 0
        if total_loss < min_loss:
            min_loss = total_loss
            torch.save(model_p.state_dict(), "models/model_p_5skill.pth")
            saved = 1
            print(f"  [saved] epoch {epoch+1}  potential_loss={total_loss:.5f}")
        p_writer.writerow([epoch + 1, total_loss, saved])
        if (epoch + 1) % 100 == 0:
            print(f"Phi-epoch {epoch+1}/{N_EPOCHS}  total_loss={total_loss:.5f}")

print(f"Potential residuals saved to {LOG_PATH_P}")
print("Done. Saved models/model_p_5skill.pth")
