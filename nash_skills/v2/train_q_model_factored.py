"""
Ablation: FactoredModel vs SimpleModel for 2-skill Q/potential training.
Output: models/model1_factored.pth, model2_factored.pth, model_p_factored.pth
Run: python train_q_model_factored.py --gamma 0.9
"""
import sys, os
# Walk up to the directory that contains model_arch.py so the script works
# whether it lives at project root or under nash_skills/v2/.
_here = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_here, os.path.join(_here, ".."), os.path.join(_here, "../..")):
    if os.path.isfile(os.path.join(_candidate, "model_arch.py")):
        sys.path.insert(0, os.path.abspath(_candidate))
        break
import argparse, csv
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
from model_arch import FactoredModel
from nash_skills.v2.labeling import compute_returns, GAMMA

DEFAULT_RALLY_PATH = "data/rallies_v2_2skill.pkl"
N_EPOCHS = 1500
LR = 0.001
SKILL_DIM = 2
MODEL1_PATH = "models/model1_factored.pth"
MODEL2_PATH = "models/model2_factored.pth"
MODEL_P_PATH = "models/model_p_factored.pth"
LOG_Q_PATH = "logs/train_q_factored.csv"
LOG_P_PATH = "logs/train_p_factored.csv"
CHECKPOINT_EVERY = 100

def build_dataset(rallies, gamma):
    X, Y1, Y2 = [], [], []
    for e in rallies:
        if not e["states"]: continue
        g1, g2 = compute_returns(e["states"], gamma=gamma, winner=e.get("winner", 0))
        for s, v1, v2 in zip(e["states"], g1, g2):
            X.append(s); Y1.append(v1); Y2.append(v2)
    return (torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(Y1), dtype=torch.float32),
            torch.tensor(np.array(Y2), dtype=torch.float32))

def train(rally_path=DEFAULT_RALLY_PATH, n_epochs=N_EPOCHS, lr=LR, gamma=GAMMA):
    print(f"Loading {rally_path}")
    rallies = pkl.load(open(rally_path, "rb"))
    X, Y1f, Y2f = build_dataset(rallies, gamma)
    Y1, Y2 = Y1f.unsqueeze(1), Y2f.unsqueeze(1)
    state_dim = X.shape[1] - SKILL_DIM
    print(f"  {len(rallies)} rallies, state_dim={state_dim}, gamma={gamma}")
    os.makedirs("models", exist_ok=True); os.makedirs("logs", exist_ok=True)
    crit = nn.MSELoss()

    # Q models
    m1 = FactoredModel(state_dim); m2 = FactoredModel(state_dim)
    o1 = torch.optim.Adam(m1.parameters(), lr=lr)
    o2 = torch.optim.Adam(m2.parameters(), lr=lr)
    print(f"\n[Factored] Training Q models ({n_epochs} epochs)...")
    with open(LOG_Q_PATH, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch","loss1","loss2"])
        for ep in range(n_epochs):
            o1.zero_grad(); o2.zero_grad()
            l1 = crit(m1(X), Y1); l2 = crit(m2(X), Y2)
            l1.backward(); l2.backward()
            o1.step(); o2.step()
            w.writerow([ep+1, l1.item(), l2.item()])
            if (ep+1) % CHECKPOINT_EVERY == 0:
                print(f"  Q-epoch {ep+1}/{n_epochs} loss1={l1.item():.5f} loss2={l2.item():.5f}")
    torch.save(m1.state_dict(), MODEL1_PATH)
    torch.save(m2.state_dict(), MODEL2_PATH)
    print(f"  Saved {MODEL1_PATH}, {MODEL2_PATH}")

    # Potential model
    m1.eval(); m2.eval()
    mp = FactoredModel(state_dim, last_layer_activation=None)
    mp.batch_norm.running_mean = m1.batch_norm.running_mean.clone()
    mp.batch_norm.running_var  = m1.batch_norm.running_var.clone()
    mp.batch_norm.momentum = 0.0; mp.eval()
    op = torch.optim.Adam(mp.parameters(), lr=lr)

    X11=X.clone(); X11[:,-2]=1.;  X11[:,-1]=1.
    X01=X.clone(); X01[:,-2]=-1.; X01[:,-1]=1.
    X10=X.clone(); X10[:,-2]=1.;  X10[:,-1]=-1.
    X00=X.clone(); X00[:,-2]=-1.; X00[:,-1]=-1.

    min_loss = float("inf")
    print(f"\n[Factored] Training potential model ({n_epochs} epochs)...")
    with open(LOG_P_PATH, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch","total_loss","saved"])
        for ep in range(n_epochs):
            tl = 0.0
            with torch.no_grad():
                t = [m1(X11).mean()-m1(X01).mean(), m1(X10).mean()-m1(X00).mean(),
                     m2(X11).mean()-m2(X10).mean(), m2(X01).mean()-m2(X00).mean()]
            for (Xa, Xb), tgt in zip([(X11,X01),(X10,X00),(X11,X10),(X01,X00)], t):
                op.zero_grad()
                lp = crit(mp(Xa).mean()-mp(Xb).mean(), tgt)
                lp.backward(); op.step(); tl += lp.item()
            saved = 0
            if tl < min_loss:
                min_loss = tl; torch.save(mp.state_dict(), MODEL_P_PATH); saved = 1
                if (ep+1) % CHECKPOINT_EVERY == 0:
                    print(f"  [saved] Phi-epoch {ep+1} loss={tl:.5f}")
            w.writerow([ep+1, tl, saved])
            if (ep+1) % CHECKPOINT_EVERY == 0 and not saved:
                print(f"  Phi-epoch {ep+1}/{n_epochs} total_loss={tl:.5f}")
    print(f"\nSaved {MODEL_P_PATH}\nDone!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rallies", default=DEFAULT_RALLY_PATH)
    p.add_argument("--epochs",  type=int,   default=N_EPOCHS)
    p.add_argument("--lr",      type=float, default=LR)
    p.add_argument("--gamma",   type=float, default=GAMMA)
    a = p.parse_args()
    train(a.rallies, a.epochs, a.lr, a.gamma)