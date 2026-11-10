"""
Train Q-value + potential models for Design B's attacker/defender
extension -- structurally identical to train_q_model_5skill_v3_one_hot.py
EXCEPT each player's action is a JOINT (attack_skill, defend_position)
pair (5x3=15 options, nash_skills.v2.joint_action.N_JOINT_ACTIONS) instead
of a single 5-way skill, over the 90-dim state
(nash_skills.v2.state_encoder_attack_defend.STATE_DIM_ATTACK_DEFEND).

This is a NEW, separate script -- does not modify
train_q_model_5skill_v3_one_hot.py or train_q_model_5skill_v3.py. Default
output paths (models/*_attack_defend.pth, logs/*_attack_defend.csv) never
collide with either.

Joint-action logic vs the one-hot trainer:
  - apply_joint_pair (below) decomposes each player's flat joint index
    into (attack_idx, defend_idx) via nash_skills.v2.joint_action, then
    delegates the one-hot write to that module's write_joint_action_pair
    -- the SAME shared function nash_skills.eval_matchup_attack_defend
    uses, so the two can never silently diverge on the convention.
  - The potential model's same-state counterfactual constraints iterate
    over N_JOINT_ACTIONS (15) instead of N_SKILLS (5) -- 15x15x14x2 =
    6,300 unilateral-full-joint-action-deviation constraints (vary one
    player's ENTIRE (attack,defend) choice, holding the other player's
    entirely fixed -- the paper's own unilateral-deviation definition).

UPDATE -- paper Section 4.1 alignment (none of this existed before; the
trainer fit and reported loss on 100% of the data, with no held-out set):
  - split_rallies(): RALLY-level train/held-out split. Paper: "Split data
    by independently generated rallies, so that correlated states from
    one rally do not appear in both training and evaluation sets." A
    state-level split would leak (consecutive decisions in one rally are
    highly correlated).
  - Held-out reporting, written to --heldout-out (JSON): Q-model MSE on
    held-out rallies (paper: "examine value-estimation error") and the
    distribution of held-out unilateral-change residuals |e| (paper Eq.
    potentialresidual). max|e| on held-out data is the empirical
    estimate of alpha in Proposition 1's (alpha + lambda) bound -- the
    paper's central theoretical claim needs exactly this number. p95 is
    reported alongside since a max over a finite sample is noisy.
  - phi_discrepancy_loss(kind="lse"): smooth version of the paper's own
    objective, Eq. potentialtraining ("minimize the largest sampled
    discrepancy"). kind="mse" (default) is the previous behavior, kept
    as the default because it is the objective already known to
    converge cleanly in the one-hot trainer; run both and compare
    held-out max|e| to see which actually gives the smaller empirical
    alpha.
  - --train-rally-limit: NESTED subsets of the training rallies (paper
    Sec 5.2 data-volume ablation -- a smaller M is always a prefix of a
    larger one for the same seed). --gamma: discount ablation (Sec 5.2).
    --seed: controls split, subsets, and torch init (multiple seeds).

Input data must come from collect_data_attack_defend.py (90-dim states,
encoder="attack_defend") -- ideally with --sampling grid for balanced
joint-action coverage (see that module's docstring).

Run:
    venv/bin/python nash_skills/v2/train_q_model_5skill_v3_attack_defend.py
    venv/bin/python nash_skills/v2/train_q_model_5skill_v3_attack_defend.py \\
        --phi-loss lse --seed 1 --train-rally-limit 200
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import copy
import csv
import json
import math
import random

import numpy as np
import pickle as pkl
import torch
import torch.nn as nn

from model_arch import SimpleModel
from nash_skills.v2.labeling import compute_returns, GAMMA
from nash_skills.v2.collect_data_attack_defend import _summarize_decision_balance
from nash_skills.v2.state_encoder_attack_defend import (
    ENCODER_DIMS, STATE_DIM_ATTACK_DEFEND as STATE_DIM, STATE_ENCODER_NAME, state_dim,
)
from nash_skills.v2.joint_action import (
    N_JOINT_ACTIONS, write_joint_pair_by_index, write_mixed_strategy_pair,
)
from nash_skills.v2.mixed_strategy import DEFAULT_P_VERTEX, sample_theta
from nash_skills.v2.phi_selection import PHI_SELECT_MODES, PHI_SELECT_STATS, PhiValidator, rally_metadata
from nash_skills.v2.state_symmetry import augment_rallies_with_swap

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
RALLY_PATH       = "data/rallies_attack_defend_grid.pkl"
N_EPOCHS         = 1500
LR               = 0.001
BATCH_SIZE       = 512
POTENTIAL_PAIRS  = 20
CHECKPOINT_EVERY = 100

DEFAULT_HOLDOUT_FRAC   = 0.2
DEFAULT_SEED           = 0
DEFAULT_LSE_BETA       = 10.0
PHI_LOSS_KINDS         = ("mse", "lse")
THETA_MODES            = ("pure", "mixed")
HELDOUT_N_CONSTRAINTS  = 400
Q_SELECT_MODES         = ("last", "val")
DEFAULT_VAL_FRAC       = 0.1
DEFAULT_PHI_VAL_EVERY  = 25
VAL_N_CONSTRAINTS      = 100

MODEL1_PATH  = "models/model1_attack_defend.pth"
MODEL2_PATH  = "models/model2_attack_defend.pth"
MODEL_P_PATH = "models/model_p_attack_defend.pth"
LOG_Q_PATH   = "logs/train_q_attack_defend.csv"
LOG_P_PATH   = "logs/train_p_attack_defend.csv"
HELDOUT_PATH = "logs/heldout_attack_defend.json"
# --------------------------------------------------------------------------- #


class _ConstantPhi(nn.Module):
    """Trivial potential (Phi = 0): its residuals are the value changes themselves --
    the baseline a learned Phi must beat."""

    def forward(self, x):
        return torch.zeros(x.shape[0], 1)


def split_rallies(rallies: list, holdout_frac: float, seed: int):
    """
    Rally-level (never state-level) split into (train, holdout). The
    order of the returned train list is a seeded permutation, which is
    what makes limit_rallies() give NESTED subsets. When there are at
    least 2 rallies and holdout_frac > 0, both sides are non-empty.
    """
    if not 0.0 <= holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in [0, 1), got {holdout_frac}")
    n = len(rallies)
    order = np.random.default_rng(seed).permutation(n)
    n_hold = int(round(n * holdout_frac))
    if holdout_frac > 0 and n >= 2:
        n_hold = min(max(n_hold, 1), n - 1)
    holdout = [rallies[i] for i in order[:n_hold]]
    train = [rallies[i] for i in order[n_hold:]]
    return train, holdout


def limit_rallies(rallies: list, limit):
    """First `limit` rallies (None = all) -- nested across limits."""
    if limit is None:
        return list(rallies)
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    return list(rallies[:limit])


def phi_discrepancy_loss(pred_diff: torch.Tensor, target_diff: torch.Tensor,
                         kind: str = "mse", beta: float = DEFAULT_LSE_BETA) -> torch.Tensor:
    """
    Loss over unilateral-change residuals e = pred_diff - target_diff.
      mse: mean(e^2).
      lse: (1/beta) * [logsumexp(beta*|e|) - log N] -- smooth max of |e|
           (paper Eq. potentialtraining). Equals c when every |e| = c;
           -> mean|e| as beta -> 0, -> max|e| as beta -> inf.
    |e| uses sqrt(e^2 + 1e-12) so the gradient stays finite at e = 0.
    """
    e = (pred_diff - target_diff).reshape(-1)
    if kind == "mse":
        return (e * e).mean()
    if kind == "lse":
        abs_e = torch.sqrt(e * e + 1e-12)
        return (torch.logsumexp(beta * abs_e, dim=0) - math.log(abs_e.numel())) / beta
    raise ValueError(f"Unknown phi loss kind '{kind}'. Choose from: {PHI_LOSS_KINDS}")


def residual_stats(residuals) -> dict:
    a = np.abs(np.asarray(residuals, dtype=float))
    return {
        "max_abs": float(a.max()),
        "mean_abs": float(a.mean()),
        "median_abs": float(np.median(a)),
        "p95_abs": float(np.percentile(a, 95)),
        "n": int(a.size),
    }


def build_constraints() -> list:
    """Every unilateral full-joint-action deviation: ("p1", i, j, i2, j)
    varies ego's joint action holding opp's fixed; ("p2", i, j, i, j2) the
    reverse."""
    constraints = []
    for i in range(N_JOINT_ACTIONS):
        for j in range(N_JOINT_ACTIONS):
            for i2 in range(N_JOINT_ACTIONS):
                if i2 != i:
                    constraints.append(("p1", i, j, i2, j))
            for j2 in range(N_JOINT_ACTIONS):
                if j2 != j:
                    constraints.append(("p2", i, j, i, j2))
    return constraints


def returns_with_horizon(n: int, winner, gamma: float, horizon: int) -> np.ndarray:
    """Player 1's continuation return G_H (paper Eq. 9) at each of a rally's n decisions: the
    point outcome after k more exchanges counts as gamma**k only when k < horizon, else 0."""
    r = {1: 1.0, 2: -1.0}.get(winner, 0.0)
    k = np.arange(n - 1, -1, -1)
    return np.where(k < horizon, r * gamma ** k, 0.0)


def build_dataset(rallies: list, gamma: float = GAMMA, horizon=None):
    """Convert rally list into (X, Y1, Y2) tensors using discounted returns
    (cut off after `horizon` exchanges when given, paper Eq. 9)."""
    x_list = []
    y1_list = []
    y2_list = []

    for entry in rallies:
        states = entry["states"]
        winner = entry.get("winner", 0)
        if len(states) == 0:
            continue

        if horizon is None:
            g1, g2 = compute_returns(states, gamma=gamma, winner=winner)
        else:
            g1 = returns_with_horizon(len(states), winner, gamma, horizon)
            g2 = -g1
        for state, v1, v2 in zip(states, g1, g2):
            x_list.append(state)
            y1_list.append([v1])
            y2_list.append([v2])

    if not x_list:
        empty = torch.zeros((0, STATE_DIM), dtype=torch.float32)
        return empty, torch.zeros((0, 1)), torch.zeros((0, 1))

    x = torch.tensor(np.array(x_list), dtype=torch.float32)
    y1 = torch.tensor(np.array(y1_list), dtype=torch.float32)
    y2 = torch.tensor(np.array(y2_list), dtype=torch.float32)
    return x, y1, y2


def apply_joint_pair(x: torch.Tensor, ego_joint: int, opp_joint: int) -> torch.Tensor:
    """
    Clone a base batch and overwrite all four one-hot blocks for a given
    pair of per-player JOINT (attack,defend) indices (0..14 each). Thin
    name-matching wrapper around the shared
    nash_skills.v2.joint_action.write_joint_pair_by_index.
    """
    return write_joint_pair_by_index(x, ego_joint, opp_joint)


def sample_base_batch(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Sample one shared base batch of states for same-state counterfactuals."""
    idx = random.sample(range(len(x)), min(batch_size, len(x)))
    return x[idx]


def _constraint_diffs(model_p, model1, model2, x_base, constraint):
    """(Phi change, V^i change) for one unilateral deviation on x_base."""
    if constraint[0] == "p1":
        _, i, j, i2, _ = constraint
        x_a = apply_joint_pair(x_base, i, j)
        x_b = apply_joint_pair(x_base, i2, j)
        value_model = model1
    else:
        _, i, j, _, j2 = constraint
        x_a = apply_joint_pair(x_base, i, j)
        x_b = apply_joint_pair(x_base, i, j2)
        value_model = model2
    with torch.no_grad():
        target = value_model(x_a) - value_model(x_b)
    return model_p(x_a) - model_p(x_b), target


def unilateral_residuals(model_p, model1, model2, x: torch.Tensor, constraints: list,
                         rng: np.random.Generator, n_constraints: int, batch_size: int) -> np.ndarray:
    """
    Paper Eq. potentialresidual evaluated on the given states (pass the
    HELD-OUT states for an honest estimate): for sampled unilateral
    deviations, e = [Phi(x,a) - Phi(x,a')] - [V^i(x,a) - V^i(x,a')],
    with the fitted Q models as V^i. Callers must put all three models
    in eval() mode first (this function does not change their mode).
    """
    picked = rng.choice(len(constraints), size=min(n_constraints, len(constraints)), replace=False)
    out = []
    with torch.no_grad():
        for k in picked:
            idx = rng.choice(len(x), size=min(batch_size, len(x)), replace=False)
            pred, target = _constraint_diffs(model_p, model1, model2, x[idx], constraints[k])
            out.append((pred - target).reshape(-1).cpu().numpy())
    return np.concatenate(out)


def sample_theta_deviation(rng: np.random.Generator, p_vertex: float = DEFAULT_P_VERTEX):
    """
    One unilateral deviation in the compact family (paper Eq.
    potentialresidual with theta^{i'} from Delta_5 x Delta_3):
    (player, theta1, theta2, theta_alt) -- theta_alt replaces `player`'s
    theta, the other player's is held fixed. Vertices are included with
    probability p_vertex, so pure-action deviations are a special case.
    """
    player = 1 if rng.random() < 0.5 else 2
    return player, sample_theta(rng, p_vertex), sample_theta(rng, p_vertex), sample_theta(rng, p_vertex)


def _theta_diffs(model_p, model1, model2, x_base, deviation):
    """(Phi change, V^i change) for one theta deviation on x_base."""
    player, th1, th2, th_alt = deviation
    x_a = write_mixed_strategy_pair(x_base, th1[0], th1[1], th2[0], th2[1])
    if player == 1:
        x_b = write_mixed_strategy_pair(x_base, th_alt[0], th_alt[1], th2[0], th2[1])
        value_model = model1
    else:
        x_b = write_mixed_strategy_pair(x_base, th1[0], th1[1], th_alt[0], th_alt[1])
        value_model = model2
    with torch.no_grad():
        target = value_model(x_a) - value_model(x_b)
    return model_p(x_a) - model_p(x_b), target


def theta_unilateral_residuals(model_p, model1, model2, x: torch.Tensor, rng: np.random.Generator,
                               n_samples: int, batch_size: int,
                               p_vertex: float = DEFAULT_P_VERTEX) -> np.ndarray:
    """theta-mode counterpart of unilateral_residuals (same eval()-mode
    contract): residuals for n_samples random theta deviations."""
    out = []
    with torch.no_grad():
        for _ in range(n_samples):
            idx = rng.choice(len(x), size=min(batch_size, len(x)), replace=False)
            pred, target = _theta_diffs(model_p, model1, model2, x[idx],
                                        sample_theta_deviation(rng, p_vertex))
            out.append((pred - target).reshape(-1).cpu().numpy())
    return np.concatenate(out)


def output_paths(out_dir, heldout_out: str = HELDOUT_PATH) -> dict:
    """Where train() writes; with out_dir set, every output goes there by basename."""
    paths = {"model1": MODEL1_PATH, "model2": MODEL2_PATH, "model_p": MODEL_P_PATH,
             "log_q": LOG_Q_PATH, "log_p": LOG_P_PATH, "heldout": heldout_out}
    if out_dir is not None:
        paths = {k: os.path.join(out_dir, os.path.basename(v)) for k, v in paths.items()}
    return paths


def rally_encoder(rallies: list) -> str:
    """The one state encoder every rally was collected with (it fixes the state size)."""
    encoders = {r.get("encoder") for r in rallies}
    if len(encoders) > 1:
        raise ValueError(f"Rallies must share one state encoder, found {sorted(map(str, encoders))}")
    encoder = next(iter(encoders), STATE_ENCODER_NAME)
    if encoder not in ENCODER_DIMS:
        raise ValueError(
            f"Rallies were collected with encoder={encoder!r}, not one of {sorted(ENCODER_DIMS)}. Recollect with "
            f"'collect_data_attack_defend.py' -- a legacy or one-hot pickle cannot be "
            f"reused here, the defend-position choices needed to re-encode were never saved.")
    return encoder


def train_q_models(x, y1, y2, n_epochs, lr, log_path, val_data=None, val_every=DEFAULT_PHI_VAL_EVERY):
    """Full-batch Q1/Q2 training. With val_data=(x_val, y1_val, y2_val), keeps the
    epoch with the lowest summed validation MSE (checked every val_every epochs and
    at the end) instead of the last epoch. Returns (model1, model2, epoch, val_mse)."""
    criterion = nn.MSELoss()
    model1 = SimpleModel(x.shape[1], [64, 32, 16], 1)
    model2 = SimpleModel(x.shape[1], [64, 32, 16], 1)
    opt1 = torch.optim.Adam(model1.parameters(), lr=lr)
    opt2 = torch.optim.Adam(model2.parameters(), lr=lr)
    best = {"epoch": n_epochs, "mse": None, "state": None}
    print(f"\nTraining Q-value models ({n_epochs} epochs"
          + (", early stopping on validation)" if val_data is not None else ") ..."))
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss1", "loss2", "val_mse"])
        for epoch in range(n_epochs):
            model1.train()
            model2.train()
            opt1.zero_grad()
            opt2.zero_grad()
            loss1 = criterion(model1(x), y1)
            loss2 = criterion(model2(x), y2)
            loss1.backward()
            loss2.backward()
            opt1.step()
            opt2.step()
            val_mse = ""
            if val_data is not None and ((epoch + 1) % val_every == 0 or epoch + 1 == n_epochs):
                model1.eval()
                model2.eval()
                with torch.no_grad():
                    val_mse = float(criterion(model1(val_data[0]), val_data[1]).item()
                                    + criterion(model2(val_data[0]), val_data[2]).item()) / 2
                if best["mse"] is None or val_mse < best["mse"]:
                    best = {"epoch": epoch + 1, "mse": val_mse,
                            "state": (copy.deepcopy(model1.state_dict()), copy.deepcopy(model2.state_dict()))}
            w.writerow([epoch + 1, loss1.item(), loss2.item(), val_mse])
            if (epoch + 1) % CHECKPOINT_EVERY == 0:
                print(f"  Q-epoch {epoch+1}/{n_epochs}  loss1={loss1.item():.5f}  loss2={loss2.item():.5f}"
                      + (f"  best val MSE {best['mse']:.5f} @ {best['epoch']}" if best["mse"] is not None else ""))
    if best["state"] is not None:
        model1.load_state_dict(best["state"][0])
        model2.load_state_dict(best["state"][1])
    model1.eval()
    model2.eval()
    return model1, model2, best["epoch"], best["mse"]


def train(rally_path: str, n_epochs: int, lr: float, *,
          holdout_frac: float = DEFAULT_HOLDOUT_FRAC, seed: int = DEFAULT_SEED,
          gamma: float = GAMMA, train_rally_limit=None,
          phi_loss: str = "mse", phi_lse_beta: float = DEFAULT_LSE_BETA,
          theta_mode: str = "pure", p_vertex: float = DEFAULT_P_VERTEX,
          heldout_out: str = HELDOUT_PATH, out_dir=None,
          phi_select: str = "train_loss", val_frac: float = DEFAULT_VAL_FRAC,
          phi_val_every: int = DEFAULT_PHI_VAL_EVERY, phi_select_stat: str = "p95",
          q_select: str = "last", augment_swap: bool = False, horizon=None,
          pure_data_only: bool = False) -> dict:
    paths = output_paths(out_dir, heldout_out)
    if q_select not in Q_SELECT_MODES:
        raise ValueError(f"Unknown Q selection '{q_select}'. Choose from: {Q_SELECT_MODES}")
    if phi_select not in PHI_SELECT_MODES:
        raise ValueError(f"Unknown phi selection '{phi_select}'. Choose from: {PHI_SELECT_MODES}")
    if phi_loss not in PHI_LOSS_KINDS:
        raise ValueError(f"Unknown phi loss kind '{phi_loss}'. Choose from: {PHI_LOSS_KINDS}")
    if theta_mode not in THETA_MODES:
        raise ValueError(f"Unknown theta mode '{theta_mode}'. Choose from: {THETA_MODES}")
    if pure_data_only and theta_mode != "pure":
        raise ValueError("pure_data_only trains the fixed-pair policy family (paper Eq. 7): use theta_mode='pure'")
    random.seed(seed)
    torch.manual_seed(seed)
    theta_rng = np.random.default_rng(seed)

    print(f"Loading {rally_path} ...")
    with open(rally_path, "rb") as f:
        rallies = pkl.load(f)
    print(f"  {len(rallies)} rallies loaded")

    encoder = rally_encoder(rallies)
    dim = state_dim(encoder)
    if theta_mode == "mixed" and not any(r.get("theta_mode") == "mixed" for r in rallies):
        raise ValueError(
            "theta_mode='mixed' needs data collected with collect_data_attack_defend.py "
            "--sampling mixed: these rallies only contain pure joint actions, so the value "
            "models would never see an interior theta and every theta-deviation target "
            "would be pure extrapolation."
        )

    if pure_data_only:
        # Paper Eq. 7: each continuation holds one fixed (target, preparation) pair per player.
        n_all = len(rallies)
        rallies = [r for r in rallies if r.get("theta_mode") != "mixed"]
        print(f"  Fixed-pair rallies only: {len(rallies)}/{n_all}")
    metadata = rally_metadata(rallies)
    print(f"  Collection settings: {metadata}")
    train_rallies, holdout_rallies = split_rallies(rallies, holdout_frac, seed)
    val_rallies = []
    if phi_select == "val_residual" or q_select == "val":
        # Carved before the volume limit, so every data-volume subset is
        # selected on the same validation rallies.
        train_rallies, val_rallies = split_rallies(train_rallies, val_frac, seed + 1)
    train_rallies = limit_rallies(train_rallies, train_rally_limit)
    if augment_swap:
        # After the split and the volume limit: held-out and validation rallies stay unseen.
        train_rallies = augment_rallies_with_swap(train_rallies)
    print(f"  Rally-level split (seed={seed}): {len(train_rallies)} train, "
          f"{len(val_rallies)} validation, {len(holdout_rallies)} held-out"
          + (f" (train limited to {train_rally_limit})" if train_rally_limit else ""))

    balance = _summarize_decision_balance(train_rallies)
    n_possible = N_JOINT_ACTIONS * N_JOINT_ACTIONS
    if balance:
        counts = list(balance.values())
        ratio = max(counts) / min(counts)
        print(f"  Train decision-level joint-action coverage: {len(balance)}/{n_possible} combos, "
              f"min={min(counts)} max={max(counts)} ratio={ratio:.2f}")
    else:
        print("  WARNING: no decisions found in training set")

    x, y1, y2 = build_dataset(train_rallies, gamma=gamma, horizon=horizon)
    x_hold, y1_hold, y2_hold = build_dataset(holdout_rallies, gamma=gamma, horizon=horizon)
    print(f"  Train X={tuple(x.shape)}, held-out X={tuple(x_hold.shape)}")
    if x.shape[1] != dim:
        raise ValueError(f"Expected {dim}-dim states (encoder {encoder}), got {x.shape[1]}")
    if len(x) < 2:
        raise ValueError("Need at least 2 training states (BatchNorm needs a batch).")

    n_nonzero = (y1.abs() > 1e-6).sum().item()
    print(f"  Non-zero Y1: {n_nonzero}/{len(y1)} = {100*n_nonzero/len(y1):.1f}%")
    print(f"  gamma={gamma}, encoder={encoder} ({dim} dims), N_JOINT_ACTIONS={N_JOINT_ACTIONS}, "
          f"phi_loss={phi_loss}" + (f" (beta={phi_lse_beta})" if phi_loss == "lse" else "")
          + f", theta_mode={theta_mode}" + (f" (p_vertex={p_vertex})" if theta_mode == "mixed" else ""))

    for path in paths.values():
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    criterion = nn.MSELoss()

    # ------------------------------------------------------------------ #
    # Train Q-value models                                                 #
    # ------------------------------------------------------------------ #
    val_data = build_dataset(val_rallies, gamma=gamma, horizon=horizon) if q_select == "val" else None
    model1, model2, q_best_epoch, q_val_mse = train_q_models(
        x, y1, y2, n_epochs, lr, paths["log_q"], val_data=val_data, val_every=phi_val_every)
    torch.save(model1.state_dict(), paths["model1"])
    torch.save(model2.state_dict(), paths["model2"])
    print(f"  Saved {paths['model1']} and {paths['model2']} (epoch {q_best_epoch}"
          + (f", validation MSE {q_val_mse:.4f})" if q_val_mse is not None else ")"))

    # ------------------------------------------------------------------ #
    # Train potential model                                                #
    # ------------------------------------------------------------------ #
    model1.eval()
    model2.eval()

    model_p = SimpleModel(dim, [64, 32, 16], 1, last_layer_activation=None)
    model_p.batch_norm.running_mean = model1.batch_norm.running_mean.clone()
    model_p.batch_norm.running_var = model1.batch_norm.running_var.clone()
    model_p.batch_norm.momentum = 0.0
    model_p.eval()

    opt_p = torch.optim.Adam(model_p.parameters(), lr=lr)
    scheduler_p = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_p, T_max=n_epochs, eta_min=1e-5
    )
    min_loss = float("inf")
    constraints = build_constraints()
    validator = None
    best_epoch, best_val = None, float("inf")
    if phi_select == "val_residual":
        x_val, _, _ = build_dataset(val_rallies, gamma=gamma, horizon=horizon)
        if len(x_val) == 0:
            raise ValueError("phi_select='val_residual' needs validation states; raise val_frac.")
        validator = PhiValidator(x_val, constraints, theta_mode, seed=seed + 7,
                                 n_constraints=VAL_N_CONSTRAINTS, batch_size=BATCH_SIZE,
                                 stat=phi_select_stat, p_vertex=p_vertex)

    if theta_mode == "pure":
        print(f"\nTraining potential model ({n_epochs} epochs, {len(constraints)} pure constraints) ...")
    else:
        print(f"\nTraining potential model ({n_epochs} epochs, {POTENTIAL_PAIRS} random "
              f"theta deviations/epoch) ...")
    with open(paths["log_p"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss", "saved"])

        for epoch in range(n_epochs):
            preds, targets = [], []
            if theta_mode == "pure":
                for c in random.sample(constraints, min(len(constraints), POTENTIAL_PAIRS)):
                    pred, target = _constraint_diffs(model_p, model1, model2,
                                                     sample_base_batch(x, BATCH_SIZE), c)
                    preds.append(pred)
                    targets.append(target)
            else:
                for _ in range(POTENTIAL_PAIRS):
                    pred, target = _theta_diffs(model_p, model1, model2,
                                                sample_base_batch(x, BATCH_SIZE),
                                                sample_theta_deviation(theta_rng, p_vertex))
                    preds.append(pred)
                    targets.append(target)
            # One loss over the WHOLE sampled set S this epoch -- for "lse"
            # this is what makes it a max over every sampled discrepancy,
            # not a max within each constraint separately.
            loss = phi_discrepancy_loss(torch.cat(preds), torch.cat(targets),
                                        kind=phi_loss, beta=phi_lse_beta)
            opt_p.zero_grad()
            loss.backward()
            opt_p.step()
            scheduler_p.step()

            saved = 0
            if validator is not None:
                if (epoch + 1) % phi_val_every == 0 or epoch + 1 == n_epochs:
                    val_score = validator.score(model_p, model1, model2)
                    if val_score < best_val:
                        best_val, best_epoch = val_score, epoch + 1
                        torch.save(model_p.state_dict(), paths["model_p"])
                        saved = 1
                        print(f"  [saved] Phi-epoch {epoch+1}  val {phi_select_stat}|e|={val_score:.5f}")
            elif loss.item() < min_loss:
                min_loss = loss.item()
                best_epoch = epoch + 1
                torch.save(model_p.state_dict(), paths["model_p"])
                saved = 1
                if (epoch + 1) % CHECKPOINT_EVERY == 0 or epoch < 10:
                    print(f"  [saved] Phi-epoch {epoch+1}  loss={loss.item():.5f}")

            w.writerow([epoch + 1, loss.item(), saved])
            if (epoch + 1) % CHECKPOINT_EVERY == 0 and not saved:
                print(f"  Phi-epoch {epoch+1}/{n_epochs}  loss={loss.item():.5f}")

    print(f"\nSaved {paths['model_p']}")

    # ------------------------------------------------------------------ #
    # Held-out evaluation (paper Sec 4.1 / 5.1)                            #
    # ------------------------------------------------------------------ #
    model_p.load_state_dict(torch.load(paths["model_p"]))
    model_p.eval()
    report = {
        "encoder": encoder, "seed": seed, "gamma": gamma, "phi_loss": phi_loss,
        "phi_lse_beta": phi_lse_beta if phi_loss == "lse" else None,
        "theta_mode": theta_mode, "p_vertex": p_vertex if theta_mode == "mixed" else None,
        "n_train_rallies": len(train_rallies), "n_holdout_rallies": len(holdout_rallies),
        "n_train_states": int(len(x)), "n_holdout_states": int(len(x_hold)),
        "phi_select": phi_select, "phi_select_stat": phi_select_stat if validator else None,
        "n_val_rallies": len(val_rallies), "phi_best_epoch": best_epoch,
        "phi_val_score": best_val if validator else None,
        "q_select": q_select, "q_best_epoch": q_best_epoch, "q_val_mse": q_val_mse,
        "augment_swap": augment_swap, "horizon": horizon, "pure_data_only": pure_data_only,
        **metadata,
    }

    def _residuals(states, phi=None):
        phi = model_p if phi is None else phi
        rng = np.random.default_rng(seed)
        if theta_mode == "pure":
            return unilateral_residuals(phi, model1, model2, states, constraints, rng,
                                        n_constraints=HELDOUT_N_CONSTRAINTS, batch_size=BATCH_SIZE)
        return theta_unilateral_residuals(phi, model1, model2, states, rng,
                                          n_samples=HELDOUT_N_CONSTRAINTS, batch_size=BATCH_SIZE,
                                          p_vertex=p_vertex)

    if len(x_hold) > 0:
        with torch.no_grad():
            report["heldout_q1_mse"] = float(criterion(model1(x_hold), y1_hold).item())
            report["heldout_q2_mse"] = float(criterion(model2(x_hold), y2_hold).item())
        report["heldout_residuals"] = residual_stats(_residuals(x_hold))
        report["heldout_q1_mse_predict_zero"] = float((y1_hold ** 2).mean().item())
        report["heldout_residuals_constant_phi"] = residual_stats(_residuals(x_hold, _ConstantPhi()))
        report["train_residuals"] = residual_stats(_residuals(x))
        h = report["heldout_residuals"]
        print(f"\nHeld-out value error: Q1 MSE={report['heldout_q1_mse']:.4f}  "
              f"Q2 MSE={report['heldout_q2_mse']:.4f}")
        print(f"Held-out unilateral residuals |e| (empirical alpha): "
              f"max={h['max_abs']:.4f}  p95={h['p95_abs']:.4f}  median={h['median_abs']:.4f}  "
              f"mean={h['mean_abs']:.4f}  "
              f"(train max={report['train_residuals']['max_abs']:.4f})")
    else:
        print("\nNo held-out rallies -- skipping held-out evaluation (holdout_frac=0?).")

    with open(paths["heldout"], "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved held-out report to {paths['heldout']}")
    print("Done.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Design B attacker/defender Q-value and per-sample potential models"
    )
    parser.add_argument("--rallies", default=RALLY_PATH,
                        help=f"Rally pickle path (default: {RALLY_PATH})")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS,
                        help=f"Training epochs (default: {N_EPOCHS})")
    parser.add_argument("--lr", type=float, default=LR,
                        help=f"Learning rate (default: {LR})")
    parser.add_argument("--holdout-frac", type=float, default=DEFAULT_HOLDOUT_FRAC,
                        help="Fraction of RALLIES (not states) held out for evaluation.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Seed for the split, nested subsets, and torch init.")
    parser.add_argument("--gamma", type=float, default=GAMMA,
                        help=f"Discount for return labels (default: labeling.GAMMA={GAMMA}).")
    parser.add_argument("--train-rally-limit", type=int, default=None,
                        help="Use only the first M training rallies (nested data-volume ablation).")
    parser.add_argument("--phi-loss", choices=PHI_LOSS_KINDS, default="mse",
                        help="'mse' (previous behavior) or 'lse' (smooth max-discrepancy, paper Eq. potentialtraining).")
    parser.add_argument("--phi-lse-beta", type=float, default=DEFAULT_LSE_BETA,
                        help="Sharpness of the 'lse' loss (larger -> closer to a hard max).")
    parser.add_argument("--theta-mode", choices=THETA_MODES, default="pure",
                        help="'pure' (deviations between pure joint actions) or 'mixed' "
                             "(deviations within Delta_5 x Delta_3, paper Sec 4.1; needs "
                             "data from collect_data_attack_defend.py --sampling mixed).")
    parser.add_argument("--p-vertex", type=float, default=DEFAULT_P_VERTEX,
                        help="--theta-mode mixed only: probability a sampled theta is a vertex.")
    parser.add_argument("--heldout-out", default=HELDOUT_PATH,
                        help=f"Held-out report JSON path (default: {HELDOUT_PATH}).")
    parser.add_argument("--out-dir", default=None,
                        help="Write models, loss logs and the held-out report here "
                             "instead of models/ and logs/ (keeps parallel runs apart).")
    parser.add_argument("--phi-select", choices=PHI_SELECT_MODES, default="train_loss",
                        help="'train_loss' (legacy: lowest sampled training loss) or 'val_residual' "
                             "(lowest validation |e| statistic, on rallies carved from the train split)")
    parser.add_argument("--phi-select-stat", choices=PHI_SELECT_STATS, default="p95")
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                        help="--phi-select val_residual: fraction of training rallies used for selection")
    parser.add_argument("--phi-val-every", type=int, default=DEFAULT_PHI_VAL_EVERY,
                        help="validation check interval (epochs) for --phi-select val_residual and --q-select val")
    parser.add_argument("--q-select", choices=Q_SELECT_MODES, default="last",
                        help="'last' (legacy: Q after the final epoch) or 'val' (early stopping on the "
                             "validation split carved from the training rallies)")
    parser.add_argument("--horizon", type=int, default=None,
                        help="continuation length H in exchanges for the value targets (paper Eq. 9); default: whole rally")
    parser.add_argument("--pure-data-only", action="store_true",
                        help="train only on fixed-pair rallies (paper Eq. 7 policy family; needs --theta-mode pure)")
    parser.add_argument("--augment-swap", action="store_true",
                        help="add the player-swapped copy of every training rally (nash_skills.v2.state_symmetry)")
    args = parser.parse_args()

    train(args.rallies, args.epochs, args.lr, augment_swap=args.augment_swap, horizon=args.horizon,
          pure_data_only=args.pure_data_only,
          phi_select=args.phi_select, phi_select_stat=args.phi_select_stat, q_select=args.q_select,
          val_frac=args.val_frac, phi_val_every=args.phi_val_every,
          holdout_frac=args.holdout_frac, seed=args.seed, gamma=args.gamma,
          train_rally_limit=args.train_rally_limit,
          phi_loss=args.phi_loss, phi_lse_beta=args.phi_lse_beta,
          theta_mode=args.theta_mode, p_vertex=args.p_vertex,
          heldout_out=args.heldout_out, out_dir=args.out_dir)
