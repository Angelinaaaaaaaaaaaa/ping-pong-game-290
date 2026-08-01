"""
Alternating (best-response) self-play for the 5-skill meta-policy.

Differs from selfplay_5skill.py (simultaneous self-play):
  - Two separate policy networks: π_A and π_B.
  - Phase A: π_A is trained (gradients), π_B is frozen. π_A plays as ego, π_B as opp.
  - Phase B: π_B is trained, π_A is frozen. π_B plays as ego, π_A as opp.
  - Alternate every `--phase-length` iters.

This matches iterative best-response: each phase finds π_X ≈ argmax E[reward(π_X, π_other_frozen)].
Compared to simultaneous self-play, this avoids both players moving together
(which can produce cycling) and gives clearer "ego learns to beat fixed opp"
training curves.

Run:
    PYTHONPATH=. python nash_skills/v2/selfplay_5skill_alt.py \
        --total-iterations 400 --phase-length 50 --rallies-per-iter 32 \
        --entropy-coef 0.05 \
        --output-prefix models/selfplay_5skill_alt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import csv
import io
import random
from contextlib import redirect_stdout
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO

from nash_skills.env_wrapper import SkillEnv
from nash_skills.skills import SKILL_NAMES, N_SKILLS, skill_from_index
from nash_skills.v2.state_encoder import encode_ego, encode_opp, STATE_DIM
from nash_skills.winner_inference import infer_terminal_winner

# --------------------------------------------------------------------------- #
PPO_MODEL_PATH      = "logs/best_model_tracker1/best_model"
HISTORY             = 4
TABLE_SHIFT         = 1.5
MAX_STEPS_PER_RALLY = 800
# Per-crossing shaping bonus. Two separate coefficients so truncated (stall)
# rallies can't out-earn a decisive win -- matches the fix already applied to
# selfplay_5skill.py: at 50 crossings a truncated rally accumulates
# 0.001 * 50 = 0.05, well below the +/-1 terminal signal.
SHAPING_COEF           = 0.005
TRUNCATED_SHAPING_COEF = 0.001
# --------------------------------------------------------------------------- #


class MetaPolicy(nn.Module):
    """Same architecture as selfplay_5skill.MetaPolicy (76 -> 64 -> 32 -> 5 logits)."""

    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_SKILLS, hidden=(64, 32)):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, n_actions)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def sample(self, state_np: np.ndarray, device):
        x = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
        logits = self.forward(x)[0]
        probs = F.softmax(logits, dim=-1)
        idx = int(torch.multinomial(probs, 1).item())
        return idx, probs.detach().cpu().numpy()


def _build_ppo_obs(obs, info, player):
    o = np.zeros(9 + 9 + 7 + 7 + 9 * HISTORY, dtype=np.float32)
    if player == 1:
        o[:9]    = obs[:9]
        o[9:18]  = obs[18:27]
        o[18:21] = info["diff_pos"]
        o[21:25] = info["diff_quat"]
        o[25:32] = info["target"]
        o[32:]   = obs[42: 42 + HISTORY * 9]
    else:
        o[:9]    = obs[9:18]
        o[9:18]  = obs[27:36]
        o[18:21] = info["diff_pos_opp"]
        o[21:25] = info["diff_quat_opp"]
        o[25:32] = info["target_opp"]
        o[32:]   = obs[42 + HISTORY * 9: 42 + 2 * HISTORY * 9]
    return o


def _swallow_step(env, action):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return env.step(action)


def classify_outcome(ego_terminal: float) -> str:
    """
    Classify a rally's raw (un-shaped) terminal reward into 'win'/'draw'/'loss'.

    Operates on ego_terminal directly, never on the shaping-inflated
    ego_reward -- a long truncated rally's accumulated shaping bonus can
    never be misread as a decisive outcome this way. Matches the thresholds
    selfplay_5skill.py already uses (ego_terminal = ego_r - opp_r, which
    recovers 2 * the raw terminal since shaping cancels out there).
    """
    if ego_terminal > 0.5:
        return "win"
    elif abs(ego_terminal) < 0.5:
        return "draw"
    else:
        return "loss"


def play_one_rally(env, ppo, trainer_policy, frozen_policy, device,
                   ego_init_idx=0, opp_init_idx=0):
    """
    Trainer plays as ego (gradients flow), frozen plays as opp (no grad).

    Returns:
        ego_log_probs    : list of trainer's log π(a|s) at each decision
        ego_entropies    : list of trainer's entropy at each decision
        ego_reward       : float (with shaping + terminal)
        ego_terminal     : float -- the raw, un-shaped outcome: +1.0 (win),
                            -1.0 (loss), 0.0 (truncated). Callers should use
                            this (via classify_outcome) rather than trying to
                            infer the outcome from ego_reward, since
                            ego_reward also contains the shaping term.
        steps            : int
    """
    env.set_skills(skill_from_index(ego_init_idx), skill_from_index(opp_init_idx))
    obs, info = env.reset()
    prev_ball_x = float(obs[36])
    ego_idx, opp_idx = ego_init_idx, opp_init_idx

    ego_log_probs = []
    ego_entropies = []

    crossings = 0
    steps = 0
    while True:
        ppo1 = _build_ppo_obs(obs, info, 1)
        ppo2 = _build_ppo_obs(obs, info, 2)
        a1, _ = ppo.predict(ppo1, deterministic=True)
        a2, _ = ppo.predict(ppo2, deterministic=True)
        action = np.zeros(18, dtype=np.float32)
        action[:9] = a1[:9]; action[9:] = a2[:9]

        obs, _r, done, _t, info = _swallow_step(env, action)
        steps += 1
        curr_ball_x = float(obs[36])

        if (prev_ball_x - TABLE_SHIFT) * (curr_ball_x - TABLE_SHIFT) < 0:
            crossings += 1
            ego_state = encode_ego(obs, info)
            opp_state = encode_opp(obs, info)

            # Ego = trainer (gradients flow)
            ego_logits = trainer_policy(
                torch.from_numpy(ego_state).float().unsqueeze(0).to(device))[0]
            ego_dist = torch.distributions.Categorical(logits=ego_logits)
            ego_action = ego_dist.sample()
            ego_log_probs.append(ego_dist.log_prob(ego_action))
            ego_entropies.append(ego_dist.entropy())
            ego_idx = int(ego_action.item())

            # Opp = frozen (no grad)
            with torch.no_grad():
                opp_idx, _ = frozen_policy.sample(opp_state, device)

            env.set_skills(skill_from_index(ego_idx), skill_from_index(opp_idx))

        prev_ball_x = curr_ball_x

        if done or steps >= MAX_STEPS_PER_RALLY:
            break

    if done:
        winner = infer_terminal_winner(obs, info, fallback="position") or "opp"
        ego_terminal = 1.0 if winner == "ego" else -1.0
        shaped = SHAPING_COEF * crossings
    else:
        # Truncated: apply the smaller per-crossing coefficient so a stall
        # can't out-earn a decisive win, and no terminal reward.
        ego_terminal = 0.0
        shaped = TRUNCATED_SHAPING_COEF * crossings
    ego_reward = shaped + ego_terminal

    return ego_log_probs, ego_entropies, ego_reward, ego_terminal, steps


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Seeded with {args.seed}")

    env = SkillEnv(proc_id=1, history=HISTORY)
    print(f"Loading PPO from {PPO_MODEL_PATH} (CPU) ...")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")

    policy_A = MetaPolicy().to(device)
    policy_B = MetaPolicy().to(device)
    if args.resume:
        if os.path.exists(args.output_prefix + "_A.pth"):
            policy_A.load_state_dict(torch.load(args.output_prefix + "_A.pth", map_location=device))
            print(f"Resumed policy_A from {args.output_prefix}_A.pth")
        if os.path.exists(args.output_prefix + "_B.pth"):
            policy_B.load_state_dict(torch.load(args.output_prefix + "_B.pth", map_location=device))
            print(f"Resumed policy_B from {args.output_prefix}_B.pth")

    opt_A = torch.optim.Adam(policy_A.parameters(), lr=args.lr)
    opt_B = torch.optim.Adam(policy_B.parameters(), lr=args.lr)
    baseline_A = 0.0
    baseline_B = 0.0

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_path = args.log or (args.output_prefix.replace("models/", "logs/") + ".csv")
    log_f = open(log_path, "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["iter", "phase", "mean_reward", "win_rate", "draw_rate", "loss_rate",
                    "baseline", "loss", "entropy"] + [f"p_{s}" for s in SKILL_NAMES])

    print(f"\nAlternating self-play (5-skill): {args.total_iterations} total iters, "
          f"phase_length={args.phase_length}, {args.rallies_per_iter} rallies/iter, "
          f"entropy_coef={args.entropy_coef}\n")

    try:
        # Track which policy is trainer in current phase: 'A' or 'B'
        for it in range(1, args.total_iterations + 1):
            # Determine current phase from iter
            phase_idx = (it - 1) // args.phase_length
            phase = 'A' if phase_idx % 2 == 0 else 'B'

            if phase == 'A':
                trainer, frozen = policy_A, policy_B
                optim, baseline = opt_A, baseline_A
            else:
                trainer, frozen = policy_B, policy_A
                optim, baseline = opt_B, baseline_B

            # Freeze the other policy
            frozen.eval()
            for p in frozen.parameters():
                p.requires_grad_(False)
            trainer.train()
            for p in trainer.parameters():
                p.requires_grad_(True)

            all_log_probs = []
            all_advantages = []
            all_entropies = []
            rewards_this_iter = []
            wins = 0
            draws = 0
            losses = 0

            for _ in range(args.rallies_per_iter):
                ego_init = random.randint(0, N_SKILLS - 1)
                opp_init = random.randint(0, N_SKILLS - 1)

                ego_lps, ego_ents, ego_r, ego_terminal, _steps = play_one_rally(
                    env, ppo, trainer, frozen, device,
                    ego_init_idx=ego_init, opp_init_idx=opp_init,
                )

                rewards_this_iter.append(ego_r)
                outcome = classify_outcome(ego_terminal)
                if outcome == "win":
                    wins += 1
                elif outcome == "draw":
                    draws += 1
                else:
                    losses += 1

                adv = ego_r - baseline
                for lp in ego_lps:
                    all_log_probs.append(lp)
                    all_advantages.append(adv)
                for ent in ego_ents:
                    all_entropies.append(ent)

            mean_r = float(np.mean(rewards_this_iter))
            win_rate = wins / args.rallies_per_iter
            draw_rate = draws / args.rallies_per_iter
            loss_rate = losses / args.rallies_per_iter
            new_baseline = (1 - args.baseline_ema) * baseline + args.baseline_ema * mean_r
            if phase == 'A':
                baseline_A = new_baseline
            else:
                baseline_B = new_baseline

            if all_log_probs:
                log_probs_t = torch.stack(all_log_probs)
                advs_t = torch.tensor(all_advantages, dtype=torch.float32, device=device)
                pg_loss = -(log_probs_t * advs_t).mean()
                ent_term = torch.stack(all_entropies).mean() if all_entropies else torch.tensor(0.0, device=device)
                loss = pg_loss - args.entropy_coef * ent_term
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainer.parameters(), 1.0)
                optim.step()
                loss_val = float(loss.item())
                ent_val = float(ent_term.item())
            else:
                loss_val = float('nan')
                ent_val = float('nan')

            # Probe trainer policy at zero state
            with torch.no_grad():
                probe = torch.zeros(1, STATE_DIM, device=device)
                p_zero = F.softmax(trainer(probe), dim=-1)[0].cpu().numpy()

            log_w.writerow([it, phase, mean_r, win_rate, draw_rate, loss_rate,
                            new_baseline, loss_val, ent_val] + list(p_zero))
            log_f.flush()

            if it % args.print_every == 0 or it == 1 or (it % args.phase_length == 1):
                p_str = " ".join(f"{s}={p:.2f}" for s, p in zip(SKILL_NAMES, p_zero))
                print(f"[iter {it:4d}/{args.total_iterations}]  phase={phase}  "
                      f"win={win_rate:.2f}  draw={draw_rate:.2f}  lossR={loss_rate:.2f}  "
                      f"loss={loss_val:.4f}  ent={ent_val:.3f}  "
                      f"P_zero=[{p_str}]", flush=True)

            # Save periodically
            if it % args.save_every == 0 or it == args.total_iterations:
                torch.save(policy_A.state_dict(), args.output_prefix + "_A.pth")
                torch.save(policy_B.state_dict(), args.output_prefix + "_B.pth")
    finally:
        env.close()
        log_f.close()

    print(f"\nDone. Saved policies to {args.output_prefix}_{{A,B}}.pth, log to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alternating best-response self-play (5-skill)")
    parser.add_argument("--total-iterations", type=int, default=400)
    parser.add_argument("--phase-length", type=int, default=50,
                        help="Iters per phase before switching trainer (default 50)")
    parser.add_argument("--rallies-per-iter", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--baseline-ema", type=float, default=0.05)
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--output-prefix", default="models/selfplay_5skill_alt",
                        help="Output prefix; will save <prefix>_A.pth and <prefix>_B.pth")
    parser.add_argument("--log", default=None,
                        help="Log csv path (default derived from --output-prefix)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    train(args)
