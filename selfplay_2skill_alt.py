"""
2-skill alternating (best-response) self-play.

Like selfplay_5skill_alt.py but restricted to {left, right} — for cleaner
Nash analysis on the 2-skill subset that has near-symmetric same-skill
matchups in the raw data.

Two policies π_A and π_B, each outputs a categorical over 2 skills.
Phase A: train π_A as ego, π_B (frozen) as opp. Phase B: swap. Alternate.

Run:
    PYTHONPATH=. python nash_skills/v2/selfplay_2skill_alt.py \\
        --total-iterations 400 --phase-length 50 --rallies-per-iter 32 \\
        --entropy-coef 0.05 \\
        --output-prefix models/selfplay_2skill_alt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import csv
import io
import random
from contextlib import redirect_stdout

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO

from nash_skills.env_wrapper import SkillEnv
from nash_skills.v2.state_encoder import encode_ego, encode_opp, STATE_DIM
from nash_skills.winner_inference import infer_terminal_winner

# --------------------------------------------------------------------------- #
PPO_MODEL_PATH      = "logs/best_model_tracker1/best_model"
HISTORY             = 4
TABLE_SHIFT         = 1.5
MAX_STEPS_PER_RALLY = 800
SHAPING_COEF        = 0.05

# 2-skill subset (matches selfplay_2skill.py)
SKILL_NAMES_2SKILL  = ["left", "right"]
N_ACTIONS           = len(SKILL_NAMES_2SKILL)   # = 2
# --------------------------------------------------------------------------- #


class MetaPolicy(nn.Module):
    """76-dim state -> categorical over 2 skills (left, right)."""

    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS, hidden=(64, 32)):
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


def play_one_rally(env, ppo, trainer_policy, frozen_policy, device,
                   ego_init_idx=0, opp_init_idx=0):
    env.set_skills(SKILL_NAMES_2SKILL[ego_init_idx], SKILL_NAMES_2SKILL[opp_init_idx])
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

            ego_logits = trainer_policy(
                torch.from_numpy(ego_state).float().unsqueeze(0).to(device))[0]
            ego_dist = torch.distributions.Categorical(logits=ego_logits)
            ego_action = ego_dist.sample()
            ego_log_probs.append(ego_dist.log_prob(ego_action))
            ego_entropies.append(ego_dist.entropy())
            ego_idx = int(ego_action.item())

            with torch.no_grad():
                opp_idx, _ = frozen_policy.sample(opp_state, device)

            env.set_skills(SKILL_NAMES_2SKILL[ego_idx], SKILL_NAMES_2SKILL[opp_idx])

        prev_ball_x = curr_ball_x

        if done or steps >= MAX_STEPS_PER_RALLY:
            break

    shaped = SHAPING_COEF * crossings
    if done:
        winner = infer_terminal_winner(obs, info, fallback="position") or "opp"
        ego_terminal = 1.0 if winner == "ego" else -1.0
    else:
        ego_terminal = 0.0
    ego_reward = shaped + ego_terminal

    return ego_log_probs, ego_entropies, ego_reward, steps


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
    log_w.writerow(["iter", "phase", "mean_reward", "win_rate", "draw_rate",
                    "baseline", "loss", "entropy", "p_left", "p_right"])

    print(f"\nAlternating self-play (2-skill: left, right): "
          f"{args.total_iterations} total iters, phase_length={args.phase_length}, "
          f"{args.rallies_per_iter} rallies/iter, entropy_coef={args.entropy_coef}\n")

    for it in range(1, args.total_iterations + 1):
        phase_idx = (it - 1) // args.phase_length
        phase = 'A' if phase_idx % 2 == 0 else 'B'

        if phase == 'A':
            trainer, frozen = policy_A, policy_B
            optim, baseline = opt_A, baseline_A
        else:
            trainer, frozen = policy_B, policy_A
            optim, baseline = opt_B, baseline_B

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

        for _ in range(args.rallies_per_iter):
            ego_init = random.randint(0, N_ACTIONS - 1)
            opp_init = random.randint(0, N_ACTIONS - 1)

            ego_lps, ego_ents, ego_r, _steps = play_one_rally(
                env, ppo, trainer, frozen, device,
                ego_init_idx=ego_init, opp_init_idx=opp_init,
            )

            rewards_this_iter.append(ego_r)
            if ego_r > 0.5: wins += 1
            elif abs(ego_r) < 0.5: draws += 1

            adv = ego_r - baseline
            for lp in ego_lps:
                all_log_probs.append(lp)
                all_advantages.append(adv)
            for ent in ego_ents:
                all_entropies.append(ent)

        mean_r = float(np.mean(rewards_this_iter))
        win_rate = wins / args.rallies_per_iter
        draw_rate = draws / args.rallies_per_iter
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

        log_w.writerow([it, phase, mean_r, win_rate, draw_rate,
                        new_baseline, loss_val, ent_val,
                        float(p_zero[0]), float(p_zero[1])])
        log_f.flush()

        if it % args.print_every == 0 or it == 1 or (it % args.phase_length == 1):
            print(f"[iter {it:4d}/{args.total_iterations}]  phase={phase}  "
                  f"win={win_rate:.2f}  draw={draw_rate:.2f}  "
                  f"loss={loss_val:.4f}  ent={ent_val:.3f}  "
                  f"P_zero=[left={p_zero[0]:.2f} right={p_zero[1]:.2f}]",
                  flush=True)

        if it % args.save_every == 0 or it == args.total_iterations:
            torch.save(policy_A.state_dict(), args.output_prefix + "_A.pth")
            torch.save(policy_B.state_dict(), args.output_prefix + "_B.pth")

    env.close()
    log_f.close()
    print(f"\nDone. Saved policies to {args.output_prefix}_{{A,B}}.pth, log to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alternating best-response self-play (2-skill)")
    parser.add_argument("--total-iterations", type=int, default=400)
    parser.add_argument("--phase-length", type=int, default=50)
    parser.add_argument("--rallies-per-iter", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--baseline-ema", type=float, default=0.05)
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--output-prefix", default="models/selfplay_2skill_alt")
    parser.add_argument("--log", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    train(args)
