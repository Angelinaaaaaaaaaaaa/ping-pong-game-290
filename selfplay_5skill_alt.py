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
import json
import random
import time
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
# Pure zero-sum terminal (winner +1, loser -1). No per-crossing shaping.
# Truncated rallies get a penalty so stalling is strictly worse than losing.
TRUNCATED_PENALTY   = -0.5
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


def play_one_rally(env, ppo, trainer_policy, frozen_policy, device,
                   ego_init_idx=0, opp_init_idx=0):
    """Trainer plays as ego (gradients flow), frozen plays as opp (no grad)."""
    env.set_skills(skill_from_index(ego_init_idx), skill_from_index(opp_init_idx))
    obs, info = env.reset()
    prev_ball_x = float(obs[36])
    ego_idx, opp_idx = ego_init_idx, opp_init_idx

    ego_log_probs = []
    ego_entropies = []
    ego_skills = [ego_init_idx]
    opp_skills = [opp_init_idx]
    ego_probs_at_pick = []
    opp_probs_at_pick = []

    crossings = 0
    steps = 0
    done = False
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
            ego_probs_at_pick.append(
                F.softmax(ego_logits, dim=-1).detach().cpu().numpy().tolist())
            ego_skills.append(ego_idx)

            # Opp = frozen (no grad)
            with torch.no_grad():
                opp_idx, opp_probs = frozen_policy.sample(opp_state, device)
            opp_probs_at_pick.append([float(x) for x in opp_probs])
            opp_skills.append(opp_idx)

            env.set_skills(skill_from_index(ego_idx), skill_from_index(opp_idx))

        prev_ball_x = curr_ball_x

        if done or steps >= MAX_STEPS_PER_RALLY:
            break

    if done:
        winner = infer_terminal_winner(obs, info, fallback="position") or "opp"
        ego_reward = 1.0 if winner == "ego" else -1.0
    else:
        winner = None
        ego_reward = TRUNCATED_PENALTY

    return {
        "ego_log_probs": ego_log_probs,
        "ego_entropies": ego_entropies,
        "ego_reward": ego_reward,
        "steps": steps,
        "crossings": crossings,
        "done": done,
        "winner": winner,
        "ego_skills": ego_skills,
        "opp_skills": opp_skills,
        "ego_probs_at_pick": ego_probs_at_pick,
        "opp_probs_at_pick": opp_probs_at_pick,
    }


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

    env = SkillEnv(proc_id=1, history=HISTORY, skill_profile="aggressive")
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
    log_w.writerow([
        "iter", "phase",
        "mean_reward", "reward_std",
        "win_rate", "loss_rate", "trunc_rate", "draw_rate",
        "mean_crossings", "mean_steps",
    ] + [f"usage_{s}" for s in SKILL_NAMES] + [
        "dominant_fraction_usage", "dominant_fraction_probe", "effective_n_skills",
    ] + [f"p_{s}" for s in SKILL_NAMES] + [
        "baseline", "loss", "pg_loss", "entropy_bonus", "entropy",
        "grad_norm", "advantage_mean", "advantage_std", "wall_time",
    ])

    rally_log_f = None
    rally_log_id = 0
    if args.log_rallies:
        rally_log_f = open(args.log_rallies, "w")

    print(f"\nAlternating self-play (5-skill): {args.total_iterations} total iters, "
          f"phase_length={args.phase_length}, {args.rallies_per_iter} rallies/iter, "
          f"entropy_coef={args.entropy_coef}")
    if rally_log_f is not None:
        print(f"Rally JSONL: {args.log_rallies}")
    print()

    for it in range(1, args.total_iterations + 1):
        iter_start_t = time.time()
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
        losses = 0
        truncs = 0
        crossings_this_iter = []
        steps_this_iter = []
        skill_usage = [0] * N_SKILLS

        for _ in range(args.rallies_per_iter):
            ego_init = random.randint(0, N_SKILLS - 1)
            opp_init = random.randint(0, N_SKILLS - 1)

            rally = play_one_rally(
                env, ppo, trainer, frozen, device,
                ego_init_idx=ego_init, opp_init_idx=opp_init,
            )

            ego_r = rally["ego_reward"]
            rewards_this_iter.append(ego_r)
            crossings_this_iter.append(rally["crossings"])
            steps_this_iter.append(rally["steps"])
            for sk in rally["ego_skills"]:
                skill_usage[sk] += 1

            if not rally["done"]:
                truncs += 1
            elif rally["winner"] == "ego":
                wins += 1
            else:
                losses += 1

            adv = ego_r - baseline
            for lp in rally["ego_log_probs"]:
                all_log_probs.append(lp)
                all_advantages.append(adv)
            for ent in rally["ego_entropies"]:
                all_entropies.append(ent)

            if rally_log_f is not None:
                rally_log_id += 1
                json.dump({
                    "iter": it,
                    "phase": phase,
                    "rally_id": rally_log_id,
                    "ego_init": ego_init,
                    "opp_init": opp_init,
                    "crossings": rally["crossings"],
                    "steps": rally["steps"],
                    "done": rally["done"],
                    "winner": rally["winner"],
                    "ego_reward": ego_r,
                    "ego_skills": rally["ego_skills"],
                    "opp_skills": rally["opp_skills"],
                    "ego_probs_at_pick": rally["ego_probs_at_pick"],
                    "opp_probs_at_pick": rally["opp_probs_at_pick"],
                }, rally_log_f)
                rally_log_f.write("\n")

        n = args.rallies_per_iter
        mean_r = float(np.mean(rewards_this_iter))
        reward_std = float(np.std(rewards_this_iter))
        win_rate = wins / n
        loss_rate = losses / n
        trunc_rate = truncs / n
        draw_rate = trunc_rate
        mean_crossings = float(np.mean(crossings_this_iter))
        mean_steps = float(np.mean(steps_this_iter))
        new_baseline = (1 - args.baseline_ema) * baseline + args.baseline_ema * mean_r
        if phase == 'A':
            baseline_A = new_baseline
        else:
            baseline_B = new_baseline

        total_usage = sum(skill_usage)
        if total_usage > 0:
            usage_probs = np.array(skill_usage, dtype=np.float64) / total_usage
            dominant_fraction_usage = float(usage_probs.max())
            usage_ent = float(-np.sum(usage_probs * np.log(usage_probs + 1e-12)))
        else:
            dominant_fraction_usage = 0.0
            usage_ent = 0.0
        effective_n_skills = float(np.exp(usage_ent))

        if all_log_probs:
            log_probs_t = torch.stack(all_log_probs)
            advs_t = torch.tensor(all_advantages, dtype=torch.float32, device=device)
            adv_mean = float(advs_t.mean().item())
            adv_std = float(advs_t.std().item()) if advs_t.numel() > 1 else 0.0
            pg_loss = -(log_probs_t * advs_t).mean()
            ent_term = torch.stack(all_entropies).mean() if all_entropies else torch.tensor(0.0, device=device)
            loss = pg_loss - args.entropy_coef * ent_term
            optim.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainer.parameters(), 1.0).item())
            optim.step()
            loss_val = float(loss.item())
            pg_loss_val = float(pg_loss.item())
            ent_val = float(ent_term.item())
            ent_bonus = float(args.entropy_coef * ent_val)
        else:
            loss_val = float('nan')
            pg_loss_val = float('nan')
            ent_val = float('nan')
            ent_bonus = float('nan')
            grad_norm = float('nan')
            adv_mean = float('nan')
            adv_std = float('nan')

        with torch.no_grad():
            probe = torch.zeros(1, STATE_DIM, device=device)
            p_zero = F.softmax(trainer(probe), dim=-1)[0].cpu().numpy()
        dominant_fraction_probe = float(p_zero.max())

        wall_time = time.time() - iter_start_t

        log_w.writerow([
            it, phase,
            mean_r, reward_std,
            win_rate, loss_rate, trunc_rate, draw_rate,
            mean_crossings, mean_steps,
        ] + list(skill_usage) + [
            dominant_fraction_usage, dominant_fraction_probe, effective_n_skills,
        ] + list(p_zero) + [
            new_baseline, loss_val, pg_loss_val, ent_bonus, ent_val,
            grad_norm, adv_mean, adv_std, wall_time,
        ])
        log_f.flush()
        if rally_log_f is not None:
            rally_log_f.flush()

        if it % args.print_every == 0 or it == 1 or (it % args.phase_length == 1):
            p_str = " ".join(f"{s[:4]}={p:.2f}" for s, p in zip(SKILL_NAMES, p_zero))
            print(f"[iter {it:4d}/{args.total_iterations}]  phase={phase}  "
                  f"win={win_rate:.2f}  loss={loss_rate:.2f}  trunc={trunc_rate:.2f}  "
                  f"cross={mean_crossings:.1f}  dom={dominant_fraction_usage:.2f}  "
                  f"pg={pg_loss_val:+.3f}  ent={ent_val:.3f}  grad={grad_norm:.2f}  "
                  f"P_zero=[{p_str}]", flush=True)

        if it % args.save_every == 0 or it == args.total_iterations:
            torch.save(policy_A.state_dict(), args.output_prefix + "_A.pth")
            torch.save(policy_B.state_dict(), args.output_prefix + "_B.pth")

    env.close()
    log_f.close()
    if rally_log_f is not None:
        rally_log_f.close()
    print(f"\nDone. Saved policies to {args.output_prefix}_{{A,B}}.pth, log to {log_path}"
          + (f", rallies to {args.log_rallies}" if args.log_rallies else ""))


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
    parser.add_argument("--log-rallies", default=None,
                        help="Optional path to per-rally JSONL log. If set, each rally emits a JSON "
                             "line with phase, skills, probs, outcome — enables phase-conditioned "
                             "cycling analysis. Off by default.")
    args = parser.parse_args()

    train(args)
