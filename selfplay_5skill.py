"""
5-skill self-play RL meta-policy.

Same algorithm as selfplay_2skill.py, generalized to all 5 skills from
nash_skills.skills.SKILL_NAMES. The meta-policy outputs a categorical over
5 actions; entropy bonus is built in to encourage mixed-strategy convergence
(in 5-skill the action space is larger so pure-strategy collapse is more
likely to be sub-optimal).

Run:
    PYTHONPATH=. python nash_skills/v2/selfplay_5skill.py \
        --iterations 500 --rallies-per-iter 32 \
        --entropy-coef 0.05 \
        --output models/selfplay_5skill.pth
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
DEFAULT_OUTPUT      = "models/selfplay_5skill.pth"
DEFAULT_LOG         = "logs/selfplay_5skill.csv"
HISTORY             = 4
TABLE_SHIFT         = 1.5
MAX_STEPS_PER_RALLY = 800
# Pure zero-sum terminal (winner +1, loser -1). No per-crossing shaping —
# rally length is not incentivised, avoiding the common-bonus that would
# corrupt the zero-sum structure. Truncated rallies get a symmetric penalty
# so stalling is strictly worse than either winning or losing.
TRUNCATED_PENALTY = -0.5
# --------------------------------------------------------------------------- #


class MetaPolicy(nn.Module):
    """76-dim state -> categorical over N_SKILLS actions."""

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


class FixedSkillPolicy:
    """Stateless opp for epsilon-baseline exploration. Implements only the
    interface used in the `not opp_uses_current` branch of play_one_rally:
    `.sample(state, device) -> (skill_idx, probs)`. `state` is ignored.

    Modes:
      - "random":   uniform over N_SKILLS every call
      - a valid skill name (e.g. "left", "center_safe"): always that skill
    """

    def __init__(self, mode: str):
        self.mode = mode
        self._uniform_probs = np.full(N_SKILLS, 1.0 / N_SKILLS, dtype=np.float32)
        if mode == "random":
            self._fixed_idx = None
        elif mode in SKILL_NAMES:
            self._fixed_idx = SKILL_NAMES.index(mode)
        else:
            raise ValueError(f"Unknown FixedSkillPolicy mode: {mode!r}")

    def sample(self, _state_np, _device):
        if self._fixed_idx is None:
            idx = random.randint(0, N_SKILLS - 1)
        else:
            idx = self._fixed_idx
        return idx, self._uniform_probs


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


def play_one_rally(env, ppo, ego_policy, opp_policy, device,
                   ego_init_idx=0, opp_init_idx=0):
    env.set_skills(skill_from_index(ego_init_idx), skill_from_index(opp_init_idx))
    obs, info = env.reset()
    prev_ball_x = float(obs[36])
    ego_idx, opp_idx = ego_init_idx, opp_init_idx

    ego_log_probs, opp_log_probs = [], []
    ego_entropies, opp_entropies = [], []
    ego_skills = [ego_init_idx]
    opp_skills = [opp_init_idx]
    ego_probs_at_pick = []
    opp_probs_at_pick = []
    opp_uses_current = (opp_policy is ego_policy)

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

            ego_logits = ego_policy(torch.from_numpy(ego_state).float().unsqueeze(0).to(device))[0]
            ego_dist = torch.distributions.Categorical(logits=ego_logits)
            ego_action = ego_dist.sample()
            ego_log_probs.append(ego_dist.log_prob(ego_action))
            ego_entropies.append(ego_dist.entropy())
            ego_idx = int(ego_action.item())
            ego_probs_at_pick.append(
                F.softmax(ego_logits, dim=-1).detach().cpu().numpy().tolist())
            ego_skills.append(ego_idx)

            if opp_uses_current:
                opp_logits = opp_policy(torch.from_numpy(opp_state).float().unsqueeze(0).to(device))[0]
                opp_dist = torch.distributions.Categorical(logits=opp_logits)
                opp_action = opp_dist.sample()
                opp_log_probs.append(opp_dist.log_prob(opp_action))
                opp_entropies.append(opp_dist.entropy())
                opp_idx = int(opp_action.item())
                opp_probs_at_pick.append(
                    F.softmax(opp_logits, dim=-1).detach().cpu().numpy().tolist())
            else:
                with torch.no_grad():
                    opp_idx, opp_probs = opp_policy.sample(opp_state, device)
                opp_probs_at_pick.append([float(x) for x in opp_probs])
            opp_skills.append(opp_idx)

            env.set_skills(skill_from_index(ego_idx), skill_from_index(opp_idx))

        prev_ball_x = curr_ball_x

        if done or steps >= MAX_STEPS_PER_RALLY:
            break

    if done:
        winner = infer_terminal_winner(obs, info, fallback="position") or "opp"
        ego_terminal = 1.0 if winner == "ego" else -1.0
        ego_reward = ego_terminal
        opp_reward = -ego_terminal
    else:
        winner = None
        ego_reward = TRUNCATED_PENALTY
        opp_reward = TRUNCATED_PENALTY

    return {
        "ego_log_probs": ego_log_probs,
        "opp_log_probs": opp_log_probs,
        "ego_entropies": ego_entropies,
        "opp_entropies": opp_entropies,
        "ego_reward": ego_reward,
        "opp_reward": opp_reward,
        "opp_uses_current": opp_uses_current,
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
    print(f"Loading PPO from {PPO_MODEL_PATH} (CPU — MlpPolicy is faster there) ...")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")

    policy = MetaPolicy().to(device)
    if args.resume and os.path.exists(args.output):
        policy.load_state_dict(torch.load(args.output, map_location=device))
        print(f"Resumed weights from {args.output}")

    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
    baseline = 0.0

    snapshot_buffer = []
    baseline_pool = [FixedSkillPolicy(m) for m in args.baseline_pool] if args.epsilon_baseline > 0 else []
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_f = open(args.log, "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow([
        "iter",
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

    print(f"\nSelf-play (5-skill): {args.iterations} iters × {args.rallies_per_iter} rallies/iter "
          f"(entropy_coef={args.entropy_coef}, snapshot_prob={args.snapshot_prob}, "
          f"epsilon_baseline={args.epsilon_baseline})")
    if baseline_pool:
        print(f"Baseline pool: {args.baseline_pool}")
    if rally_log_f is not None:
        print(f"Rally JSONL: {args.log_rallies}")
    print()

    for it in range(1, args.iterations + 1):
        iter_start_t = time.time()
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
            # Opp selection: three independent probability regions
            #   [0, epsilon_baseline):                    fixed baseline
            #   [epsilon_baseline, +snapshot_prob):       snapshot buffer
            #   otherwise:                                current policy (self)
            r = random.random()
            if baseline_pool and r < args.epsilon_baseline:
                opp_policy = random.choice(baseline_pool)
                opp_type = "baseline"
            elif snapshot_buffer and r < args.epsilon_baseline + args.snapshot_prob:
                opp_policy = random.choice(snapshot_buffer)
                opp_type = "snapshot"
            else:
                opp_policy = policy
                opp_type = "self"

            ego_init = random.randint(0, N_SKILLS - 1)
            opp_init = random.randint(0, N_SKILLS - 1)

            rally = play_one_rally(
                env, ppo, policy, opp_policy, device,
                ego_init_idx=ego_init, opp_init_idx=opp_init,
            )

            ego_r = rally["ego_reward"]
            opp_r = rally["opp_reward"]
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

            ego_adv = ego_r - baseline
            opp_adv = opp_r - baseline

            for lp in rally["ego_log_probs"]:
                all_log_probs.append(lp)
                all_advantages.append(ego_adv)
            for ent in rally["ego_entropies"]:
                all_entropies.append(ent)
            if rally["opp_uses_current"]:
                for lp in rally["opp_log_probs"]:
                    all_log_probs.append(lp)
                    all_advantages.append(opp_adv)
                for ent in rally["opp_entropies"]:
                    all_entropies.append(ent)

            if rally_log_f is not None:
                rally_log_id += 1
                json.dump({
                    "iter": it,
                    "rally_id": rally_log_id,
                    "opp_type": opp_type,
                    "ego_init": ego_init,
                    "opp_init": opp_init,
                    "crossings": rally["crossings"],
                    "steps": rally["steps"],
                    "done": rally["done"],
                    "winner": rally["winner"],
                    "ego_reward": ego_r,
                    "opp_reward": opp_r,
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
        baseline = (1 - args.baseline_ema) * baseline + args.baseline_ema * mean_r

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
            grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item())
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

        # Probe policy distribution on a zero state
        with torch.no_grad():
            probe = torch.zeros(1, STATE_DIM, device=device)
            p_zero = F.softmax(policy(probe), dim=-1)[0].cpu().numpy()
        dominant_fraction_probe = float(p_zero.max())

        wall_time = time.time() - iter_start_t

        log_w.writerow([
            it,
            mean_r, reward_std,
            win_rate, loss_rate, trunc_rate, draw_rate,
            mean_crossings, mean_steps,
        ] + list(skill_usage) + [
            dominant_fraction_usage, dominant_fraction_probe, effective_n_skills,
        ] + list(p_zero) + [
            baseline, loss_val, pg_loss_val, ent_bonus, ent_val,
            grad_norm, adv_mean, adv_std, wall_time,
        ])
        log_f.flush()
        if rally_log_f is not None:
            rally_log_f.flush()

        if it % args.print_every == 0 or it == 1:
            p_str = " ".join(f"{s[:4]}={p:.2f}" for s, p in zip(SKILL_NAMES, p_zero))
            print(f"[iter {it:4d}/{args.iterations}]  "
                  f"mean_r={mean_r:+.3f}  win={win_rate:.2f}  loss={loss_rate:.2f}  trunc={trunc_rate:.2f}  "
                  f"cross={mean_crossings:.1f}  dom={dominant_fraction_usage:.2f}  "
                  f"pg={pg_loss_val:+.3f}  ent={ent_val:.3f}  grad={grad_norm:.2f}  "
                  f"buf={len(snapshot_buffer)}  P(zero)=[{p_str}]", flush=True)

        if it % args.snapshot_every == 0:
            snap = deepcopy(policy).eval()
            for p in snap.parameters():
                p.requires_grad_(False)
            snapshot_buffer.append(snap)
            if len(snapshot_buffer) > args.snapshot_buffer_size:
                snapshot_buffer.pop(0)

        if it % args.save_every == 0 or it == args.iterations:
            torch.save(policy.state_dict(), args.output)

    env.close()
    log_f.close()
    if rally_log_f is not None:
        rally_log_f.close()
    print(f"\nDone. Saved policy to {args.output}, log to {args.log}"
          + (f", rallies to {args.log_rallies}" if args.log_rallies else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5-skill self-play REINFORCE meta-policy")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--rallies-per-iter", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--baseline-ema", type=float, default=0.05)
    parser.add_argument("--entropy-coef", type=float, default=0.05,
                        help="Entropy bonus α — subtract α·H(π) from loss to encourage mixing. "
                             "5-skill default: 0.05 (try 0.0 for pure REINFORCE, 0.1+ for stronger mixing)")
    parser.add_argument("--snapshot-prob", type=float, default=0.5)
    parser.add_argument("--snapshot-every", type=int, default=25)
    parser.add_argument("--snapshot-buffer-size", type=int, default=10)
    parser.add_argument("--epsilon-baseline", type=float, default=0.0,
                        help="Probability of sampling opp from the fixed-baseline pool "
                             "instead of snapshot/current (default: 0.0 = disabled). "
                             "Try 0.2 to expose the policy to interpretable fixed strategies.")
    parser.add_argument("--baseline-pool", nargs="+",
                        default=["random"] + list(SKILL_NAMES),
                        help="Fixed strategies to sample from when epsilon-baseline fires. "
                             "Each entry must be 'random' or a valid skill name.")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for random/numpy/torch (default: None = OS entropy)")
    parser.add_argument("--log-rallies", default=None,
                        help="Optional path to per-rally JSONL log. If set, each rally emits a JSON "
                             "line with skills, probs, outcome. Enables post-hoc analysis of skill "
                             "matchups, landing patterns, and per-decision policy state. Off by default.")
    args = parser.parse_args()

    train(args)
