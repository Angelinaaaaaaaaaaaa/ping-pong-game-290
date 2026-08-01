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
DEFAULT_OUTPUT      = "models/selfplay_5skill.pth"
DEFAULT_LOG         = "logs/selfplay_5skill.csv"
HISTORY             = 4
TABLE_SHIFT         = 1.5
MAX_STEPS_PER_RALLY = 800
# Per-crossing shaping bonus. Two separate coefficients so truncated (stall)
# rallies can't out-earn a decisive win: at 50 crossings a truncated rally
# accumulates 0.001 * 50 = 0.05, well below the ±1 terminal signal, while
# a decisive rally still gets a modest per-crossing bonus (0.005 * ~7 = 0.035).
SHAPING_COEF           = 0.005
TRUNCATED_SHAPING_COEF = 0.001
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
    opp_uses_current = (opp_policy is ego_policy)

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

            ego_logits = ego_policy(torch.from_numpy(ego_state).float().unsqueeze(0).to(device))[0]
            ego_dist = torch.distributions.Categorical(logits=ego_logits)
            ego_action = ego_dist.sample()
            ego_log_probs.append(ego_dist.log_prob(ego_action))
            ego_entropies.append(ego_dist.entropy())
            ego_idx = int(ego_action.item())

            if opp_uses_current:
                opp_logits = opp_policy(torch.from_numpy(opp_state).float().unsqueeze(0).to(device))[0]
                opp_dist = torch.distributions.Categorical(logits=opp_logits)
                opp_action = opp_dist.sample()
                opp_log_probs.append(opp_dist.log_prob(opp_action))
                opp_entropies.append(opp_dist.entropy())
                opp_idx = int(opp_action.item())
            else:
                with torch.no_grad():
                    opp_idx, _ = opp_policy.sample(opp_state, device)

            env.set_skills(skill_from_index(ego_idx), skill_from_index(opp_idx))

        prev_ball_x = curr_ball_x

        if done or steps >= MAX_STEPS_PER_RALLY:
            break

    if done:
        shaped = SHAPING_COEF * crossings
        winner = infer_terminal_winner(obs, info, fallback="position") or "opp"
        ego_terminal = 1.0 if winner == "ego" else -1.0
    else:
        # Truncated: apply the smaller per-crossing coefficient so a stall
        # can't out-earn a decisive win, and no terminal reward.
        shaped = TRUNCATED_SHAPING_COEF * crossings
        ego_terminal = 0.0
    ego_reward = shaped + ego_terminal
    opp_reward = shaped - ego_terminal

    return (ego_log_probs, opp_log_probs, ego_entropies, opp_entropies,
            ego_reward, opp_reward, opp_uses_current, steps)


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
    log_w.writerow(["iter", "mean_reward", "win_rate", "draw_rate",
                    "baseline", "loss", "entropy"] + [f"p_{s}" for s in SKILL_NAMES])

    print(f"\nSelf-play (5-skill): {args.iterations} iters × {args.rallies_per_iter} rallies/iter "
          f"(entropy_coef={args.entropy_coef}, snapshot_prob={args.snapshot_prob}, "
          f"epsilon_baseline={args.epsilon_baseline})")
    if baseline_pool:
        print(f"Baseline pool: {args.baseline_pool}")
    print()

    for it in range(1, args.iterations + 1):
        all_log_probs = []
        all_advantages = []
        all_entropies = []
        rewards_this_iter = []
        wins = 0
        draws = 0

        for _ in range(args.rallies_per_iter):
            # Opp selection: three independent probability regions
            #   [0, epsilon_baseline):                    fixed baseline
            #   [epsilon_baseline, +snapshot_prob):       snapshot buffer
            #   otherwise:                                current policy (self)
            r = random.random()
            if baseline_pool and r < args.epsilon_baseline:
                opp_policy = random.choice(baseline_pool)
            elif snapshot_buffer and r < args.epsilon_baseline + args.snapshot_prob:
                opp_policy = random.choice(snapshot_buffer)
            else:
                opp_policy = policy

            ego_init = random.randint(0, N_SKILLS - 1)
            opp_init = random.randint(0, N_SKILLS - 1)

            (ego_lps, opp_lps, ego_ents, opp_ents,
             ego_r, opp_r, opp_uses_current, _steps) = play_one_rally(
                env, ppo, policy, opp_policy, device,
                ego_init_idx=ego_init, opp_init_idx=opp_init,
            )

            rewards_this_iter.append(ego_r)
            ego_terminal = ego_r - opp_r
            if ego_terminal > 0.5: wins += 1
            elif abs(ego_terminal) < 0.5: draws += 1

            ego_adv = ego_r - baseline
            opp_adv = opp_r - baseline

            for lp in ego_lps:
                all_log_probs.append(lp)
                all_advantages.append(ego_adv)
            for ent in ego_ents:
                all_entropies.append(ent)
            if opp_uses_current:
                for lp in opp_lps:
                    all_log_probs.append(lp)
                    all_advantages.append(opp_adv)
                for ent in opp_ents:
                    all_entropies.append(ent)

        mean_r = float(np.mean(rewards_this_iter))
        win_rate = wins / args.rallies_per_iter
        draw_rate = draws / args.rallies_per_iter
        baseline = (1 - args.baseline_ema) * baseline + args.baseline_ema * mean_r

        if all_log_probs:
            log_probs_t = torch.stack(all_log_probs)
            advs_t = torch.tensor(all_advantages, dtype=torch.float32, device=device)
            pg_loss = -(log_probs_t * advs_t).mean()
            ent_term = torch.stack(all_entropies).mean() if all_entropies else torch.tensor(0.0, device=device)
            loss = pg_loss - args.entropy_coef * ent_term
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            loss_val = float(loss.item())
            ent_val = float(ent_term.item())
        else:
            loss_val = float('nan')
            ent_val = float('nan')

        # Probe policy distribution on a zero state
        with torch.no_grad():
            probe = torch.zeros(1, STATE_DIM, device=device)
            p_zero = F.softmax(policy(probe), dim=-1)[0].cpu().numpy()

        log_w.writerow([it, mean_r, win_rate, draw_rate, baseline,
                        loss_val, ent_val] + list(p_zero))
        log_f.flush()

        if it % args.print_every == 0 or it == 1:
            p_str = " ".join(f"{s[:4]}={p:.2f}" for s, p in zip(SKILL_NAMES, p_zero))
            print(f"[iter {it:4d}/{args.iterations}]  "
                  f"mean_r={mean_r:+.3f}  win={win_rate:.2f}  draw={draw_rate:.2f}  "
                  f"loss={loss_val:.4f}  ent={ent_val:.3f}  buf={len(snapshot_buffer)}  "
                  f"P(zero)=[{p_str}]", flush=True)

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
    print(f"\nDone. Saved policy to {args.output}, log to {args.log}")


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
    args = parser.parse_args()

    train(args)
