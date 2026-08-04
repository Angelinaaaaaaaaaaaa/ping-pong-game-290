"""
2-skill self-play RL meta-policy (left vs right).

The meta-policy maps a 76-dim encoded state to a categorical over 2 skills
(0 = left, 1 = right). Both players use the same policy (or a frozen
historical snapshot, sampled per rollout to dampen cycling). Skills are
re-selected at each net crossing — matches eval_matchup_2skill.py.

REINFORCE with EMA baseline. Reward signal per rally:
    +1 if ego wins, -1 if opp wins, 0 if truncated.

The low-level PPO controller is frozen — only the meta-policy learns.

Run:
    PYTHONPATH=. python nash_skills/v2/selfplay_2skill.py \
        --iterations 500 --rallies-per-iter 32 \
        --output models/selfplay_2skill.pth
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
from nash_skills.v2.state_encoder import encode_ego, encode_opp, STATE_DIM
from nash_skills.winner_inference import infer_terminal_winner

# --------------------------------------------------------------------------- #
PPO_MODEL_PATH      = "logs/best_model_tracker1/best_model"
DEFAULT_OUTPUT      = "models/selfplay_2skill.pth"
DEFAULT_LOG         = "logs/selfplay_2skill.csv"
SKILL_NAMES_2SKILL  = ["left", "right"]   # idx 0 -> left, idx 1 -> right
HISTORY             = 4
TABLE_SHIFT         = 1.5
MAX_STEPS_PER_RALLY = 1000
# Pure zero-sum terminal (winner +1, loser -1). No per-crossing shaping —
# rally length is not incentivised. Truncated rallies get a symmetric penalty
# so stalling is strictly worse than either winning or losing.
TRUNCATED_PENALTY   = -0.5
# --------------------------------------------------------------------------- #


class MetaPolicy(nn.Module):
    """76-dim state -> categorical over 2 skills."""

    def __init__(self, state_dim: int = STATE_DIM, hidden=(64, 32)):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)            # logits, shape (..., 2)

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
        out = env.step(action)
    return out


def play_one_rally(env, ppo, ego_policy, opp_policy, device, ego_init_idx=0, opp_init_idx=0):
    """
    Roll out a single rally with both players' skills chosen by a meta-policy at
    each net crossing. Returns:
        ego_log_probs : list[Tensor]  — log π(a|s) for ego decisions
        opp_log_probs : list[Tensor]  — log π(a|s) for opp decisions  (only if opp uses ego_policy; else empty)
        ego_reward    : float in {-1, 0, +1}
        opp_uses_current : bool       — whether opp gradients should flow
        steps         : int
    """
    env.set_skills(SKILL_NAMES_2SKILL[ego_init_idx], SKILL_NAMES_2SKILL[opp_init_idx])
    obs, info = env.reset()
    prev_ball_x = float(obs[36])
    ego_idx, opp_idx = ego_init_idx, opp_init_idx

    ego_log_probs = []
    opp_log_probs = []
    ego_entropies = []
    opp_entropies = []

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
            # Ball crossed the net — both players re-pick.
            ego_state = encode_ego(obs, info)
            opp_state = encode_opp(obs, info)

            # Ego (gradients always flow)
            ego_logits = ego_policy(torch.from_numpy(ego_state).float().unsqueeze(0).to(device))[0]
            ego_dist = torch.distributions.Categorical(logits=ego_logits)
            ego_action = ego_dist.sample()
            ego_log_probs.append(ego_dist.log_prob(ego_action))
            ego_entropies.append(ego_dist.entropy())
            ego_idx = int(ego_action.item())

            # Opp (gradients only if same policy; otherwise no_grad)
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

            env.set_skills(SKILL_NAMES_2SKILL[ego_idx], SKILL_NAMES_2SKILL[opp_idx])

        prev_ball_x = curr_ball_x

        if done or steps >= MAX_STEPS_PER_RALLY:
            break

    if done:
        winner = infer_terminal_winner(obs, info, fallback="position") or "opp"
        ego_terminal = 1.0 if winner == "ego" else -1.0
        ego_reward = ego_terminal
        opp_reward = -ego_terminal
    else:
        ego_reward = TRUNCATED_PENALTY
        opp_reward = TRUNCATED_PENALTY

    return (ego_log_probs, opp_log_probs, ego_entropies, opp_entropies,
            ego_reward, opp_reward, opp_uses_current, steps)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    env = SkillEnv(proc_id=1, history=HISTORY, skill_profile="aggressive")
    print(f"Loading PPO from {PPO_MODEL_PATH} (on CPU — MlpPolicy is faster there) ...")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")

    policy = MetaPolicy().to(device)
    if args.resume and os.path.exists(args.output):
        policy.load_state_dict(torch.load(args.output, map_location=device))
        print(f"Resumed weights from {args.output}")

    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
    baseline = 0.0   # EMA of ego_reward — variance reduction

    snapshot_buffer = []   # list[MetaPolicy] (frozen copies)

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_f = open(args.log, "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["iter", "mean_reward", "win_rate", "draw_rate",
                    "p_left_mean", "baseline", "loss", "entropy"])

    print(f"\nSelf-play: {args.iterations} iters × {args.rallies_per_iter} rallies/iter "
          f"(snapshot_prob={args.snapshot_prob}, snapshot_every={args.snapshot_every})\n")

    for it in range(1, args.iterations + 1):
        all_log_probs = []
        all_advantages = []
        all_entropies = []
        rewards_this_iter = []
        wins = 0
        draws = 0

        for _ in range(args.rallies_per_iter):
            if snapshot_buffer and random.random() < args.snapshot_prob:
                opp_policy = random.choice(snapshot_buffer)
            else:
                opp_policy = policy

            ego_init = random.randint(0, 1)
            opp_init = random.randint(0, 1)

            (ego_lps, opp_lps, ego_ents, opp_ents,
             ego_r, opp_r, opp_uses_current, _steps) = play_one_rally(
                env, ppo, policy, opp_policy, device,
                ego_init_idx=ego_init, opp_init_idx=opp_init,
            )

            rewards_this_iter.append(ego_r)
            # Win/draw classification uses the TERMINAL component only (strip shaping)
            ego_terminal = ego_r - opp_r   # = 2 * terminal (shaping cancels out)
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
            loss = pg_loss - args.entropy_coef * ent_term     # subtract entropy → encourage mixing
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            loss_val = float(loss.item())
            ent_val = float(ent_term.item())
        else:
            loss_val = float('nan')
            ent_val = float('nan')

        # Quick P(left) probe on a synthetic zero-state
        with torch.no_grad():
            probe = torch.zeros(1, STATE_DIM, device=device)
            p_left_probe = float(F.softmax(policy(probe), dim=-1)[0, 0].item())

        log_w.writerow([it, mean_r, win_rate, draw_rate, p_left_probe, baseline, loss_val, ent_val])
        log_f.flush()

        if it % args.print_every == 0 or it == 1:
            print(f"[iter {it:4d}/{args.iterations}]  "
                  f"mean_r={mean_r:+.3f}  win={win_rate:.2f}  draw={draw_rate:.2f}  "
                  f"baseline={baseline:+.3f}  loss={loss_val:.4f}  ent={ent_val:.3f}  "
                  f"P(left|zero)={p_left_probe:.3f}  buf={len(snapshot_buffer)}",
                  flush=True)

        # Snapshot
        if it % args.snapshot_every == 0:
            snap = deepcopy(policy).eval()
            for p in snap.parameters():
                p.requires_grad_(False)
            snapshot_buffer.append(snap)
            if len(snapshot_buffer) > args.snapshot_buffer_size:
                snapshot_buffer.pop(0)

        # Save
        if it % args.save_every == 0 or it == args.iterations:
            torch.save(policy.state_dict(), args.output)

    env.close()
    log_f.close()
    print(f"\nDone. Saved policy to {args.output}, log to {args.log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-skill self-play REINFORCE meta-policy")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--rallies-per-iter", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--baseline-ema", type=float, default=0.05,
                        help="EMA coefficient for return baseline (variance reduction)")
    parser.add_argument("--entropy-coef", type=float, default=0.0,
                        help="Entropy bonus (subtract α·H(π) from loss). "
                             "0 = off. Try 0.01-0.1 to force mixing.")
    parser.add_argument("--snapshot-prob", type=float, default=0.5,
                        help="Probability opp samples from snapshot buffer (rest = current policy)")
    parser.add_argument("--snapshot-every", type=int, default=25,
                        help="Snapshot the current policy into the buffer every N iters")
    parser.add_argument("--snapshot-buffer-size", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing --output checkpoint if it exists")
    args = parser.parse_args()

    train(args)
