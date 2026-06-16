"""
Quick eval for the 5-skill self-play meta-policy.

Loads models/selfplay_5skill.pth and plays N rallies against each fixed-skill
baseline (random + 5 fixed skills).

Reports per opponent:
  - Real win rate  = wins / (wins + losses)
  - Truncation rate
  - Avg crossings per rally
  - Ego skill usage distribution

Run:
    python eval_selfplay_5skill.py
    python eval_selfplay_5skill.py --checkpoint models/selfplay_5skill.pth --rallies 30
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import io
import random
from contextlib import redirect_stdout

import numpy as np
import torch
from stable_baselines3 import PPO

from nash_skills.env_wrapper import SkillEnv
from nash_skills.skills import SKILL_NAMES, N_SKILLS, skill_from_index
from nash_skills.v2.state_encoder import encode_ego, encode_opp
from nash_skills.winner_inference import infer_terminal_winner
from selfplay_5skill import (
    MetaPolicy, _build_ppo_obs,
    HISTORY, TABLE_SHIFT, MAX_STEPS_PER_RALLY, PPO_MODEL_PATH,
)


def _swallow_step(env, action):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return env.step(action)


def pick_baseline(name, rng):
    if name == "random":
        return lambda s: rng.randint(0, N_SKILLS - 1)
    if name in SKILL_NAMES:
        idx = SKILL_NAMES.index(name)
        return lambda s: idx
    raise ValueError(f"Unknown baseline {name}")


def eval_one(env, ppo, ego_policy, opp_pick_fn, device, n_rallies, rng):
    wins = 0; losses = 0; trunc = 0
    crossings_list = []
    ego_skill_count = [0] * N_SKILLS

    for _ in range(n_rallies):
        ego_init = rng.randint(0, N_SKILLS - 1)
        opp_init = rng.randint(0, N_SKILLS - 1)
        env.set_skills(skill_from_index(ego_init), skill_from_index(opp_init))
        obs, info = env.reset()
        prev_ball_x = float(obs[36])
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
                with torch.no_grad():
                    ego_idx, _ = ego_policy.sample(ego_state, device)
                opp_idx = opp_pick_fn(opp_state)
                ego_skill_count[ego_idx] += 1
                env.set_skills(skill_from_index(ego_idx), skill_from_index(opp_idx))

            prev_ball_x = curr_ball_x

            if done or steps >= MAX_STEPS_PER_RALLY:
                break

        crossings_list.append(crossings)
        if done:
            winner = infer_terminal_winner(obs, info, fallback="position") or "opp"
            if winner == "ego":
                wins += 1
            else:
                losses += 1
        else:
            trunc += 1

    n_done = wins + losses
    real_wr = wins / n_done if n_done > 0 else float("nan")
    total_skills = max(sum(ego_skill_count), 1)
    return {
        "wins": wins, "losses": losses, "trunc": trunc,
        "real_wr": real_wr,
        "trunc_rate": trunc / n_rallies,
        "avg_xs": float(np.mean(crossings_list)),
        "skill_pct": [c / total_skills for c in ego_skill_count],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/selfplay_5skill.pth")
    parser.add_argument("--rallies", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baselines", nargs="+",
                        default=["random"] + list(SKILL_NAMES))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    print(f"Loading policy from {args.checkpoint} ...")
    policy = MetaPolicy().to(device)
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    policy.eval()

    print(f"Loading PPO from {PPO_MODEL_PATH} (CPU) ...")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")

    env = SkillEnv(proc_id=1, history=HISTORY)

    print(f"\nEvaluating {args.rallies} rallies vs each baseline\n" + "=" * 100)
    skill_short = [s[:5] for s in SKILL_NAMES]
    header = (f"{'opp':<12} {'real_wr':>8} {'wins':>5} {'loss':>5} {'trunc':>6} "
              f"{'trunc%':>7} {'avg_xs':>7}  ego skill %: "
              + " ".join(f"{s:>6}" for s in skill_short))
    print(header)
    print("-" * len(header))

    for opp_name in args.baselines:
        pick_fn = pick_baseline(opp_name, rng)
        r = eval_one(env, ppo, policy, pick_fn, device, args.rallies, rng)
        skill_str = " ".join(f"{p:>5.1%}" for p in r["skill_pct"])
        print(f"{opp_name:<12} {r['real_wr']:>8.1%} "
              f"{r['wins']:>5d} {r['losses']:>5d} {r['trunc']:>6d} "
              f"{r['trunc_rate']:>7.1%} {r['avg_xs']:>7.1f}  "
              f"             {skill_str}")

    print("-" * len(header))
    env.close()


if __name__ == "__main__":
    main()
