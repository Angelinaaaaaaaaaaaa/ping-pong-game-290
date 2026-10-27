"""
Quick eval for the 2-skill self-play meta-policy.

Loads models/selfplay_2skill.pth (a MetaPolicy) and plays N rallies against
each fixed-skill baseline opponent (random / left / right).

Reports:
  - Real win rate  = wins / (wins + losses)   [excludes truncations]
  - Truncation rate
  - Avg crossings per rally
  - Skill usage by ego

Run:
    PYTHONPATH=. python nash_skills/v2/eval_selfplay_2skill.py
    PYTHONPATH=. python nash_skills/v2/eval_selfplay_2skill.py \
        --checkpoint models/selfplay_2skill.pth --rallies 30
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
import torch.nn.functional as F
from stable_baselines3 import PPO

from nash_skills.env_wrapper import SkillEnv
from nash_skills.v2.state_encoder import encode_ego, encode_opp
from nash_skills.winner_inference import infer_terminal_winner
from selfplay_2skill import (
    MetaPolicy, SKILL_NAMES_2SKILL, _build_ppo_obs,
    HISTORY, TABLE_SHIFT, MAX_STEPS_PER_RALLY, PPO_MODEL_PATH,
)


def _swallow_step(env, action):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return env.step(action)


def pick_baseline(name: str, rng):
    """Return a function (state_np) -> skill_idx for a baseline policy."""
    if name == "random":
        return lambda s: rng.randint(0, 1)
    if name == "left":
        return lambda s: 0
    if name == "right":
        return lambda s: 1
    raise ValueError(f"Unknown baseline {name}")


def eval_one(env, ppo, ego_policy, opp_pick_fn, device, n_rallies, rng):
    wins = 0
    losses = 0
    trunc = 0
    crossings_list = []
    ego_skill_count = [0, 0]

    for _ in range(n_rallies):
        ego_init = rng.randint(0, 1)
        opp_init = rng.randint(0, 1)
        env.set_skills(SKILL_NAMES_2SKILL[ego_init], SKILL_NAMES_2SKILL[opp_init])
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
                env.set_skills(SKILL_NAMES_2SKILL[ego_idx], SKILL_NAMES_2SKILL[opp_idx])

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
    return {
        "wins": wins,
        "losses": losses,
        "truncated": trunc,
        "real_win_rate": real_wr,
        "trunc_rate": trunc / n_rallies,
        "avg_crossings": float(np.mean(crossings_list)),
        "ego_skill_pct": (
            ego_skill_count[0] / max(sum(ego_skill_count), 1),
            ego_skill_count[1] / max(sum(ego_skill_count), 1),
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/selfplay_2skill.pth")
    parser.add_argument("--rallies", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baselines", nargs="+",
                        default=["random", "left", "right"])
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

    print(f"\nEvaluating {args.rallies} rallies vs each baseline\n"
          f"{'='*72}")
    header = f"{'opp':<10} {'real_wr':>9} {'wins':>5} {'loss':>5} {'trunc':>5} {'trunc%':>7} {'avg_xs':>7} {'ego L%':>7} {'ego R%':>7}"
    print(header)
    print("-" * len(header))

    for opp_name in args.baselines:
        pick_fn = pick_baseline(opp_name, rng)
        r = eval_one(env, ppo, policy, pick_fn, device, args.rallies, rng)
        print(f"{opp_name:<10} {r['real_win_rate']:>9.2%} "
              f"{r['wins']:>5d} {r['losses']:>5d} {r['truncated']:>5d} "
              f"{r['trunc_rate']:>7.1%} {r['avg_crossings']:>7.1f} "
              f"{r['ego_skill_pct'][0]:>7.1%} {r['ego_skill_pct'][1]:>7.1%}")

    print("-" * len(header))
    print("\nreal_wr = wins / (wins + losses)  — excludes truncations")
    print("avg_xs  = avg net crossings per rally  (higher = longer rallies)")
    env.close()


if __name__ == "__main__":
    main()
