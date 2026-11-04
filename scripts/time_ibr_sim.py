"""
Measure simulated-IBR cost per decision (and its result vs random) on a few rallies,
to size the Table 1 IBR budget.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/time_ibr_sim.py --profile aggressive --rallies 3
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import time

import numpy as np


def main():
    from stable_baselines3 import PPO
    from nash_skills.eval_matchup_attack_defend import run_matchup_joint
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="aggressive")
    ap.add_argument("--rallies", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--ppo", default="logs/best_model_tracker1/best_model")
    args = ap.parse_args()
    ppo = PPO.load(args.ppo, device="cpu")
    np.random.seed(0)
    t = time.time()
    r = run_matchup_joint("ibr-sim", "random", ppo, None, n_episodes=args.rallies, max_steps_per_episode=2500,
                          defend_mechanism="preparation_cal", reset_targets=True, skill_profile=args.profile,
                          decision_timing="contact",
                          ibr_sim_config={"rounds": args.rounds, "horizon_exchanges": args.horizon})
    s = r.picker_stats[1]
    n = max(1, s["decisions"])
    print(f"{args.profile} rounds={args.rounds} H={args.horizon}: {r.episodes} rallies, IBR won {r.ego_wins}, "
          f"{s['decisions']} decisions, {s['seconds'] / n:.2f} s/decision, {s['rollouts'] / n:.1f} rollouts/decision, "
          f"exchanges {r.exchanges}, wall {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
