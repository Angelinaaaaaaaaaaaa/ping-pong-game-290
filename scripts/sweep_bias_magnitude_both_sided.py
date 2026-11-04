"""
Sweep BIAS_MAGNITUDE under BOTH-SIDED reactive-mechanism activation --
scripts/sweep_both_sided_skill_pairs.py showed universal (~100%)
truncation at the current BIAS_MAGNITUDE=0.15 across all 25 skill
pairs when both players have the mechanism active simultaneously
(Design B's real usage), regardless of which skills are paired. This
sweeps the bias magnitude down to find where decisive outcomes become
achievable again, while checking contacts stay above the old (fully
broken) static-pose mechanism's baseline (~4-12 total contacts/rally
in the no-mechanism-at-all case, per this session's earlier full-scale
validations).

Monkey-patches nash_skills.defend_reactive_pose.BIAS_MAGNITUDE directly
(a module-level constant looked up at call time by lateral_bias(), so
reassigning it externally affects subsequent calls -- no code change
needed to sweep it).

Fixed attack=right/right (skill pairing was shown not to matter) and
defend=center/center, both-sided, matching the isolating sweep this
replaces one variable in.

Run:
    MUJOCO_GL=cgl PYTHONPATH=. venv/bin/python3.12 \\
        scripts/sweep_bias_magnitude_both_sided.py --rallies-per-value 20
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import statistics

import numpy as np
from stable_baselines3 import PPO

import nash_skills.defend_reactive_pose as drp
from nash_skills.env_wrapper import SkillEnv
from nash_skills.winner_inference import infer_terminal_winner
from nash_skills.eval_matchup import _capture_env_step, _parse_contact_lines
from nash_skills.v2.collect_data import _build_ppo_obs

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
MAX_STEPS = 2500
DEFAULT_RALLIES_PER_VALUE = 20
BIAS_VALUES = [0.15, 0.10, 0.06, 0.03, 0.01, 0.0]


def run_one(env, ppo, seed):
    np.random.seed(seed)
    env.set_skills("right", "right")
    env.set_defend_reactive_positions("center", "center")
    obs, info = env.reset()
    steps = 0
    done = False
    ego_contacts = opp_contacts = 0
    while True:
        ppo1 = _build_ppo_obs(obs, info, player=1)
        ppo2 = _build_ppo_obs(obs, info, player=2)
        a1, _ = ppo.predict(ppo1, deterministic=True)
        a2, _ = ppo.predict(ppo2, deterministic=True)
        action = np.zeros(18, dtype=np.float32)
        action[:9] = a1[:9]
        action[9:] = a2[:9]
        (obs, _r, done, _t, info), lines = _capture_env_step(env, action)
        ec, oc, _es, _osu = _parse_contact_lines(lines)
        ego_contacts += ec
        opp_contacts += oc
        steps += 1
        if done or steps >= MAX_STEPS:
            break
    winner = infer_terminal_winner(obs, info, fallback="position") if done else None
    return steps, done, winner, ego_contacts, opp_contacts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rallies-per-value", type=int, default=DEFAULT_RALLIES_PER_VALUE)
    ap.add_argument("--base-seed", type=int, default=0)
    args = ap.parse_args()

    env = SkillEnv(proc_id=1, reset_mode="ready", skill_profile="aggressive")
    print(f"Loading PPO from {PPO_MODEL_PATH} ...")
    ppo = PPO.load(PPO_MODEL_PATH)

    print(f"Sweeping BIAS_MAGNITUDE in {BIAS_VALUES}, {args.rallies_per_value} rallies each, "
          f"both-sided, attack=right/right, defend=center/center\n")

    seed_counter = args.base_seed
    results = []
    for bias in BIAS_VALUES:
        drp.BIAS_MAGNITUDE = bias
        trunc_count = 0
        contacts = []
        steps_list = []
        for _ in range(args.rallies_per_value):
            steps, done, _winner, ec, oc = run_one(env, ppo, seed_counter)
            seed_counter += 1
            if not done:
                trunc_count += 1
            contacts.append(ec + oc)
            steps_list.append(steps)
        trunc_rate = trunc_count / args.rallies_per_value
        avg_contacts = statistics.mean(contacts)
        avg_steps = statistics.mean(steps_list)
        results.append((bias, trunc_rate, avg_contacts, avg_steps))
        print(f"BIAS_MAGNITUDE={bias:<6} trunc={trunc_rate:.2f}  "
              f"avg_total_contacts={avg_contacts:6.1f}  avg_steps={avg_steps:7.1f}", flush=True)

    env.close()

    print("\n" + "=" * 70)
    print("For reference: no-mechanism-at-all baseline (this session's earlier")
    print("full-scale validations) was ~4-12 total contacts/rally -- looking for")
    print("a bias magnitude where trunc drops meaningfully below 1.0 while")
    print("avg_total_contacts stays clearly above that range.")


if __name__ == "__main__":
    main()
