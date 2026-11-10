"""
Systematic sweep: all 5x5=25 attack-skill pairs, BOTH players' reactive
defend mechanism active simultaneously (Design B's actual real usage,
unlike mechanism_validation_reactive_pose.py's one-sided matched-pair
design) -- to find which specific skill-pairings are "pathological"
(rally essentially never terminates within the step budget) versus
fine, since collect_data_attack_defend.py's throughput collapsed to
~0.5% and this is the untested part of the space.

Fixed defend=center for both players throughout, to isolate the
ATTACK-skill-pairing dimension specifically (the one never fully
validated -- mechanism_validation_reactive_pose.py only tested
left/right, same skill for both players). Defend-combo variation is a
natural follow-up once this result is in.

Run:
    MUJOCO_GL=cgl PYTHONPATH=. venv/bin/python3.12 \\
        scripts/sweep_both_sided_skill_pairs.py --rallies-per-pair 15
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import statistics

import numpy as np
from stable_baselines3 import PPO

from nash_skills.env_wrapper import SkillEnv
from nash_skills.skills import SKILL_NAMES
from nash_skills.winner_inference import infer_terminal_winner
from nash_skills.eval_matchup import _capture_env_step, _parse_contact_lines

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
MAX_STEPS = 2500
DEFAULT_RALLIES_PER_PAIR = 15


def run_one(env, ppo, ego_skill, opp_skill, seed):
    np.random.seed(seed)
    env.set_skills(ego_skill, opp_skill)
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
    ap.add_argument("--rallies-per-pair", type=int, default=DEFAULT_RALLIES_PER_PAIR)
    ap.add_argument("--base-seed", type=int, default=0)
    args = ap.parse_args()

    global _build_ppo_obs
    from nash_skills.v2.collect_data import _build_ppo_obs as _bpo
    _build_ppo_obs = _bpo

    env = SkillEnv(proc_id=1, reset_mode="ready", skill_profile="aggressive")
    print(f"Loading PPO from {PPO_MODEL_PATH} ...")
    ppo = PPO.load(PPO_MODEL_PATH)

    n_pairs = len(SKILL_NAMES) * len(SKILL_NAMES)
    print(f"{n_pairs} skill pairs x {args.rallies_per_pair} rallies = "
          f"{n_pairs * args.rallies_per_pair} total rally-attempts, both-sided reactive, defend=center/center\n")

    seed_counter = args.base_seed
    results = []
    pair_num = 0
    for ego_skill in SKILL_NAMES:
        for opp_skill in SKILL_NAMES:
            pair_num += 1
            trunc_count = 0
            contacts = []
            steps_list = []
            for _ in range(args.rallies_per_pair):
                steps, done, _winner, ec, oc = run_one(env, ppo, ego_skill, opp_skill, seed_counter)
                seed_counter += 1
                if not done:
                    trunc_count += 1
                contacts.append(ec + oc)
                steps_list.append(steps)
            trunc_rate = trunc_count / args.rallies_per_pair
            avg_contacts = statistics.mean(contacts)
            avg_steps = statistics.mean(steps_list)
            results.append((ego_skill, opp_skill, trunc_rate, avg_contacts, avg_steps))
            print(f"[{pair_num:2d}/{n_pairs}] ego={ego_skill:<12} opp={opp_skill:<12} "
                  f"trunc={trunc_rate:.2f}  avg_total_contacts={avg_contacts:5.1f}  avg_steps={avg_steps:6.1f}",
                  flush=True)

    env.close()

    print("\n" + "=" * 90)
    print("SORTED BY TRUNCATION RATE (worst first)")
    print("=" * 90)
    for ego_skill, opp_skill, trunc_rate, avg_contacts, avg_steps in sorted(results, key=lambda r: -r[2]):
        print(f"ego={ego_skill:<12} opp={opp_skill:<12} trunc={trunc_rate:.2f}  "
              f"avg_total_contacts={avg_contacts:5.1f}  avg_steps={avg_steps:6.1f}")

    overall_trunc = statistics.mean(r[2] for r in results)
    print(f"\nOverall mean truncation rate across all {n_pairs} pairs: {overall_trunc:.3f}")


if __name__ == "__main__":
    main()
