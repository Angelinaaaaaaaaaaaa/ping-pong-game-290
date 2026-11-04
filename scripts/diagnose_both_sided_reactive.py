"""
Diagnostic: mechanism_validation_reactive_pose.py only ever activates
set_defend_reactive_positions for ONE player at a time (the matched-pair
methodology that isolates causal effect: tested_player gets it, the
other gets None) -- never validated what happens when BOTH players have
it active simultaneously, which is Design B's actual real usage (every
player always holds a defend choice) and what
collect_data_attack_defend.py / selfplay_attack_defend.py both do.

This traces a few rallies with BOTH players' reactive mechanism active,
fixed matching attack skill (mirroring the validation script's own
setup otherwise), to see if "both sides enhanced" is the aggravating
factor behind the near-total collection-throughput collapse.

Run:
    MUJOCO_GL=cgl PYTHONPATH=. venv/bin/python3.12 scripts/diagnose_both_sided_reactive.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO

from nash_skills.env_wrapper import SkillEnv
from nash_skills.winner_inference import infer_terminal_winner
from nash_skills.eval_matchup import _capture_env_step, _parse_contact_lines
from nash_skills.v2.collect_data import _build_ppo_obs

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
MAX_STEPS = 2500
N_RALLIES = 5


def run_one(env, ppo, attack_skill, defend1, defend2, seed):
    np.random.seed(seed)
    env.set_skills(attack_skill, attack_skill)
    env.set_defend_reactive_positions(defend1, defend2)
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
    env = SkillEnv(proc_id=1, reset_mode="ready", skill_profile="aggressive")
    ppo = PPO.load(PPO_MODEL_PATH)

    print("-- ONE-SIDED (matches mechanism_validation_reactive_pose.py's own setup) --")
    for i in range(N_RALLIES):
        steps, done, winner, ec, oc = run_one(env, ppo, "right", "center", None, seed=100 + i)
        print(f"  rally {i}: steps={steps} done={done} winner={winner} ego_contacts={ec} opp_contacts={oc}")

    print("\n-- BOTH-SIDED (Design B's real usage -- both players' reactive mechanism active) --")
    for i in range(N_RALLIES):
        steps, done, winner, ec, oc = run_one(env, ppo, "right", "center", "center", seed=100 + i)
        print(f"  rally {i}: steps={steps} done={done} winner={winner} ego_contacts={ec} opp_contacts={oc}")

    env.close()


if __name__ == "__main__":
    main()
