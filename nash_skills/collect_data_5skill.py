"""
Collect rally trajectory data for the 5-skill expanded Nash pipeline.

Rolls out the trained PPO policy with all 25 skill combinations (5x5)
and saves the resulting rallies to data/rallies_5skill.pkl.

Each entry in the pickle file is a dict:
    {
        'skill1':  str,        # ego player skill name
        'skill2':  str,        # opponent skill name
        'states':  list[obs],  # observations at each rally crossing
    }

Run:
    python nash_skills/collect_data_5skill.py
    (use 'mjpython' on macOS)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pickle as pkl
import time
import itertools
from stable_baselines3 import PPO

from nash_skills.env_wrapper import SkillEnv
from nash_skills.skills import SKILL_NAMES

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #
PPO_MODEL_PATH  = "logs/best_model_tracker1/best_model"
OUTPUT_PATH     = "data/rallies_5skill.pkl"
STEPS_PER_COMBO = 20_000   # simulation steps per skill combination
HISTORY         = 4
# --------------------------------------------------------------------------- #

env   = SkillEnv(proc_id=1, history=HISTORY)
model = PPO.load(PPO_MODEL_PATH)

all_rallies = []

for skill1, skill2 in itertools.product(SKILL_NAMES, SKILL_NAMES):
    print(f"\n=== Collecting: ego={skill1}  opp={skill2} ===")
    env.set_skills(skill1, skill2)
    obs, info = env.reset()

    prev_ball_x = obs[36]
    curr_ball_x = obs[36]
    curr_rally  = []

    for _ in range(STEPS_PER_COMBO):
        obs1 = np.zeros(9 + 9 + 7 + 7 + 9 * HISTORY)
        obs1[:9]   = obs[:9]
        obs1[9:18] = obs[18:27]
        obs1[18:21] = info['diff_pos']
        obs1[21:25] = info['diff_quat']
        obs1[25:32] = info['target']
        obs1[32:]   = obs[42: 42 + HISTORY * 9]

        obs2 = np.zeros(9 + 9 + 7 + 7 + 9 * HISTORY)
        obs2[:9]   = obs[9:18]
        obs2[9:18] = obs[27:36]
        obs2[18:21] = info['diff_pos_opp']
        obs2[21:25] = info['diff_quat_opp']
        obs2[25:32] = info['target_opp']
        obs2[32:]   = obs[42 + HISTORY * 9: 42 + 2 * HISTORY * 9]

        action1, _ = model.predict(obs1, deterministic=True)
        action2, _ = model.predict(obs2, deterministic=True)

        action_combined       = np.zeros(18)
        action_combined[:9]   = action1[:9]
        action_combined[9:]   = action2[:9]

        obs, reward, done, _, info = env.step(action_combined)

        curr_ball_x = obs[36]
        if (prev_ball_x - 1.5) * (curr_ball_x - 1.5) < 0:
            curr_rally.append(obs.copy())

        prev_ball_x = curr_ball_x

        if done:
            if len(curr_rally) > 0:
                all_rallies.append({
                    'skill1': skill1,
                    'skill2': skill2,
                    'states': curr_rally,
                })
            curr_rally = []
            env.set_skills(skill1, skill2)
            obs, info = env.reset()
            prev_ball_x = obs[36]
            curr_ball_x = obs[36]

    # Flush any incomplete rally at end of combo
    if len(curr_rally) > 0:
        all_rallies.append({
            'skill1': skill1,
            'skill2': skill2,
            'states': curr_rally,
        })

env.close()

os.makedirs(os.path.dirname(OUTPUT_PATH) if os.path.dirname(OUTPUT_PATH) else ".", exist_ok=True)
with open(OUTPUT_PATH, "wb") as f:
    pkl.dump(all_rallies, f)

print(f"\nSaved {len(all_rallies)} rallies to {OUTPUT_PATH}")
