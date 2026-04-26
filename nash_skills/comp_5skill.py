"""
Competition script for the 5-skill expanded Nash pipeline.

Strategies available for STRATEGY1 / STRATEGY2:
  'random'       - choose uniformly from all 5 skills each rally
  'left'         - always use "left" skill
  'right'        - always use "right" skill
  'left_short'   - always use "left_short" skill
  'right_short'  - always use "right_short" skill
  'center_safe'  - always use "center_safe" skill
  'nash-p'       - potential-based argmax approximation (best-response w.r.t. learned potential)

Run:
    mjpython nash_skills/comp_5skill.py    # macOS
    python   nash_skills/comp_5skill.py    # Ubuntu
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pickle as pkl
import time
import torch

from stable_baselines3 import PPO
from model_arch import SimpleModel
from nash_skills.env_wrapper import SkillEnv
from nash_skills.skills import SKILL_NAMES, N_SKILLS, skill_index, skill_from_index
from nash_skills.obs_constants import EGO_HIST, OPP_HIST, HISTORY as _HISTORY

# --------------------------------------------------------------------------- #
# Strategy selection — edit these two lines to change each player's strategy  #
# --------------------------------------------------------------------------- #
STRATEGY1 = 'nash-p'    # ego player   — choose from options listed above
STRATEGY2 = 'random'    # opponent     — choose from options listed above
# --------------------------------------------------------------------------- #

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
MODEL1_PATH    = "models/model1_5skill.pth"
MODEL2_PATH    = "models/model2_5skill.pth"
MODEL_P_PATH   = "models/model_p_5skill.pth"
HISTORY        = _HISTORY

# Load value / potential models
model1  = SimpleModel(116, [64, 32, 16], 1)
model2  = SimpleModel(116, [64, 32, 16], 1)
model_p = SimpleModel(116, [64, 32, 16], 1, last_layer_activation=None)
model1.load_state_dict(torch.load(MODEL1_PATH,  weights_only=True))
model2.load_state_dict(torch.load(MODEL2_PATH,  weights_only=True))
model_p.load_state_dict(torch.load(MODEL_P_PATH, weights_only=True))
model1.eval()
model2.eval()
model_p.eval()

# Load PPO policy
ppo = PPO.load(PPO_MODEL_PATH)

env = SkillEnv(proc_id=1, history=HISTORY)
obs, info = env.reset()

prev_ball_x = obs[36]
curr_ball_x = obs[36]
rallies     = []
curr_rally  = []

# Track current skill indices for nash-p
curr_skill1_idx = 0
curr_skill2_idx = 0

def pick_skill(strategy: str, player: int, obs_vec: np.ndarray,
               other_skill_idx: int) -> int:
    """Return the skill index for `player` given the current strategy."""
    if strategy in SKILL_NAMES:
        return skill_index(strategy)

    if strategy == 'random':
        return np.random.randint(N_SKILLS)

    if strategy == 'nash-p':
        # Potential-based argmax: pick the skill that maximises the learned
        # potential Φ given the opponent's fixed skill — a best-response
        # approximation, not a true Nash equilibrium solver.
        # Collect ALL skill values first, then take argmax — avoids the
        # best_idx=0 initialization bias that always returns LEFT on flat surfaces.
        all_vals = []
        for s in range(N_SKILLS):
            x = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
            if player == 1:
                x[0, -2] = s / (N_SKILLS - 1)
                x[0, -1] = other_skill_idx / (N_SKILLS - 1)
            else:
                x[0, -2] = other_skill_idx / (N_SKILLS - 1)
                x[0, -1] = s / (N_SKILLS - 1)
            with torch.no_grad():
                all_vals.append(model_p(x).item())
        return int(np.argmax(all_vals))

    raise ValueError(f"Unknown strategy '{strategy}'")


for step in range(2_000_000):
    obs1 = np.zeros(9 + 9 + 7 + 7 + 9 * HISTORY)
    obs1[:9]    = obs[:9]
    obs1[9:18]  = obs[18:27]
    obs1[18:21] = info['diff_pos']
    obs1[21:25] = info['diff_quat']
    obs1[25:32] = info['target']
    obs1[32:]   = obs[EGO_HIST]

    obs2 = np.zeros(9 + 9 + 7 + 7 + 9 * HISTORY)
    obs2[:9]    = obs[9:18]
    obs2[9:18]  = obs[27:36]
    obs2[18:21] = info['diff_pos_opp']
    obs2[21:25] = info['diff_quat_opp']
    obs2[25:32] = info['target_opp']
    obs2[32:]   = obs[OPP_HIST]

    action1, _ = ppo.predict(obs1, deterministic=True)
    action2, _ = ppo.predict(obs2, deterministic=True)

    action_combined       = np.zeros(18)
    action_combined[:9]   = action1[:9]
    action_combined[9:]   = action2[:9]

    obs, reward, done, _, info = env.step(action_combined)
    env.render()
    time.sleep(0.01)

    curr_ball_x = obs[36]

    # At each rally crossing, pick new skills
    if (prev_ball_x - 1.5) * (curr_ball_x - 1.5) < 0:
        curr_rally.append(obs.copy())

        curr_skill1_idx = pick_skill(STRATEGY1, 1, obs, curr_skill2_idx)
        curr_skill2_idx = pick_skill(STRATEGY2, 2, obs, curr_skill1_idx)

        skill1 = skill_from_index(curr_skill1_idx)
        skill2 = skill_from_index(curr_skill2_idx)
        env.set_skills(skill1, skill2)
        print(f"Rally crossing — ego: {skill1}  opp: {skill2}")

    prev_ball_x = curr_ball_x

    if done:
        rallies.append(curr_rally)
        print(f"Episode done  ({len(curr_rally)} crossings)")
        curr_rally = []
        env.set_skills(skill_from_index(curr_skill1_idx),
                       skill_from_index(curr_skill2_idx))
        obs, info = env.reset()
        pkl.dump(rallies, open("data/rallies_5skill_comp.pkl", "wb"))

env.close()
