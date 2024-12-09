from mujoco_env_kuka_with_gantry_table import KukaTennisEnv
from mujoco_env_only_kuka_ik import KukaTennisEnv as KukaTennisEnvIK
# from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3 import PPO
import time
import numpy as np


env = KukaTennisEnv(proc_id=1)
model = PPO.load("logs/best_model_tracker/best_model")
obs, info = env.reset()
history=4
for i in range(20000):
    obs1 = np.zeros(9+9+7+7+9*history)
    obs1[:18] = obs[:18]
    obs1[18:21] = info['diff_pos']
    obs1[21:25] = info['diff_quat']
    obs1[25:32] = info['target']
    action, info = model.predict(obs1, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    env.render()
    if done:
        print("Reset requested")
        obs, _ = env.reset()
    
env.close()
