from mujoco_env_kuka_with_gantry_table import KukaTennisEnv
from mujoco_env_only_kuka_ik import KukaTennisEnv as KukaTennisEnvIK
# from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3 import PPO
import time
import numpy as np


env = KukaTennisEnv(proc_id=1)
model = PPO.load("logs/best_model_combined/best_model")
obs, _ = env.reset()
for i in range(20000):
    action,_ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    env.render()
    if done:
        print("Reset requested")
        obs, _ = env.reset()
    
env.close()
