from mujoco_env_only_kuka_ik import KukaTennisEnv
# from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3 import PPO
import time
import numpy as np


env = KukaTennisEnv(proc_id=1)
model = PPO.load("logs/best_model_ik/best_model")
obs, _ = env.reset()
for i in range(20000):
    # print(i)
    t1 = time.time()
    for j in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
    t2 = time.time()
    print("Time taken:",t2-t1)
    # print(obs[:9])
    # print(i, reward)
    print("Done")
    env.render()
    # if i>1000 :
    #     break
    if done:
        print("Reset requested")
        obs, _ = env.reset()
    # time.sleep(0.1)
    # if done:
    #     obs, _  = env.reset()

env.close()
