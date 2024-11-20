from mujoco_env_only_kuka_ik import KukaTennisEnv as KukaTennisEnvIK
# from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3 import PPO
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gantry', action='store_true', help='Enable gantry (XY movement)')
parser.add_argument('--table', action='store_true', help='Enable table in simulation')
parser.add_argument('--render', action='store_true', help='Enable rendering')
parser.add_argument('--ik_rl', action='store_true', help='Enable using RL based IK')

args = parser.parse_args()
if args.ik_rl and args.gantry:
    print("IK RL and Gantry cannot be enabled at the same time")
    exit(0)

if args.gantry:
    if args.table:
        from mujoco_env_kuka_with_gantry_table import KukaTennisEnv
    else :
        from mujoco_env_kuka_with_gantry import KukaTennisEnv
else:
    if args.table:
        from mujoco_env_kuka_with_table import KukaTennisEnv
    else:
        from mujoco_env_only_kuka import KukaTennisEnv



if args.gantry:
    model = PPO.load("logs/best_model2/best_model")
else:
    model = PPO.load("logs/best_model/best_model")

if args.ik_rl:
    env_ik = KukaTennisEnvIK(proc_id=1)
    model_ik = PPO.load("logs/best_model_ik1/best_model")
    obs_ik, _ = env_ik.reset()
env = KukaTennisEnv(proc_id=1)
obs, _ = env.reset()

for i in range(20000):
    if args.ik_rl:
        env_ik.reset_pose(env.data.qpos[:7])
        env_ik.set_target_pose(env.curr_target)
        for j in range(50):
            action_ik, _ = model_ik.predict(obs_ik, deterministic=True)
            obs_ik, reward, done, _, info = env_ik.step(action_ik)
        action = (env_ik.data.qpos[:7]- env.data.qpos[:7])
    else :
        action,_ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    if args.render :
        env.render()
    if done:
        print("Reset requested")
        obs, _ = env.reset()
    
env.close()
