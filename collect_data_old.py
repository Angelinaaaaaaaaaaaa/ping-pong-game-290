from mujoco_env_comp import KukaTennisEnv
from stable_baselines3 import PPO
import time
import numpy as np
import pickle as pkl

env = KukaTennisEnv(proc_id=1)
model = PPO.load("logs/best_model_combined/best_model")
history = 4
obs, _ = env.reset()
prev_ball_x = obs[36]
curr_ball_x = obs[36]
rallies = []
curr_rally = []
for i in range(20000):
    obs1 = np.zeros(9+9+6+9*history+1)
    obs1[:9] = obs[:9]
    obs1[9:18] = obs[18:27]
    obs1[18:24] = obs[36:42]
    obs1[24:24+9*history] = obs[42:42+history*9]
    obs1[-1] = obs[-2]

    obs2 = np.zeros(9+9+6+9*history+1)
    obs2[:9] = obs[9:18]
    obs2[9:18] = obs[27:36]
    obs2[18:24] = obs[36:42]
    obs2[24:24+9*history] = obs[42+history*9:42+2*history*9]
    obs2[-1] = obs[-1]

    # Frame transformation for the ball on opponent side
    obs2[21:23] = -obs2[21:23]
    obs2[18] = 2*1.5 -obs2[18]
    obs2[19] = -obs2[19]
    # obs1 = obs[:9] + obs[18:27] + obs[36:42] + obs[42:42+history*9] + obs[-2:-1]
    action1,_ = model.predict(obs1, deterministic=True)
    action2,_ = model.predict(obs2, deterministic=True)
    
    action_combined = np.zeros(18)
    action_combined[:9] = action1[:9]
    action_combined[9:] = action2[:9]
    # print(action+[0.]*9)
    obs, reward, done, _, info = env.step(action_combined)
    # env.render()

    # Whenever the ball crosses the net, we consider it a new rally state
    if (prev_ball_x-1.5)*(curr_ball_x-1.5) < 0:
        curr_rally.append(obs)
        env.side_target = np.random.choice([-1., 1.])
        env.side_target_opp = np.random.choice([-1., 1.])
        
    prev_ball_x = curr_ball_x
    if done:
        rallies.append(curr_rally)
        print("Reset requested", len(curr_rally))
        curr_rally = []
        obs, _ = env.reset()
        pkl.dump(rallies, open("data/rallies.pkl", "wb"))

    
env.close()
