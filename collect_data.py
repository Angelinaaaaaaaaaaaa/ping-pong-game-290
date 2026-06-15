from mujoco_env_comp import KukaTennisEnv
from stable_baselines3 import PPO
import time
import numpy as np
import pickle as pkl

env = KukaTennisEnv(proc_id=1)
model = PPO.load("logs/best_model_tracker1/best_model")
history = 4
obs, info = env.reset()
prev_ball_x = obs[36]
curr_ball_x = obs[36]
rallies = []
curr_rally = []
for i in range(2000000):
    obs1 = np.zeros(9+9+7+7+9*history)
    obs1[:9] = obs[:9]
    obs1[9:18] = obs[18:27]
    obs1[18:21] = info['diff_pos']
    obs1[21:25] = info['diff_quat']
    obs1[25:32] = info['target']
    obs1[32:] = obs[42:42+history*9]
    

    obs2 = np.zeros(9+9+7+7+9*history)
    obs2[:9] = obs[9:18]
    obs2[9:18] = obs[27:36]
    obs2[18:21] = info['diff_pos_opp']
    obs2[21:25] = info['diff_quat_opp']
    obs2[25:32] = info['target_opp']
    obs2[32:] = obs[42+history*9:42+2*history*9]
    
    # obs1 = obs[:9] + obs[18:27] + obs[36:42] + obs[42:42+history*9] + obs[-2:-1]
    action1,_ = model.predict(obs1, deterministic=True)
    action2,_ = model.predict(obs2, deterministic=True)
    
    action_combined = np.zeros(18)
    action_combined[:9] = action1[:9]
    action_combined[9:] = action2[:9]
    # print(action+[0.]*9)
    obs, reward, done, _, info = env.step(action_combined)
    
    env.render()
    time.sleep(0.01)

    curr_ball_x = obs[36]
    # Whenever the ball crosses the net, we consider it a new rally state
    if (prev_ball_x-1.5)*(curr_ball_x-1.5) < 0:
        curr_rally.append(obs)
        # if curr_ball_x-prev_ball_x > 0. :
        #     env.side_target_opp = -env.side_target_opp
        # else:
        #     env.side_target = -env.side_target
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
