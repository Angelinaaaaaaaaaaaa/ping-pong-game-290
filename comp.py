from mujoco_env_comp import KukaTennisEnv
from stable_baselines3 import PPO
import time
import numpy as np
import pickle as pkl
import torch 
import torch.nn as nn
from model_arch import SimpleModel

STRATEGY1 = 'random' # Choose from 'left', 'right', 'random', 'nash', 'nash-ibr', 'nash-p'
STRATEGY2 = 'random' # Choose from 'left', 'right', 'random', 'nash', 'nash-ibr', 'nash-p'

model1 = SimpleModel(116, [64, 32, 16], 1)
model2 = SimpleModel(116, [64, 32, 16], 1)

# Load model1 weights from logs/model1.pt and model2 weights from logs/model2.pt
model1.load_state_dict(torch.load("models/model1.pth"))
model2.load_state_dict(torch.load("models/model2.pth"))

# Load model_p weights from logs/model_p.pt
model_p = SimpleModel(116, [64, 32, 16], 1)
model_p.load_state_dict(torch.load("models/model_p.pth"))

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
        if curr_ball_x-prev_ball_x > 0. :
            if STRATEGY2 == 'left':
                env.side_target_opp = -1.
            elif STRATEGY2 == 'right':
                env.side_target_opp = 1.
            elif STRATEGY2 == 'random':
                env.side_target_opp = np.random.choice([-1., 1.])
            elif STRATEGY2 == 'nash':
                env.side_target_opp = -env.side_target_opp
            elif STRATEGY2 == 'nash-ibr':
                env.side_target_opp = -env.side_target_opp
            elif STRATEGY2 == 'nash-ibr':
                curr_side_target = 1.
                curr_side_target_opp = 1.
                for i in range(4) :
                    # Optimize for player 0
                    X = torch.tensor([obs,obs]).float()
                    X[:,-1] = curr_side_target_opp
                    X[0,-2] = 1.
                    X[1,-2] = -1.
                    output1 = torch.argmax(model1(X)[:,0])
                    curr_side_target = float(X[output1,-2])
                    # Optimize for player 1
                    X = torch.tensor([obs,obs]).float()
                    X[:,-2] = curr_side_target
                    X[0,-1] = 1.
                    X[1,-1] = -1.
                    output2 = torch.argmax(model2(X)[:,0])
                    curr_side_target_opp = float(X[output2,-1])
                # env.side_target = curr_side_target
                env.side_target_opp = curr_side_target_opp
            elif STRATEGY2 == 'nash-p':
                X = torch.tensor([obs,obs,obs,obs]).float()
                X[0,-2] = 1.
                X[0,-1] = 1.
                X[1,-2] = -1.
                X[1,-1] = 1.
                X[2,-2] = 1.
                X[2,-1] = -1.
                X[3,-2] = -1.
                X[3,-1] = -1.
                output = torch.argmax(model_p(X)[:,0])
                env.side_target_opp = float(X[output,-1])
        else:
            if STRATEGY1 == 'left':
                env.side_target = -1.
            elif STRATEGY1 == 'right':
                env.side_target = 1.
            elif STRATEGY1 == 'random':
                env.side_target = np.random.choice([-1., 1.])
            elif STRATEGY1 == 'nash':
                env.side_target = -env.side_target
            elif STRATEGY1 == 'nash-ibr':
                curr_side_target = 1.
                curr_side_target_opp = 1.
                for i in range(4) :
                    # Optimize for player 0
                    X = torch.tensor([obs,obs]).float()
                    X[:,-1] = curr_side_target_opp
                    X[0,-2] = 1.
                    X[1,-2] = -1.
                    output1 = torch.argmax(model1(X)[:,0])
                    curr_side_target = float(X[output1,-2])
                    # Optimize for player 1
                    X = torch.tensor([obs,obs]).float()
                    X[:,-2] = curr_side_target
                    X[0,-1] = 1.
                    X[1,-1] = -1.
                    output2 = torch.argmax(model2(X)[:,0])
                    curr_side_target_opp = float(X[output2,-1])
                env.side_target = curr_side_target
                # env.side_target_opp = curr_side_target_opp
            elif STRATEGY1 == 'nash-p':
                X = torch.tensor([obs,obs,obs,obs]).float()
                X[0,-2] = 1.
                X[0,-1] = 1.
                X[1,-2] = -1.
                X[1,-1] = 1.
                X[2,-2] = 1.
                X[2,-1] = -1.
                X[3,-2] = -1.
                X[3,-1] = -1.
                output = torch.argmax(model_p(X)[:,0])
                env.side_target = float(X[output,-2])


    prev_ball_x = curr_ball_x
    if done:
        rallies.append(curr_rally)
        print("Reset requested", len(curr_rally))
        curr_rally = []
        obs, _ = env.reset()
        pkl.dump(rallies, open("data/rallies_nash1.pkl", "wb"))

    
env.close()
