import torch.nn as nn
import torch
import pickle as pkl
import numpy as np
from model_arch import SimpleModel

rallies = pkl.load(open("data/rallies.pkl", "rb"))
X = []
Y1 = []
Y2 = []

n_epochs = 1500
learning_rate = 0.001
# Pre-processing the data
for rally in rallies :
    L = len(rally)
    i=0
    last_ball_pos = rally[-1][36:39]
    last_ball_vel = rally[-1][39:42]
    v_table = -np.sqrt(last_ball_vel[2]**2+2*9.81*max(last_ball_pos[2]-0.56,0))
    t = (last_ball_vel[2]-v_table)/9.81
    x_table = last_ball_pos[0]+last_ball_vel[0]*t
    y_table = last_ball_pos[1]+last_ball_vel[1]*t
    if last_ball_vel[0] > 0:
        if x_table<1.5 or x_table>2.87 or y_table<-0.76 or y_table>0.76:
            rally = rally[:-1]
    else:
        if x_table<0.13 or x_table>1.5 or y_table<-0.76 or y_table>0.76:
            rally = rally[:-1]
    for state in rally:
        ball_pos = state[36:39]
        ball_vel = state[39:42]
        player_no = 1 if ball_vel[0] > 0 else 0
        X.append(state)
        if i < L-2:
            Y1.append([0])
            Y2.append([0])
        elif i == L-2:
            if player_no == 0:
                Y1.append([1])
                Y2.append([0])
            else:
                Y1.append([0])
                Y2.append([1])
        else:
            if player_no == 0:
                Y1.append([-1])
                Y2.append([0])
            else:
                Y1.append([0])
                Y2.append([-1])    
        i += 1

X = torch.tensor(X).float()
Y1 = torch.tensor(Y1).float()
Y2 = torch.tensor(Y2).float()
print(X.shape, Y1.shape, Y2.shape)
model1 = SimpleModel(X.shape[1], [64, 32, 16], 1)
model2 = SimpleModel(X.shape[1], [64, 32, 16], 1)

optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

for i in range(n_epochs):
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    output1 = model1(X)
    output2 = model2(X)
    loss1 = criterion(output1, Y1)
    loss2 = criterion(output2, Y2)
    loss1.backward()
    loss2.backward()
    optimizer1.step()
    optimizer2.step()
    print(f"Epoch {i+1}/{n_epochs}, Loss1: {loss1.item()}, Loss2: {loss2.item()}")

torch.save(model1.state_dict(), "models/model1.pth")
torch.save(model2.state_dict(), "models/model2.pth")

model1.eval()
model2.eval()

model_p = SimpleModel(X.shape[1], [64, 32, 16], 1, last_layer_activation=None)
model_p.batch_norm.running_mean = model1.batch_norm.running_mean
model_p.batch_norm.running_var = model1.batch_norm.running_var
model_p.batch_norm.momentum = 0.
print(model_p.batch_norm.running_mean)
print(model_p.batch_norm.running_var)
optimizer_p = torch.optim.Adam(model_p.parameters(), lr=0.1)

X11 = X.clone()
X11[:,-2] = 1.
X11[:,-1] = 1.
X01 = X.clone()
X01[:,-2] = 0.
X01[:,-1] = 1.
X10 = X.clone()
X10[:,-2] = 1.
X10[:,-1] = 0.
X00 = X.clone()
X00[:,-2] = 0.
X00[:,-1] = 0.
model_p.eval()
min_loss = 0.1
for i in range(n_epochs):
    total_loss = 0
    optimizer_p.zero_grad()
    output1 = model1(X11)
    output2 = model1(X01)
    output_p = output1 - output2
    # print("11:",output_p[:10])
    
    # print("01:",model_p(X11[:10])-model_p(X01[:10]))
    loss_p = criterion(model_p(X11)-model_p(X01), output_p)
    loss_p.backward()
    optimizer_p.step()
    total_loss += loss_p.item()
    
    optimizer_p.zero_grad()
    output1 = model2(X11)
    output2 = model2(X10)
    output_p = output1 - output2
    loss_p = criterion(model_p(X11)-model_p(X10), output_p)
    loss_p.backward()
    optimizer_p.step()
    total_loss += loss_p.item()

    optimizer_p.zero_grad()
    output1 = model1(X10)
    output2 = model1(X00)
    output_p = output1 - output2
    loss_p = criterion(model_p(X10)-model_p(X00), output_p)
    loss_p.backward()
    optimizer_p.step()
    total_loss += loss_p.item()

    optimizer_p.zero_grad()
    output1 = model2(X01)
    output2 = model2(X00)
    output_p = output1 - output2
    loss_p = criterion(model_p(X01)-model_p(X00), output_p)
    loss_p.backward()
    optimizer_p.step()
    total_loss += loss_p.item()
    if total_loss < min_loss:
        min_loss = total_loss
        print("Saved model")
        torch.save(model_p.state_dict(), "models/model_p.pth")
    print(f"Epoch Psi {i+1}/{n_epochs}, Total_loss: {total_loss}")
