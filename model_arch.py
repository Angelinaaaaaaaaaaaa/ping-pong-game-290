import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, last_layer_activation = 'tanh'):
        super(SimpleModel, self).__init__()
        self.fc = []
        self.batch_norm = nn.BatchNorm1d(input_size,affine=False)
        # Add a normalization layer
        # self.fc.append()
        curr_h = input_size
        for h in hidden_size:

            self.fc.append(nn.Linear(curr_h, h))
            self.fc.append(nn.ReLU())
            curr_h = h
        self.fc.append(nn.Linear(curr_h, output_size))
        # Last layer tanh activation
        if last_layer_activation == 'tanh':
            self.fc.append(nn.Tanh())
        elif last_layer_activation == 'sigmoid':
            self.fc.append(nn.Sigmoid())
        self.fc = nn.Sequential(*self.fc)
        
    def forward(self, x):
        if len(x.shape) == 2 :
            x = self.batch_norm(x)
        elif len(x.shape) == 3 :
            x1 = self.batch_norm(x.reshape((-1,x.shape[-1])))
            x = x1.reshape(x.shape)
        out = self.fc(x)
        return out
