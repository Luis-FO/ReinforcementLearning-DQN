import torch
import torch.nn as nn
import random
import math

class DQN(nn.Module):

    def __init__(self, num_obs, num_actions, device='cuda'):
        super(DQN, self).__init__()

        self.net = nn.Sequential(nn.Linear(num_obs, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, num_actions))
        
        self.steps = 0
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.device = device
    

    def forward(self, x):
        return self.net(x)
    

if __name__ == "__main__":

    input_data = torch.tensor([[1, 2], [1, 3], [1, 3]], dtype=torch.float)
    agent = DQN(2, 2)

    result = agent(input_data).max(dim=1)
    print(result)