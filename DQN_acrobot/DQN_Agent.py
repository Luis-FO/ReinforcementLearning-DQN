import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from ReplayMem import ReplayMemory, Transition

from DQN import DQN

class Agent():
    def __init__(self, input_size, output_size, env, device= 'cuda', lr = 0.001, memory_capacity=10000, batch_size=128, gamma=0.99):
        self.policy_net = DQN(input_size, output_size)
        self.target_net = DQN(input_size, output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.env = env
        optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        criterion = nn.SmoothL1Loss()
        memory = ReplayMemory(memory_capacity)

        self.steps = 0  # Training steps
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.device = device

    def select_action(self, obs):
        val = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END)*math.exp(-1. * self.steps / self.EPS_DECAY)
        self.steps+=1
        if val>eps_threshold:
            with torch.no_grad():
                action = self.forward(obs).max(1).indices.view(1, 1)
        else:
            action = torch.tensor([[self.env.action_space.sample()]], device=self.device)
        return action
    
    def optimize_model(self):
        pass


if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make('Acrobot-v1')
    obs, info = env.reset()

    num_obs = len(obs)
    num_actions = env.action_space.n

    agent  =  Agent(input_size=num_obs, output_size=num_actions, env = env)
    action = agent.select_action(obs)
    print(action)