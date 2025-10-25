import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import gymnasium as gym
from replay_memory import ReplayMemory, Transition

from dqn_model import DQN

class Agent():
    def __init__(self, env_name, device= 'cuda', lr = 0.001, memory_capacity=10000, batch_size=128, gamma=0.99):

        self.env_name = env_name    
        self.env = gym.make(env_name)

     
        obs, info = self.env.reset()
        input_size = len(obs)
        output_size = self.env.action_space.n

        self.policy_net = DQN(input_size, output_size).to(device=device)
        self.target_net = DQN(input_size, output_size).to(device=device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayMemory(memory_capacity)

        self.steps = 0  # Training steps
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.device = device

        self.gamma = gamma
        self.batch_size = batch_size

    def select_action(self, obs):
        val = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END)*math.exp(-1. * self.steps / self.EPS_DECAY)

        self.steps+=1
        if val>eps_threshold:
            with torch.no_grad():
                action = self.policy_net(obs).max(1).indices.view(1, 1)
        else:
            action = torch.tensor([[random.randrange(self.env.action_space.n)]], device=self.device, dtype=torch.long)
        return action


if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make('Acrobot-v1')
    obs, info = env.reset()

    num_obs = len(obs)
    num_actions = env.action_space.n

    obs = torch.tensor(obs).unsqueeze(0)

    agent  =  Agent(input_size=num_obs, output_size=num_actions, env = env)
    action = agent.select_action(obs)
    print(action)