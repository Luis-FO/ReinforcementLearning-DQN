import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from ReplayMem import ReplayMemory, Transition

from DQN import DQN

class Agent():
    def __init__(self, input_size, output_size, env, device= 'cuda', lr = 0.001, memory_capacity=10000, batch_size=128, gamma=0.99):
        self.policy_net = DQN(input_size, output_size).to(device=device)
        self.target_net = DQN(input_size, output_size).to(device=device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.env = env
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayMemory(memory_capacity)

        self.steps = 0  # Training steps
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.device = device

        self.gamma = 0.99
        self.batch_size = batch_size

    def select_action(self, obs):
        val = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END)*math.exp(-1. * self.steps / self.EPS_DECAY)
        self.steps+=1
        if val>eps_threshold:
            with torch.no_grad():
                action = self.policy_net(obs).max(1).indices.view(1, 1)
        else:
            action = torch.tensor([[self.env.action_space.sample()]], device=self.device)
        return action
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size :
            return
        
        samples = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*samples))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state \
                                                if s is not None])
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, \
                                            batch.next_state)), device=self.device, dtype = torch.bool)

        predicted_values = self.policy_net(state_batch).gather(dim = 1, index = action_batch)

        next_state_values = torch.zeros(self.batch_size , device = self.device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(dim = 1).values
        
        expected_values = next_state_values * self.gamma + reward_batch

        loss = self.criterion(predicted_values, expected_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_polycy_net(self):
        torch.save(self.policy_net.state_dict(), "./policy_net.pt")


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