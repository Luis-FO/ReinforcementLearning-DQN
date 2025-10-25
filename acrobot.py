
import torch
from DQN_Agent import Agent

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

env = 'LunarLander-v3'

agent = Agent(env_name=env, device=device, lr = 1e-4, memory_capacity=10000, batch_size=128, gamma=0.99)

num_episodes = 1000

agent.train(num_episodes=num_episodes, tau=0.005, show_train=(True, 100))





