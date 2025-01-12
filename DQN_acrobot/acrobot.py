
import torch
from DQN_Agent import Agent

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


env = 'Acrobot-v1'

agent = Agent(env_name=env, device=device)

num_episodes = 600

agent.train(num_episodes=num_episodes,tau=0.05, show_train=(True, 20))





