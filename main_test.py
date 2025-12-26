
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from agent import Agent
from dqn_model import DQN
from trainer import DQNTrainer
from replay_memory import ReplayMemory
"""
Melhor 'trial': 41
Melhor Métrica (Recompensa Média): 213.39
Melhores Hiperparâmetros:
{'lr': 0.00024501566707817494, 'batch_size': 64, 'gamma': 0.9867111914598459, 'tau': 0.009592709227691799, 'eps_decay': 0.9847127999191941}
"""
ENV_NAME = 'LunarLander-v3' 
LR = 0.0005
MEMORY_CAPACITY = 100000
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
NUM_EPISODES = 1500

EPS_START = 1.0
EPS_END = 0.005
EPS_DECAY = 0.98

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

temp_env = gym.make(ENV_NAME)
obs, info = temp_env.reset()
n_observations = len(obs)
n_actions = temp_env.action_space.n
temp_env.close()

policy_net = DQN(n_observations, n_actions)
target_net = DQN(n_observations, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
criterion = nn.SmoothL1Loss()

agent = Agent(policy_net=policy_net, n_actions = n_actions, device=device)

trainer = DQNTrainer(env_name=ENV_NAME,
                     agent=agent,
                     policy_net=policy_net,
                     target_net=target_net,
                     memory_capacity=MEMORY_CAPACITY,
                     optimizer=optimizer,
                     criterion=criterion,
                     batch_size=BATCH_SIZE,
                     gamma=GAMMA,
                     tau=TAU,
                     device=device,
                     eps_start=EPS_START,
                     eps_end=EPS_END,
                     eps_decay=EPS_DECAY)

trainer.load_pretmodel("./lunar_lander_dqn_interrompido.pt")

try:
    trainer.policy_net.eval()
    trainer.test_agent(num_episodes=10, render=True)
except KeyboardInterrupt:
    print("Test interrupted by user.")





