
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from agent import Agent
from dqn_model import DQN
from trainer import DQNTrainer
from replay_memory import ReplayMemory


ENV_NAME = 'LunarLander-v3' 
LR = 0.0001
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
NUM_EPISODES = 1000

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

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


try:
    trainer.train(num_episodes=NUM_EPISODES, show_train_after=100) 

    trainer.save_policy_net("./lunar_lander_dqn.pt")

except KeyboardInterrupt:
    print("\nTreinamento interrompido. Salvando modelo atual")
    trainer.save_policy_net("./lunar_lander_dqn_interrompido.pt")




