
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from core.agent import Agent
from core.dqn_model import DQN
from core.trainer import DQNTrainer
from core.logtrigger import segmented_limit_trigger
from core.info_overlay import InfoOverlay

ENV_NAME = 'CartPole-v1' 
LR = 0.0005
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64
GAMMA = 0.95
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


format_type = "stories"  # 'stories', 'feed', 'portrait', or None
name_prefix = "Mountain-Car-dqn_train"

# Setup environment with InfoOverlay and RecordVideo
dqn_env = gym.make(ENV_NAME, render_mode="rgb_array")
dqn_env = InfoOverlay(dqn_env, format_type = format_type)
dqn_env = RecordVideo(dqn_env, video_folder="./CartPole", name_prefix="CartPole-dqn_train_v1", episode_trigger=segmented_limit_trigger)

trainer = DQNTrainer(env_name=ENV_NAME,
                     env=dqn_env,
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
    
    trainer.train(num_episodes=NUM_EPISODES, show_train_after=-1)
    trainer.save_policy_net("./CartPole_dqn.pt")

except KeyboardInterrupt:
    print("\nTreinamento interrompido. Salvando modelo atual")
    trainer.save_policy_net("./CartPole_interrompido.pt")



