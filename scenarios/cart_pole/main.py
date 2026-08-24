
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from pathlib import Path

from core.dqn_agent import DQNAgent
from core.dqn_model import DQN
from core.trainer import Trainer
from core.logtrigger import segmented_limit_trigger
from core.info_overlay import InfoOverlay
from core.exploration import EpsilonGreedyStrategy

# Diretório onde o script está localizado
BASE_DIR = Path(__file__).resolve().parent
# Cria diretórios para salvar modelos e vídeos, se não existirem
(BASE_DIR / "model").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "VideosCartPole").mkdir(parents=True, exist_ok=True)

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

policy_net_optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
policy_net_criterion = nn.SmoothL1Loss()
exploration_strategy = EpsilonGreedyStrategy(EPS_START, EPS_END, EPS_DECAY)

agent = DQNAgent(policy_net=policy_net, 
                 optimizer=policy_net_optimizer, 
                 criterion=policy_net_criterion,
                 memory_capacity=MEMORY_CAPACITY,
                 exploration_strategy=exploration_strategy,
                 gamma=GAMMA,
                 tau=TAU,
                 batch_size=BATCH_SIZE,
                 device=device)


format_type = "stories"  # 'stories'
name_prefix = "Mountain-Car-dqn_train"

# Setup environment with InfoOverlay and RecordVideo
dqn_env = gym.make(ENV_NAME, render_mode="rgb_array")
dqn_env = InfoOverlay(dqn_env, format_type = format_type)
dqn_env = RecordVideo(dqn_env, video_folder=f"{BASE_DIR}/VideosCartPole", name_prefix="CartPole-dqn_train_v1", episode_trigger=segmented_limit_trigger)

trainer = Trainer(env=dqn_env, agent=agent)

try:
    
    trainer.train(num_episodes=NUM_EPISODES)
    agent.save(f"{BASE_DIR}/model/CartPole_dqn.pt")

except KeyboardInterrupt:
    print("\nTreinamento interrompido. Salvando modelo atual")
    agent.save(f"{BASE_DIR}/model/CartPole_interrompido.pt")


