
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from pathlib import Path

from core.tester import Tester
from core.info_overlay import InfoOverlay
from core.ddpg_agent import DDPGAgent
from core.networks import Actor, Critic
from core.exploration import GaussianDecayNoise

# Diretório onde o script está localizado
BASE_DIR = Path(__file__).resolve().parent
# Cria diretórios para salvar modelos e vídeos, se não existirem
(BASE_DIR / "model").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "VideosPendulumV2_test").mkdir(parents=True, exist_ok=True)

ENV_NAME = 'Pendulum-v1' 
LR_ACTOR = 0.0001
LR_CRITIC = 0.001

MEMORY_CAPACITY = 100000
BATCH_SIZE = 64
GAMMA = 0.95
TAU = 0.005
NUM_EPISODES = 15

START_STD = 1.0
MIN_STD = 0.005
STD_DECAY = 0.98

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

temp_env = gym.make(ENV_NAME)

state_dim = temp_env.observation_space.shape[0]
action_dim = temp_env.action_space.shape[0]
max_action = float(temp_env.action_space.high[0])

temp_env.close()

actor = Actor(state_dim, action_dim, max_action).to(device)
actor.load(f"{BASE_DIR}/model/Pendulum_actor_interrompido.pt")

critic = Critic(state_dim, action_dim).to(device)
critic.load(f"{BASE_DIR}/model/Pendulum_critic_interrompido.pt")

actor_optim = optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic_optim = optim.Adam(critic.parameters(), lr=LR_CRITIC)


# Instanciando estratégia stateful
noise_strategy = GaussianDecayNoise(start_std=START_STD, min_std=MIN_STD, decay_rate=STD_DECAY)

# Critério de perda para o critic | O critério do actor é implícito na maximização do valor Q
criterion = nn.MSELoss()

agent = DDPGAgent(
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optim,
        critic_optimizer=critic_optim,
        criterion=criterion,
        exploration_strategy=noise_strategy
)


format_type = "stories"  # 'stories'
name_prefix = "Pendulum-ddpg_test"

def record_trigger(episode_id: int) -> bool:
    return True
    
# Setup environment with InfoOverlay and RecordVideo
env = gym.make(ENV_NAME, render_mode="rgb_array")
env = InfoOverlay(env, format_type = format_type)
env = RecordVideo(env, video_folder=f"{BASE_DIR}/VideosPendulumV2_test", name_prefix="Pendulum-ddpg_test_v1", episode_trigger=record_trigger)

tester = Tester(env=env, agent=agent)

try:
    tester.test(num_episodes=NUM_EPISODES)

except KeyboardInterrupt:
    print("\nTeste interrompido.")

finally:
    env.close()