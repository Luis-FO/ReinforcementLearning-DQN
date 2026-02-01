
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from pathlib import Path

from core.tester import Tester
from core.info_overlay import InfoOverlay
from core.td3_agent import TD3Agent
from core.networks import Actor, Critic
from core.exploration import GaussianNoise

# Diretório onde o script está localizado
BASE_DIR = Path(__file__).resolve().parent
# Cria diretórios para salvar modelos e vídeos, se não existirem
(BASE_DIR / "model").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "VideosBipedalWalker_TD3_v2_Test").mkdir(parents=True, exist_ok=True)

ENV_NAME = 'BipedalWalker-v3' 
LR_ACTOR = 0.0003
LR_CRITIC = 0.0003

MEMORY_CAPACITY = 1000000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
NUM_EPISODES = 2500

STD = 0.1

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
actor.load(f"{BASE_DIR}/model/BipedalWalker_actor_td3_v_interrompido.pt")
critic_1 = Critic(state_dim, action_dim).to(device)
critic_1.load(f"{BASE_DIR}/model/BipedalWalker_critic_1_td3_v_interrompido.pt")
critic_2 = Critic(state_dim, action_dim).to(device)
critic_2.load(f"{BASE_DIR}/model/BipedalWalker_critic_2_td3_v_interrompido.pt")


actor_optim = optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic_1_optim = optim.Adam(critic_1.parameters(), lr=LR_CRITIC)
critic_2_optim = optim.Adam(critic_2.parameters(), lr=LR_CRITIC)


# Instanciando estratégia stateful
noise_strategy = GaussianNoise(std=STD)

# Critério de perda para o critic | O critério do actor é implícito na maximização do valor Q
criterion = nn.MSELoss()

agent = TD3Agent(actor=actor, 
                 critic_1=critic_1, 
                 critic_2=critic_2, 
                 actor_optim=actor_optim,
                 critic_optim_1=critic_1_optim, 
                 critic_optim_2=critic_2_optim, 
                 exploration_strategy=noise_strategy,
                 criterion=criterion, 
                 memory_capacity=MEMORY_CAPACITY, 
                 device=device,
                 gamma=GAMMA, 
                 tau=TAU, 
                 batch_size=BATCH_SIZE, 
                 policy_noise=0.2,
                 noise_clip=0.5, 
                 policy_freq=2
                )

def record_trigger(episode_id: int) -> bool:
    return True

format_type = "stories"  # 'stories'
name_prefix = "BipedalWalker-TD3_test_v2"

# Setup environment with InfoOverlay and RecordVideo
env = gym.make(ENV_NAME, render_mode="rgb_array")
env = InfoOverlay(env, format_type = format_type)
env = RecordVideo(env, video_folder=f"{BASE_DIR}/VideosBipedalWalker_TD3_v2_Test", name_prefix="BipedalWalker-TD3_train_v2_Test", episode_trigger=record_trigger)


tester = Tester(env=env, agent=agent)

try:
    
    tester.test(num_episodes=NUM_EPISODES)



    env.close()

except KeyboardInterrupt:



    env.close()
