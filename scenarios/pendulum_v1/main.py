
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from pathlib import Path

from core.trainer import Trainer
from core.logtrigger import segmented_limit_trigger
from core.info_overlay import InfoOverlay
from core.ddpg_agent import DDPGAgent
from core.networks import Actor, Critic
from core.exploration import GaussianDecayNoise

# Diretório onde o script está localizado
BASE_DIR = Path(__file__).resolve().parent
# Cria diretórios para salvar modelos e vídeos, se não existirem
(BASE_DIR / "model").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "VideosPendulumV2").mkdir(parents=True, exist_ok=True)

ENV_NAME = 'Pendulum-v1' 
LR_ACTOR = 0.0001
LR_CRITIC = 0.001

MEMORY_CAPACITY = 100000
BATCH_SIZE = 64
GAMMA = 0.95
TAU = 0.005
NUM_EPISODES = 1500

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
critic = Critic(state_dim, action_dim).to(device)

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
        exploration_strategy=noise_strategy,
        memory_capacity=MEMORY_CAPACITY,
        gamma=GAMMA,
        tau=TAU,
        batch_size=BATCH_SIZE,
        device=device
)


format_type = "stories"  # 'stories'
name_prefix = "Pendulum-dqn_train"

def record_trigger(episode_id: int) -> bool:
    if episode_id < 10:
        return True
    elif episode_id < 50:
        return episode_id % 5 == 0
    else:
        return episode_id % 15 == 0
    
# Setup environment with InfoOverlay and RecordVideo
dqn_env = gym.make(ENV_NAME, render_mode="rgb_array")
dqn_env = InfoOverlay(dqn_env, format_type = format_type)
dqn_env = RecordVideo(dqn_env, video_folder=f"{BASE_DIR}/VideosPendulumV2", name_prefix="Pendulum-dqn_train_v1", episode_trigger=record_trigger)

trainer = Trainer(env=dqn_env, agent=agent)

try:
    
    trainer.train(num_episodes=NUM_EPISODES)

    agent.actor.save(f"{BASE_DIR}/model/Pendulum_actor.pt")
    agent.critic.save(f"{BASE_DIR}/model/Pendulum_critic.pt")

    dqn_env.close()

except KeyboardInterrupt:
    print("\nTreinamento interrompido. Salvando modelo atual")
    agent.actor.save(f"{BASE_DIR}/model/Pendulum_actor_interrompido.pt")
    agent.critic.save(f"{BASE_DIR}/model/Pendulum_critic_interrompido.pt")

    dqn_env.close()
