
import torch
from torch import nn
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from pathlib import Path

from core.distributions import NormalDistribution
from core.ppo_agent import PPOAgent

from core.logtrigger import segmented_limit_trigger
from core.info_overlay import InfoOverlay


# Diretório onde o script está localizado
BASE_DIR = Path(__file__).resolve().parent
# Cria diretórios para salvar modelos e vídeos, se não existirem
(BASE_DIR / "model").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "VideosPendulumV2").mkdir(parents=True, exist_ok=True)

ENV_NAME = 'Pendulum-v1' 
LR = 3e-4  # 0.0005  
GAMMA = 0.99 # 0.95 # Fator de desconto
EPS_CLIP = 0.2        # Parâmetro de clipping do PPO

NUM_EPISODES = 1500

K_EPOCHS = 20          # Quantas vezes atualizar a rede com o mesmo batch
    
UPDATE_TIMESTEPS = 2048 # Atualizar a cada X passos



device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

temp_env = gym.make(ENV_NAME)

state_dim = temp_env.observation_space.shape[0]
action_dim = temp_env.action_space.shape[0]
max_action = float(temp_env.action_space.high[0])

temp_env.close()


format_type = "stories"  # 'stories'
name_prefix = "Pendulum-PPO_train"

# Setup environment with InfoOverlay and RecordVideo
dqn_env = gym.make(ENV_NAME, render_mode="rgb_array")
dqn_env = InfoOverlay(dqn_env, format_type = format_type)
dqn_env = RecordVideo(dqn_env, video_folder=f"{BASE_DIR}/VideosPendulumV2", name_prefix="Pendulum-PPO_train_v1", episode_trigger=segmented_limit_trigger)



agent = PPOAgent(
    env=dqn_env,
    state_dim=state_dim,
    action_dim=action_dim,
    learning_rate=LR,
    device=device,
    distribution_class=NormalDistribution,
    gamma=GAMMA,
    eps_clip=EPS_CLIP,
    k_epochs=K_EPOCHS,
)

try:
    
    agent.train(total_steps=NUM_EPISODES * UPDATE_TIMESTEPS, rollout_size=UPDATE_TIMESTEPS)
    agent.save(f"{BASE_DIR}/model/Pendulum_PPO.pt")

except KeyboardInterrupt:
    print("\nTreinamento interrompido. Salvando modelo atual")
    agent.save(f"{BASE_DIR}/model/Pendulum_PPO_interrompido.pt")


