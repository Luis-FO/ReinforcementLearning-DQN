import torch

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from pathlib import Path

from core.ppo_agent import PPOAgent

from core.logtrigger import segmented_limit_trigger
from core.info_overlay import InfoOverlay
from core.distributions import CategoricalDistribution

# Diretório onde o script está localizado
BASE_DIR = Path(__file__).resolve().parent
# Cria diretórios para salvar modelos e vídeos, se não existirem
(BASE_DIR / "model").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "VideosCartPole_PPO").mkdir(parents=True, exist_ok=True)

ENV_NAME = 'CartPole-v1' 
LR = 3e-4  # 0.0005  
GAMMA = 0.99 # 0.95 # Fator de desconto
EPS_CLIP = 0.2        # Parâmetro de clipping do PPO

NUM_EPISODES = 1500

K_EPOCHS = 10          # Quantas vezes atualizar a rede com o mesmo batch
    
UPDATE_TIMESTEPS = 4096 # Atualizar a cada X passos



device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

temp_env = gym.make(ENV_NAME)
obs, info = temp_env.reset()
n_observations = len(obs)
n_actions = temp_env.action_space.n
temp_env.close()


format_type = "stories"  # 'stories'
name_prefix = "CartPole-PPO_train_v1"

# Setup environment with InfoOverlay and RecordVideo
dqn_env = gym.make(ENV_NAME, render_mode="rgb_array")
# dqn_env = InfoOverlay(dqn_env, format_type = format_type)
# dqn_env = RecordVideo(dqn_env, video_folder=f"{BASE_DIR}/VideosCartPole_PPO", name_prefix=name_prefix, episode_trigger=segmented_limit_trigger)

agent = PPOAgent(env=dqn_env,
                 state_dim=n_observations,
                action_dim=n_actions,
                learning_rate=LR,
                device=device,
                distribution_class=CategoricalDistribution,
                gamma=GAMMA,
                ent_coef=0.01,
                gae_lambda=0.95,
                eps_clip=EPS_CLIP,
                k_epochs=K_EPOCHS
                )

try:
    
    agent.train(total_steps=NUM_EPISODES * UPDATE_TIMESTEPS, rollout_size=UPDATE_TIMESTEPS)
    agent.save(f"{BASE_DIR}/model/CartPole_PPO.pt")

except KeyboardInterrupt:
    print("\nTreinamento interrompido. Salvando modelo atual")
    agent.save(f"{BASE_DIR}/model/CartPole_PPO_interrompido.pt")


