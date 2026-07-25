import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from gymnasium.wrappers import RecordVideo
from core.ppo_agent import PPOAgent
from core.distributions import CategoricalDistribution
from core.networks import ActorCritic
from core.info_overlay import InfoOverlay



BASE_DIR = Path(__file__).resolve().parent
ENV_NAME = "CartPole-v1"
MODEL_PATH = BASE_DIR / "model" / "CartPole_PPO_Interrompido.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Espaços do ambiente
temp_env = gym.make(ENV_NAME)
obs, _ = temp_env.reset()
n_observations = len(obs)
n_actions = temp_env.action_space.n
temp_env.close()

format_type = "stories"  # 'stories'
name_prefix = "CartPole-PPO_test_v1"
# Ambiente de teste sem gravação de vídeo
env = gym.make(ENV_NAME, render_mode="rgb_array")
env = InfoOverlay(env, format_type=format_type)  # Adiciona informações na tela
env = RecordVideo(env, video_folder=f"{BASE_DIR}/VideosCartPole_PPO_TEST", name_prefix=name_prefix, episode_trigger=lambda episode_id: True)  # Grava todos os episódios

agent = PPOAgent(
    env=env,
    state_dim=n_observations,
    action_dim=n_actions,
    learning_rate=3e-4,
    device=device,
    distribution_class=CategoricalDistribution,
    gamma=0.99,
    gae_lambda=0.95,
    eps_clip=0.2,
    k_epochs=10,
    ent_coef=0.01,
)
# print(agent.policy)


agent.load(str(MODEL_PATH))

# Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in agent.policy.state_dict():
#     print(param_tensor, "\t", agent.policy.state_dict()[param_tensor].size())
num_episodes = 10
episode_rewards = []

for episode in range(1, num_episodes + 1):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _, _ = agent.select_action(obs, training=False)
        action = int(np.asarray(action).item())

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    episode_rewards.append(total_reward)
    print(f"Episódio {episode}/{num_episodes}: recompensa = {total_reward:.2f}")

env.close()

mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
print(f"\nRecompensa média: {mean_reward:.2f} ± {std_reward:.2f}")