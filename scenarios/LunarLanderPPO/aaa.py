import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
from core.info_overlay import InfoOverlay
from pathlib import Path

# Diretório onde o script está localizado
BASE_DIR = Path(__file__).resolve().parent
# Cria diretórios para salvar modelos e vídeos, se não existirem
(BASE_DIR / "model").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "VideosLunarLander_PPO").mkdir(parents=True, exist_ok=True)

format_type = "stories"  # 'stories'
name_prefix = "LunarLander-PPO_train"

def record_trigger(episode_id: int) -> bool:
    if episode_id < 10:
        return True
    elif episode_id < 50:
        return episode_id % 5 == 0
    else:
        return episode_id % 15 == 0
    
# 1. Configuração do ambiente
# Usamos 'LunarLanderContinuous-v3' (versão mais recente do Gymnasium)
env_id = "LunarLanderContinuous-v3"
env = gym.make(env_id, render_mode="rgb_array")
env = InfoOverlay(env, format_type = format_type)
env = RecordVideo(env, video_folder=f"{BASE_DIR}/VideosLunarLander_PPO", name_prefix=name_prefix, episode_trigger=record_trigger)

# 2. Definição do Modelo
# MlpPolicy é ideal para entradas de vetores (sensores do lander)
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.0003, # Taxa de aprendizado padrão
    gamma=0.999,          # Fator de desconto para focar no longo prazo
    device="auto"         # Usa GPU se disponível
)

print("--- Iniciando Treinamento ---")
# 500k passos costumam ser suficientes para um pouso estável
model.learn(total_timesteps=500000)

# 3. Salvar o agente
model.save("ppo_lunar_lander_continuous")
print("Modelo salvo!")

# 4. Avaliação
# Verificamos a recompensa média em 10 episódios
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Recompensa média: {mean_reward:.2f} +/- {std_reward:.2f}")

env.close()