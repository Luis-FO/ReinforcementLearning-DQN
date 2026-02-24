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
    
# 1. Criar o ambiente
env = gym.make("LunarLander-v3", render_mode="rgb_array")  # Usando o ambiente clássico LunarLander-v3
env = InfoOverlay(env, format_type = format_type)
env = RecordVideo(env, video_folder=f"{BASE_DIR}/VideosLunarLander_PPO", name_prefix=name_prefix, episode_trigger=record_trigger)
# 2. Instanciar o modelo PPO
# MlpPolicy significa que estamos usando uma rede neural padrão (não convolucional)
model = PPO("MlpPolicy", env, verbose=1)

# 3. Treinar o agente
print("Iniciando treinamento...")
model.learn(total_timesteps=100000)

# 4. Salvar o modelo
model.save("ppo_lunar_lander")

env.close()
env = gym.make("LunarLander-v3", render_mode="human") 
# Testar
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    env.render()
    if dones or truncated:
        obs, _ = env.reset()