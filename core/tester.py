import numpy as np

class Tester:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def test(self, num_episodes):

        episode_rewards = []
        
        # print(f"--- Iniciando Teste no ambiente: {self.env.spec.id if self.env.spec else 'Unknown'} ---")

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # IMPORTANTE: training=False desativa a exploração (epsilon-greedy).
                # O agente deve escolher apenas a melhor ação conhecida (Greedy).
                action = self.agent.select_action(obs, training=False)
                
                # Executa a ação
                next_obs, reward, terminate, truncate, _ = self.env.step(action)
                
                done = terminate or truncate
                obs = next_obs
                total_reward += reward
                
                # Nota: Não chamamos agent.update() nem memory.push() aqui.
                # O objetivo é apenas avaliar a performance do modelo congelado.

            episode_rewards.append(total_reward)
            print(f"Teste Episódio {episode + 1}/{num_episodes}: Recompensa = {total_reward:.2f}")

        self.env.close()

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        print(f"\n--- Teste Concluído ---")
        print(f"Média: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        return mean_reward, std_reward
