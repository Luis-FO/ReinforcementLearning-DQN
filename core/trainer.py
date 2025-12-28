import gymnasium as gym

class Trainer():

    def __init__(self, env, agent):
        self.env_name = env.spec.id
        self.env = env
        self.agent = agent
        self.steps = 0 
    
    def train(self, num_episodes):
        episode_rewards = []

        # Training loop
        for episode in range(num_episodes):
            # Resetando o ambiente no início de cada episódio
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            # Loop até o episódio terminar
            while not done:
                # Selecionar ação com política epsilon-greedy.
                action = self.agent.select_action(obs, training = True)
                self.steps+=1

                # Executar ação no ambiente
                next_obs, reward, terminate, truncate, _ = self.env.step(action)
                total_reward += reward
                done = terminate or truncate
                
                # Armazenar a transição na memória de replay
                self.agent.memory.push(obs, action, reward, next_obs, done)
                self.agent.update()
                obs = next_obs
            
            # Decair epsilon após cada episódio
            self.agent.exploration_strategy.decay()
            episode_rewards.append(total_reward)
            print(f"Episódio {episode}: Recompensa Total = {total_reward}, Epsilon = {self.agent.exploration_strategy.epsilon:.4f}  ")

        self.env.close()
        
        if len(episode_rewards) >= 100:
            final_metric = sum(episode_rewards[-100:]) / 100
        else:
            # Caso o treino seja muito curto ou interrompido
            final_metric = sum(episode_rewards) / len(episode_rewards) if len(episode_rewards) > 0 else -1000

        print(f"Treino concluído. Métrica final (média de 100): {final_metric}")
        return final_metric
        

if __name__ == "__main__":
    pass