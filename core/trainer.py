import gymnasium as gym

class Trainer():

    def __init__(self, env, agent, warmup_steps=0):
        self.env_name = env.spec.id
        self.env = env
        self.agent = agent
        self.steps = 0 
        self.warmup_steps = warmup_steps
    
    def train(self, num_episodes):
        episode_rewards = []

        # TODO: Evitar o uso de hasattr, talvez com um padrão Strategy mais robusto.
        if hasattr(self.agent.exploration_strategy, 'reset'):
            self.agent.exploration_strategy.reset()

        # Training loop
        for episode in range(num_episodes):
            # Resetando o ambiente no início de cada episódio
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            # Loop até o episódio terminar
            while not done:
                if self.steps < self.warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    # Selecionar ação com política epsilon-greedy.
                    action = self.agent.select_action(obs, training = True)
                

                # Executar ação no ambiente
                next_obs, reward, terminate, truncate, _ = self.env.step(action)
                
                done = terminate or truncate

                # Armazenar a transição na memória de replay
                self.agent.memory.push(obs, action, reward, next_obs, done)

                # Atualizar o agente após cada passo se estiver fora do warmup
                if self.steps >= self.warmup_steps:
                    self.agent.update()
                
                obs = next_obs
                total_reward += reward
                self.steps+=1

            # Decair o valor de exploração se aplicável
            if hasattr(self.agent.exploration_strategy, 'decay') and self.steps >= self.warmup_steps:
                exploration_value = self.agent.exploration_strategy.decay()
                print(f"Episódio {episode}: Recompensa = {total_reward:.2f}, Exploration = {exploration_value:.4f}")
            else:
                status = "Warm Up" if self.steps < self.warmup_steps else "Treinando"
                print(f"Episódio {episode}: Recompensa {total_reward:.2f} | Status: {status} | Total Steps: {self.steps}")

            episode_rewards.append(total_reward)
            

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