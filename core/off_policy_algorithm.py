from abc import ABC, abstractmethod

from core.base_class import BaseAlgorithm
from random import random
import torch
import numpy as np

class OffPolicyAlgorithm(BaseAlgorithm, ABC):
    def __init__(self, env, device, warmup_steps=1000):
        super().__init__(env, device)

        # TODO: How to force type of memory here?
        self.memory = None
        self.warmup_steps = warmup_steps


    def on_train_start(self):
        pass

    def on_step(self):
        pass

    def _store_transition(self, obs, action, reward, next_obs, done):
        self.memory.push(obs, action, reward, next_obs, done)

    @abstractmethod
    def _update(self):
        pass

    @abstractmethod
    def select_action(self, obs, training=True):
        pass

    def setEnv(self, env):
        self.env = env

    def get_info(self):
        pass

    def train(self, total_steps):
        episode_rewards = []

        self.on_train_start()

        # Training loop
        steps = 0
        episode = 0
        while steps < total_steps:
            # Resetando o ambiente no início de cada episódio
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            # Loop até o episódio terminar ou atingir o limite de passos
            while not done and steps < total_steps:
                if steps < self.warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    # Selecionar ação com política epsilon-greedy.
                    action = self.select_action(obs, training = True)
                

                # Executar ação no ambiente
                next_obs, reward, terminate, truncate, _ = self.env.step(action)
                
                done = terminate or truncate

                # Armazenar a transição na memória de replay
                self._store_transition(obs, action, reward, next_obs, done)

                # Atualizar o agente após cada passo se estiver fora do warmup
                if steps >= self.warmup_steps:
                    self._update()
                
                obs = next_obs
                total_reward += reward
                steps+=1
                self.on_step()

            # Decair o valor de exploração se aplicável
            if steps >= self.warmup_steps:
                info = self.get_info()
                print(f"Episódio {episode}: Recompensa = {total_reward:.2f} {info}")
            else:
                status = "Warm Up" if steps < self.warmup_steps else "Treinando"
                print(f"Episódio {episode}: Recompensa {total_reward:.2f} | Status: {status} | Total Steps: {steps}")

            episode_rewards.append(total_reward)
            episode += 1

        self.env.close()
        
        if len(episode_rewards) >= 100:
            final_metric = sum(episode_rewards[-100:]) / 100
        else:
            # Caso o treino seja muito curto ou interrompido
            final_metric = sum(episode_rewards) / len(episode_rewards) if len(episode_rewards) > 0 else -1000

        print(f"Treino concluído. Métrica final (média de 100): {final_metric}")
        return final_metric
        
    def test(self, num_episodes):
        episode_rewards = []

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.select_action(obs, training=False)
                next_obs, reward, terminate, truncate, _ = self.env.step(action)

                done = terminate or truncate
                obs = next_obs
                total_reward += reward

            episode_rewards.append(total_reward)
            print(f"Teste Episódio {episode + 1}/{num_episodes}: Recompensa = {total_reward:.2f}")

        self.env.close()

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        print(f"\n--- Teste Concluído ---")
        print(f"Média: {mean_reward:.2f} +/- {std_reward:.2f}")

        return mean_reward, std_reward

    # # Check if I should put this method in the base class, since it's used in both on-policy and off-policy algorithms
    # def _update(self):
    #     pass

    # def _soft_update(self, target_net, policy_net):
    #     pass
    
    # def save(self):
    #     pass

    # def load(self):
    #     pass