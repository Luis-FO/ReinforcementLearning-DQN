from abc import ABC, abstractmethod
import numpy as np

class ExplorationStrategy(ABC):
    """
    Interface genérica para estratégias de exploração.
    Pode retornar um escalar (Epsilon para DQN) ou um Tensor/Array (Ruído para DDPG).
    """
    @abstractmethod
    def get_value(self, action_dim, current_episode=None):
        pass

    @abstractmethod
    def decay(self):
        pass


class GaussianDecayNoise(ExplorationStrategy):
    """
    Estratégia para Espaço Contínuo (DDPG/TD3).
    Retorna: Vetor de ruído para ser SOMADO à ação.
    """
    def __init__(self, initial_std=0.1, decay_rate=50):
        self.initial_std = initial_std
        self.decay_rate = decay_rate

    def get_value(self, action_dim, current_episode=0):
        std = max(self.initial_std, 1.0 - current_episode / self.decay_rate)
        return np.random.normal(0, std, size=action_dim)
    
    def decay(self):
        return super().decay()

class EpsilonGreedyStrategy(ExplorationStrategy):
    """
    Estratégia para Espaço Discreto (DQN).
    Retorna: Escalar (float) representando a probabilidade de exploração.
    """
    def __init__(self, start=1.0, end=0.05, decay=0.99):
        self.epsilon = start
        self.end = end
        self.decay_rate = decay

    def get_value(self, action_dim=None, current_episode=None):
        return self.epsilon
    
    def decay(self):
        self.epsilon = max(self.end, self.epsilon * self.decay_rate)