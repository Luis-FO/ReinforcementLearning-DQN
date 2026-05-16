from abc import ABC, abstractmethod
import numpy as np


class ExplorationStrategy(ABC):
    """
    Interface para estratégias de exploração com estado interno.
    """
    @abstractmethod
    def get_value(self, action_dim=None):
        """Retorna o valor atual (ruído ou epsilon) baseado no estado interno."""
        pass

    def decay(self):
        """Optional: decay the internal state (e.g., epsilon, noise std)."""
        return None

    def reset(self):
        """Optional: reset internal state at the start of training/episode."""
        return None

    def on_step(self):
        """Optional: called on every environment step if agents choose to."""
        return None

    def on_train_start(self):
        """Optional: called at the beginning of training loops."""
        return None


class FixedEpsilonGreedy(ExplorationStrategy):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def get_value(self, action_dim=None):
        return self.epsilon
    
    def decay(self):
        pass
    def reset(self):
        pass


class DecayingExplorationStrategy(ExplorationStrategy):
    """Interface segregada: Adiciona comportamento de decaimento e reinício."""
    @abstractmethod
    def decay(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class GaussianNoise(ExplorationStrategy):
    def __init__(self, std=0.1):
        self.std = std

    def get_value(self, action_dim=1):
        return np.random.normal(0, self.std, size=action_dim)

class GaussianDecayNoise(DecayingExplorationStrategy):
    """
    Estratégia para Espaço Contínuo (DDPG).
    """
    def __init__(self, start_std=1.0, min_std=0.1, decay_rate=0.995):
        self.start_std = start_std
        self.current_std = start_std
        self.min_std = min_std
        self.decay_rate = decay_rate

    def get_value(self, action_dim=1):
        return np.random.normal(0, self.current_std, size=action_dim)

    def decay(self) -> float:
        self.current_std = max(self.min_std, self.current_std * self.decay_rate)
        return self.current_std

    def reset(self):
        self.current_std = self.start_std

# New class that decay epsilon Greedy based on steps, not episodes, and has a reset method to restart the decay process.
class LinearDecayEpsilonGreedy(DecayingExplorationStrategy):
    def __init__(self, start=1.0, end=0.05, decay_steps=10000):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.current_epsilon = start
        self.steps = 0

    def get_value(self, action_dim=None):
        return self.current_epsilon

    def decay(self):
        if self.steps < self.decay_steps:
            self.current_epsilon = self.start - (self.start - self.end) * (self.steps / self.decay_steps)
            self.steps += 1
        else:
            self.current_epsilon = self.end
        return self.current_epsilon
    
    def reset(self):
        self.current_epsilon = self.start
        self.steps = 0

class ExponentialDecayEpsilonGreedy(DecayingExplorationStrategy):
    def __init__(self, start=1.0, end=0.05, decay_steps=10000):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.current_epsilon = start
        self.steps = 0

    def get_value(self, action_dim=None):
        return self.current_epsilon

    def decay(self):
        if self.steps < self.decay_steps:
            self.current_epsilon = self.end + (self.start - self.end) * np.exp(-self.steps / self.decay_steps)
            self.steps += 1
        else:
            self.current_epsilon = self.end
        return self.current_epsilon
    
    def reset(self):
        self.current_epsilon = self.start
        self.steps = 0


class EpsilonGreedyStrategy(DecayingExplorationStrategy):
    """
    Estratégia para Espaço Discreto (DQN).
    """
    def __init__(self, start=1.0, end=0.05, decay_rate=0.99):
        self.start = start
        self.end = end
        self.decay_rate = decay_rate
        self.current_epsilon = start

    def get_value(self, action_dim=None):
        return self.current_epsilon

    def decay(self):
        self.current_epsilon = max(self.end, self.current_epsilon * self.decay_rate)
        return self.current_epsilon
    
    def reset(self):
        self.current_epsilon = self.start