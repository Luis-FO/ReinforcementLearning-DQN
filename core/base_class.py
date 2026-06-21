from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):

    def __init__(self,
                 env,
                 device):
        
        # self.policy = policy
        self.env =  env
        self.device = device
    
    def train(self):
        raise NotImplementedError("Train method not implemented yet.")
    
    @abstractmethod
    def _update(self):
        """Update the agent's networks based on a batch of experiences."""
        pass
    
    @abstractmethod
    def select_action(self, obs, training=True):
        """Select action for given observation."""
        pass    

    def predict(self, obs):
        """Predict action for given observation."""
        pass

    def save(self):
        pass

    def load(self):
        pass