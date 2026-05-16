from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):

    def __init__(self,
                 env,
                 device):
        
        # self.policy = policy
        self.env =  env
        self.device = device

    # @abstractmethod
    # def _setup_model(self) -> None:
    #     """Create networks, buffer and optimizers."""
    #     pass

    # @abstractmethod
    # def learn(self):
    #     """Main learning loop."""
    #     pass
    
    def train(self):
        raise NotImplementedError("Train method not implemented yet.")

    def predict(self, obs):
        """Predict action for given observation."""
        pass

    def save(self):
        pass

    def load(self):
        pass