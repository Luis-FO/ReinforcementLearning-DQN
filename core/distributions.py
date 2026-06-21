from torch.distributions import Categorical
from torch.distributions import Normal
from torch import nn
import torch

__all__ = ["CategoricalDistribution", "NormalDistribution", "BaseDistribution"]

class BaseDistribution:
    def __init__(self):
        pass
    
    def proba_distribution_net(self, *args, **kwargs):
        raise NotImplementedError
    
    def proba_distribution(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_actions(self):
        raise NotImplementedError

    def log_prob(self, action):
        raise NotImplementedError
    

class CategoricalDistribution(BaseDistribution):
    
    dist: Categorical

    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim):
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits
    
    def proba_distribution(self, logits):
        self.dist = Categorical(logits=logits)
        return self
    
    def get_actions(self):
        return self.dist.sample()

    def log_prob(self, action):
        return self.dist.log_prob(action)
    
    def entropy(self):
        return self.dist.entropy()
    
class NormalDistribution(BaseDistribution):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def get_actions(self):
        return self.dist.rsample()

    def proba_distribution_net(self, latent_dim, log_std_init=0.0):
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std
    
    def proba_distribution(self, mean: torch.Tensor, log_std: torch.Tensor):
        std = torch.ones_like(mean) * log_std.exp()  
        self.dist = Normal(mean, std)
        return self
    
    def log_prob(self, action):
        # Sum to get total log probability for multi-dimensional actions
        # [log_prob_dim1, log_prob_dim2, ...] -> log_prob_dim1 + log_prob_dim2 + ...
        return self.dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions
    
    def entropy(self):
        return self.dist.entropy().sum(dim=-1)  # Sum over action dimensions