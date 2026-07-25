import torch
import torch.nn as nn

from core.distributions import BaseDistribution, CategoricalDistribution, NormalDistribution

class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path, **kwargs):
        self.load_state_dict(torch.load(path, **kwargs))


class ActorCritic(BaseNetwork):
    """
    Actor-Critic network for PPO. Can be used with different action distributions (e.g., Categorical for discrete actions, Normal for continuous actions).

    """
    def __init__(self, state_dim, action_dim, distribution_class: BaseDistribution):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        assert distribution_class is not None, "distribution_class must be provided (e.g., CategoricalDistribution or NormalDistribution)"
        self.distribution_class = distribution_class
        self.action_dist = distribution_class(action_dim)
        self.action_net = None
        self.log_std = None
        self._build()
    
    def save_checkpoint(self, path):

        checkpoint = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "distribution": self.distribution_class.__name__,
            "model_state_dict": self.state_dict(),
        }

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path, distribution_registry, device="cpu"):

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        distribution_class = distribution_registry[
            checkpoint["distribution"]
        ]

        model = cls(
            state_dim=checkpoint["state_dim"],
            action_dim=checkpoint["action_dim"],
            distribution_class=distribution_class,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        return model

    def _build(self):
        if isinstance(self.action_dist, NormalDistribution):
            # TODO: Change the way latent_dim is extracted to be more robust to changes in the actor architecture
            # Actually, the last layer of the actor is a nn.Tanh(), so we need to get the second to last layer's output features
            self.action_net, log_std = self.action_dist.proba_distribution_net(self.actor[-2].out_features)
            # self.action_net = action_net
            self.log_std = nn.Parameter(log_std.data)
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(self.actor[-2].out_features)
        else:
            raise NotImplementedError("Unsupported distribution class")
        
    def _predict(self, state):
        action_logits = self.actor(state)
        state_value = self.critic(state)
        return action_logits, state_value
    
    def act(self, state):
        # Compute action probabilities based on current policy
        latent_pi = self.actor(state)

        state_value = self.critic(state)
        # Sample an action from the distribution
        dist = self._get_action_distribution(latent_pi)
        actions = dist.get_actions()
        log_prob = dist.log_prob(actions)

        # Return the action and its log probability along with the state value
        return actions, log_prob, state_value

    def _get_action_distribution(self, latent_pi):

        mean_actions = self.action_net(latent_pi)
        
        if isinstance(self.action_dist, CategoricalDistribution):
            return self.action_dist.proba_distribution(mean_actions)
        elif isinstance(self.action_dist, NormalDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        else:
            raise ValueError("Unsupported distribution class")
        
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        latent_pi = self.actor(state)
        dist = self._get_action_distribution(latent_pi)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state).squeeze(-1)
        return action_logprobs, state_values, dist_entropy
    
    def __str__(self):
        return f"{self.state_dict()}"
    

class Actor(BaseNetwork):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=400):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.net(x)

class Critic(BaseNetwork):
    def __init__(self, state_dim, action_dim, hidden_dim=400):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))
    

class DQN(nn.Module):

    def __init__(self, num_obs, num_actions):
        super(DQN, self).__init__()

        self.net = nn.Sequential(nn.Linear(num_obs, 64), 
                                 nn.ReLU(),
                                 nn.Linear(64, 64), 
                                 nn.ReLU(),
                                 nn.Linear(64, num_actions))
        

    def forward(self, x):
        return self.net(x)
    
    @property
    def output_size(self):
        return self.net[-1].out_features
