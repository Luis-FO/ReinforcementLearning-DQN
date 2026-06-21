import torch
import torch.nn as nn

from core.distributions import BaseDistribution, CategoricalDistribution, NormalDistribution

class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))


class ActorCritic(BaseNetwork):


    # TODO: Add option for distribution type (discrete vs continuous) and 
    # handle action sampling accordingly
    def __init__(self, state_dim, action_dim, distribution_class: BaseDistribution):
        super(ActorCritic, self).__init__()
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
        self.log_std = None
        self._build()

    def _build(self):
        if isinstance(self.action_dist, NormalDistribution):
            # TODO: Change the way latent_dim is extracted to be more robust to changes in the actor architecture
            action_net, log_std = self.action_dist.proba_distribution_net(self.actor[-2].out_features)
            self.action_net = action_net
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
        # action_logits = self.actor(state)
        latent_pi = self.actor(state)

        state_value = self.critic(state)
        # Sample an action from the distribution
        dist = self._get_action_distribution(latent_pi)
        actions = dist.get_actions()
        log_prob = dist.log_prob(actions)

        # Return both the action and its log probability for later use in the update step
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
    

class DQNCNN(nn.Module):

    def __init__(self, num_obs, num_actions):
        super(DQNCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(num_obs[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Calcula o tamanho da saída da CNN dinamicamente para permitir diferentes tamanhos de ecrã
        with torch.no_grad():
            dummy_input = torch.zeros(1, *num_obs)
            cnn_out_dim = self.features(dummy_input).flatten().shape[0]
            
        self.net = nn.Sequential(
            nn.Linear(cnn_out_dim, 512), 
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.net(x)
    
    @property
    def output_size(self):
        return self.net[-1].out_features