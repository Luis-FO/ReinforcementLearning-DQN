import torch
import torch.nn as nn
from torch.distributions import Categorical

class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))


class ActorCritic(BaseNetwork):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        # Compute Value function from the same state input
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def act(self, state):
        # Compute action probabilities based on current policy
        action_logits = self.actor(state)
        # Sample an action from the distribution
        dist = Categorical(logits=action_logits)
        action = dist.sample() # Return the sampled index (as a tensor) of the action in the action space.
        state_value = self.critic(state)
        # Return both the action and its log probability for later use in the update step
        return action.detach(), dist.log_prob(action).detach(), state_value.detach()
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        action_logits = self.actor(state)
        dist = Categorical(logits=action_logits)
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