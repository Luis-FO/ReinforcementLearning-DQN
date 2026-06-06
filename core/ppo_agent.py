import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from core.on_policy_algorithm import OnPolicyAlgorithm
from core.buffers import RolloutBuffer
from core.networks import ActorCritic

class PPOAgent(OnPolicyAlgorithm):
    
    policy: ActorCritic

    def __init__(self, env, state_dim, action_dim, learning_rate, device):
        super().__init__(env, learning_rate, device)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.eps_clip = 0.2
        self.k_epochs = 4
        self.vf_coef = 0.5
        self.ent_coef = 0.01
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.rollout_buffer = RolloutBuffer(size=2048)

    @torch.no_grad()
    def select_action(self, state, training=True):
        
        obs_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
 
        action_t, log_prob_t, value_t = self.policy_old.act(obs_t)
 
        action   = action_t.squeeze(0).cpu().numpy()
        log_prob = log_prob_t.item()
        value    = value_t.item()
 
        return action, log_prob, value

    def _update(self):

        states, actions, rewards, values, next_states, dones, old_log_probs = self.rollout_buffer.sample()
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        
        advantages = self.rollout_buffer.advantages
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns = self.rollout_buffer.returns
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        for _ in range(self.k_epochs):
            logprobs, values, dist_entropy = self.policy.evaluate(states, actions)
            values = values.squeeze(-1)
            ratio = torch.exp(logprobs - old_log_probs)  # pi(a|s) / pi_old(a|s)
            surr1 = ratio * advantages 
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1+self.eps_clip)*advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Dif entre returns e valores preditos pelo crítico
            critic_loss = F.mse_loss(returns, values)
            
            # Ensure the entropy term is reduced to a scalar before backward
            entropy_loss = dist_entropy.mean()
            loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
        self.policy_old.load_state_dict(self.policy.state_dict())


    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_old.load_state_dict(self.policy.state_dict())