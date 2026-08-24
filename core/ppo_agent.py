import torch
import torch.optim as optim
import torch.nn.functional as F
from core.on_policy_algorithm import OnPolicyAlgorithm
from core.buffers import RolloutBuffer
from core.networks import ActorCritic
from core.distributions import BaseDistribution, CategoricalDistribution, NormalDistribution

class PPOAgent(OnPolicyAlgorithm):
    
    policy: ActorCritic
    distribution_class: BaseDistribution
    def __init__(self, env, state_dim, action_dim, learning_rate, device, 
                 distribution_class: BaseDistribution, gamma=0.99, gae_lambda=0.95, eps_clip=0.2, 
                 k_epochs=4, vf_coef=0.5, ent_coef=0.01):
        super().__init__(env, learning_rate, device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        assert distribution_class is not None, "distribution_class must be provided (e.g., CategoricalDistribution or NormalDistribution)"
        self.distribution_class = distribution_class
        self.policy = ActorCritic(state_dim, action_dim, distribution_class).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.policy_old = ActorCritic(state_dim, action_dim, distribution_class).to(device)
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
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        
        advantages = self.rollout_buffer.advantages
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns = self.rollout_buffer.returns
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        # TODO: Normalize returns directly in the buffer and remove normalization from here
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        for epoch in range(self.k_epochs):
            logprobs, values, dist_entropy = self.policy.evaluate(states, actions)
            values = values.squeeze(-1)
            ratio = torch.exp(logprobs - old_log_probs)  # pi(a|s) / pi_old(a|s)
            surr1 = ratio * advantages 
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1+self.eps_clip)*advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = F.mse_loss(returns, values)
            
            # Ensure the entropy term is reduced to a scalar before backward
            entropy_loss = dist_entropy.mean()
            loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, path):
        self.policy.save_checkpoint(path)

    def load(self, path):
        # TODO: Move this to a more appropriate place, maybe in the ActorCritic class or a separate utility function
        distribution_registry = {
            'CategoricalDistribution': CategoricalDistribution,
            'NormalDistribution': NormalDistribution,
        }
        self.policy = ActorCritic.load_checkpoint(path, distribution_registry, device=self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())