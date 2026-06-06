from core.base_class import BaseAlgorithm
from core.buffers import RolloutBuffer
import numpy as np

class OnPolicyAlgorithm(BaseAlgorithm):

    rollout_buffer: RolloutBuffer
    
    def __init__(self, env, learning_rate, device):
        super().__init__(env, device)

        self.rollout_buffer = None # TODO: Explicitly define type of rollout_buffer

    def learn(self):
        pass
    
    # TODO: values ans log_probs should raise an error if not provided on PPO and be optional for other algorithms
    def _store_transition(self, obs, action, reward, next_obs, done, values=None, log_probs=None):
        self.rollout_buffer.add(
            obs, action, reward, 
            next_obs, done, values,
            log_probs,
        )

    # TODO: Use just fixed steps, and reset should be called when done is True
    def collect_rollouts(self, n_steps):
        obs, _ = self.env.reset()
        done = False
        steps = 0
        total_reward = 0
        while not done and steps < n_steps:
            action, log_prob, value = self.select_action(obs, training=True)
            next_obs, reward, terminate, truncate, _ = self.env.step(action)

            total_reward += reward
            done = terminate or truncate

            # Armazenar a transição na memória
            self._store_transition(obs, action, reward, next_obs, done, value, log_prob)

            obs = next_obs
            steps += 1
        
        _, _, last_value = self.select_action(obs, training=True)
        # TODO: Use namedtuple for the buffer sample return
        # Calcular GAE e returns
        self.rollout_buffer._compute_returns_and_advantage(
            rewards=np.array(self.rollout_buffer.rewards),
            values=np.array(self.rollout_buffer.values),
            dones=np.array(self.rollout_buffer.is_terminals),
            last_value=last_value
        )
        return steps, total_reward
    
    def train(self, total_steps):
        steps = 0
        remaining_steps = total_steps
        while steps < total_steps:
            # Collect Rollouts
            n_steps, total_reward = self.collect_rollouts(n_steps=remaining_steps)  # Adjust n_steps as needed
            # Log rewards or other info if needed
            print(f"Steps: {steps}, Total Reward: {total_reward}")
            # Update Policy
            self._update()
            steps += n_steps
            remaining_steps -= n_steps
            self.rollout_buffer.reset()