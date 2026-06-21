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
    
    # TODO: values ans log_probs should raise an error if not provided on PPO and be optional for other algorithms?

    def _store_transition(self, obs, action, reward, next_obs, done, values=None, log_probs=None):
        self.rollout_buffer.add(
            obs,
            action,
            reward,
            next_obs,
            done,
            values,
            log_probs,
        )

    # TODO: Use just fixed steps, and reset should be called when done is True
    def collect_rollouts(self, n_steps):
        obs, _ = self.env.reset()
        done = False
        steps = 0
        total_reward = 0
        while steps < n_steps:
            if done:
                obs, _ = self.env.reset()
                done = False
            action, log_prob, value = self.select_action(obs, training=True)
            next_obs, reward, terminate, truncate, _ = self.env.step(action)

            total_reward += reward
            done = terminate or truncate

            # Armazenar a transição na memória usando apenas o terminal real
            self._store_transition(obs, action, reward, next_obs, terminate, value, log_prob)

            obs = next_obs
            steps += 1

        last_value = 0 if terminate else self.select_action(obs, training=True)[2]

        self.rollout_buffer._compute_returns_and_advantage(
            rewards=np.array(self.rollout_buffer.rewards),
            values=np.array(self.rollout_buffer.values),
            dones=np.array(self.rollout_buffer.is_terminals),
            last_value=last_value
        )
        return steps, total_reward
    
    def train(self, total_steps, rollout_size):
        assert self.rollout_buffer is not None, "rollout_buffer must be initialized before training"
        assert total_steps > 0, "total_steps must be a positive integer"
        assert rollout_size > 0, "rollout_size must be a positive integer"
        steps = 0
        while steps < total_steps:
            # Collect Rollouts
            n_steps, total_reward = self.collect_rollouts(n_steps=rollout_size)  # Adjust n_steps as needed
            # Log rewards or other info if needed
            print(f"Steps: {steps}, Total Reward: {total_reward}")
            # Update Policy
            self._update()
            steps += rollout_size
            self.rollout_buffer.reset()