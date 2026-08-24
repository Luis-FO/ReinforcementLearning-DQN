import numpy as np
from collections import deque
from random import sample
import numpy as np

class RolloutBuffer:
    def __init__(self, size, gamma=0.99, gae_lambda=0.95):
        self.size = size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()


    def add(self, state, action, reward, next_state, done, values=None, log_probs=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.logprobs.append(log_probs)
        self.is_terminals.append(done)
        self.values.append(values)

    # TODO: attributes should be numpy arrays instead of lists
    def reset(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
        self.returns = []
        self.advantages = []

    def sample(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.next_states),
            np.array(self.is_terminals),
            np.array(self.logprobs)
        )
    
    def _compute_returns_and_advantage(self, rewards, values, dones, last_value):
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_value = last_value
        for step in reversed(range(n)):
            delta = rewards[step] + self.gamma * (1 - dones[step]) * next_value - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages[step] = gae
            next_value = values[step]
        returns = advantages + values
        self.returns = returns
        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return self.returns, self.advantages
    
    # def _compute_returns_and_advantage(self, rewards, values, dones, last_value):
    #     returns = []
    #     advantages = []
    #     gae = 0
    #     next_value = last_value  # V(s_T) — bootstrap do fim do rollout

    #     for step in reversed(range(len(rewards))):
    #         # next_value aqui é sempre V(s_{t+1}), correto
    #         delta = rewards[step] + self.gamma * (1 - dones[step]) * next_value - values[step]
    #         gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
    #         advantages.insert(0, gae)
    #         returns.insert(0, gae + values[step])
    #         next_value = values[step]  # para o próximo step (t-1), o "próximo" é o atual (t)

    #     self.returns = np.array(returns)
    #     self.advantages = np.array(advantages)

    #     # Normalização das vantagens — essencial para estabilidade do PPO
    #     self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    #     return self.returns, self.advantages



class ReplayMemory():
    def __init__(self, size):
        self.memory = deque([], maxlen=size)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, size):
        samples = sample(self.memory, size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.memory)
    
    def __str__(self):
        return f"{self.memory}"