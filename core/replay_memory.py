from collections import deque
from random import sample
import numpy as np


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
    

if __name__ == "__main__":
    mem = ReplayMemory(100)
    mem.push(1,2,3,4)
    print(mem)
