from collections import namedtuple, deque
from random import sample
import torch
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    def __init__(self, size):
        self.memory = deque([], maxlen=size)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, size):
        samples = sample(self.memory, size)
        batch = Transition(*zip(*samples))
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = batch.next_state
        return states, actions, rewards, next_states
    
    def __len__(self):
        return len(self.memory)
    
    def __str__(self):
        return f"{self.memory}"
    

if __name__ == "__main__":
    mem = ReplayMemory(100)
    mem.push(1,2,3,4)
    print(mem)
