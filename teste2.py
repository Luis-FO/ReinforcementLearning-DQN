import torch
import numpy as np
import gymnasium as gym
from core.dqn_agent import DQNAgent
from core.dqn_model import DQN

# Create Continuous Action Space Environment
# env = gym.make('MountainCarContinuous-v0')
# state, info = env.reset()
# print("Estado inicial:", state)

# state_tensor = torch.FloatTensor(state).unsqueeze(0)
# # print("Estado como Tensor:", state_tensor)
# arr = np.array(([1], [1], [6]))
# # print(arr)
# arr_tensor = torch.FloatTensor(arr)
# print(arr_tensor)
# a = torch.randn_like(arr_tensor)
# print(a)
# # print(arr_tensor.unsqueeze(0))
print(torch.cuda.is_available())
# print(torch.cuda.device_count())