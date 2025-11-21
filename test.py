
import numpy as np
import torch

if __name__ == "__main__":
    # mem = ReplayMemory(100)
    # mem.push(1,2,3,4)
    # mem.push(1,2,3,4)
    # mem.push(1,2,3,4)
    # mem.push(1,2,3,4)
    # samples = mem.sample(4)

    # batch = Transition(*zip(*samples))
    # print(type(batch.state))

    # state_batch = np.array(batch.state)
    # print(type(state_batch))
    # print(state_batch)
    obs = 1.0
    obs = torch.tensor(obs, device="cpu").unsqueeze(0)
    print(obs)