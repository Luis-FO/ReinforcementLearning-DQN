import torch
import random
from dqn_model import DQN

class Agent():
    def __init__(self, policy_net, n_actions, device= 'cuda', ):

        self.policy_net = policy_net
        self.device = device
        self.n_actions = n_actions

    def select_action(self, obs, eps_threshold = 0):
        
        val = random.random()   
        
        if val>eps_threshold:
            with torch.no_grad():
                action = self.policy_net(obs).max(1).indices.view(1, 1)
        else:
            action = torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
        return action


if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make('Acrobot-v1')
    obs, info = env.reset()

    num_obs = len(obs)
    num_actions = env.action_space.n
    device = 'cpu'

    obs = torch.tensor(obs).unsqueeze(0)
    policy_net = DQN(num_obs, num_actions).to(device=device)
    agent  =  Agent(policy_net=policy_net, n_actions=num_actions, device=device)
    action = agent.select_action(obs)
    print(action)