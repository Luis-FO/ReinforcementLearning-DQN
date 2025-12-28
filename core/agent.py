import torch
import random

class Agent():
    def __init__(self, policy_net, n_actions, device= 'cuda'):

        self.policy_net = policy_net
        self.device = device
        self.n_actions = n_actions

    def select_action(self, obs, eps_threshold = 0):
        
        if random.random() < eps_threshold:
            return random.randrange(self.n_actions)
            # return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)   
        
        with torch.no_grad():
            # Converte observation de array para tensor 2D 
            obs_tensor =  torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            return self.policy_net(obs_tensor).argmax().item()
            # return self.policy_net(obs_tensor).max(1).indices.view(1, 1)


if __name__ == "__main__":
    from dqn_model import DQN
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