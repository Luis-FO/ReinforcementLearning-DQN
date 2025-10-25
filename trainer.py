import torch
import math
import gymnasium as gym
from replay_memory import ReplayMemory, Transition
from agent import Agent

from dqn_model import DQN

class DQNTrainer():

    def __init__(self, env_name, agent, policy_net, target_net, optimizer, criterion, memory_capacity=10000, device= 'cuda',  batch_size=128, gamma=0.99, tau = 0.005, eps_start = 0.9, eps_end=0.05, eps_decay=1000):

        self.env_name = env_name
        self.env = gym.make(env_name)

        self.agent = agent
        self.policy_net = policy_net
        self.target_net = target_net

        self.optimizer = optimizer
        self.criterion = criterion
        self.memory = ReplayMemory(memory_capacity)

        self.device = device
        self.gamma = gamma # Fator de desconto
        self.tau = tau  # Taxa de atualização da target_net
        self.batch_size = batch_size

        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        
        self.steps = 0 
    
    def get_epsilon(self):
        return self.EPS_END + (self.EPS_START - self.EPS_END) * \
               math.exp(-1. * self.steps / self.EPS_DECAY)

    def update_target_net(self):
        """
        soft update.
        θ_target = τ * θ_policy + (1 - τ) * θ_target
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + \
                                         target_net_state_dict[key] * (1 - self.tau)
        
        self.target_net.load_state_dict(target_net_state_dict)


    def train(self, num_episodes, show_train_after = -1):
        for i in range(num_episodes):
            if show_train_after >= 0 and i >= show_train_after:
                self.env.close()
                self.env = gym.make(self.env_name, render_mode="human")
                show_train_after = -1 
            
            obs, info = self.env.reset()
            obs = torch.tensor(obs, device=self.device).unsqueeze(0)
            
            done = False
            total_reward = 0
            while not done:

                epsilon = self.get_epsilon()
                action = self.agent.select_action(obs, epsilon)
                self.steps+=1

                next_obs, reward, terminate, truncate, info = self.env.step(action.item())
                total_reward += reward

                reward = torch.tensor([reward], device=self.device)
                done = terminate or truncate

                if terminate:
                    next_obs = None
                else:
                    next_obs = torch.tensor(next_obs, device=self.device).unsqueeze(0)

                self.memory.push(obs, action, next_obs, reward)

                obs = next_obs
                self.optimize_model()

                self.update_target_net()
            print(f"Episódio {i}: Recompensa Total = {total_reward}, Epsilon = {epsilon:.4f}")
        self.env.close()

    def optimize_model(self):
        if len(self.memory) < self.batch_size :
            return
        
        samples = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*samples))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state \
                                                if s is not None])
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, \
                                            batch.next_state)), device=self.device, dtype = torch.bool)

        predicted_values = self.policy_net(state_batch).gather(dim = 1, index = action_batch)

        next_state_values = torch.zeros(self.batch_size , device = self.device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(dim = 1).values
        
        expected_values = next_state_values * self.gamma + reward_batch

        loss = self.criterion(predicted_values, expected_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_policy_net(self, path="./policy_net.pt"):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Modelo salvo em {path}")
    
    def load_pretmodel(self, path: str):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict()) 
        

if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make('Acrobot-v1')
    obs, info = env.reset()

    num_obs = len(obs)
    num_actions = env.action_space.n

    obs = torch.tensor(obs).unsqueeze(0)

    agent  =  Agent(input_size=num_obs, output_size=num_actions, env = env)
    action = agent.select_action(obs)
    print(action)