import torch
import copy
import numpy as np

from core.networks import Actor, Critic
from core.replay_memory import ReplayMemory
from core.exploration import ExplorationStrategy

class TD3Agent:
    def __init__(self, actor: Actor, critic_1: Critic, critic_2: Critic,
                 actor_optim, critic_optim_1, critic_optim_2,
                 exploration_strategy: ExplorationStrategy, criterion,
                 memory_capacity=100000, device='cpu', gamma=0.99, tau=0.005, 
                 batch_size=256, policy_noise=0.2, noise_clip=0.5, policy_freq=2
                 ):
        
        self.device = device

        self.actor = actor
        self.actor_target = copy.deepcopy(actor).to(device)

        self.actor_optimizer = actor_optim

        self.critic_1 = critic_1.to(device)
        self.critic_1_target = copy.deepcopy(critic_1).to(device)
        self.critic_optimizer_1 = critic_optim_1

        self.critic_2 = critic_2.to(device)
        self.critic_2_target = copy.deepcopy(critic_2).to(device)
        self.critic_optimizer_2 = critic_optim_2

        self.criterion = criterion

        self.memory = ReplayMemory(memory_capacity)
        self.exploration_strategy = exploration_strategy

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        
    def select_action(self, state, training=True):
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Seleciona ação com base na política atual
        self.actor.eval() # Coloca o ator em modo de avaliação, para desativar dropout/batchnorm
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train() # Volta ao modo de treinamento para futuras chamadas

        # Adiciona ruído para exploração se estiver em modo de treinamento
        if training:
            noise = self.exploration_strategy.get_value(action_dim=action.shape[0])
            action = noise + action

        # Garante que a ação esteja dentro dos limites válidos
        return np.clip(action, -self.actor.max_action, self.actor.max_action)
    
    
    def update(self):
        
        self.total_it += 1

        if len(self.memory) < self.batch_size:
            return

        
        # Pega uma amostra de transições da memória de replay
        batch = self.memory.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_states, dones = batch

        # Converte para tensores
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        next_states_batch = torch.FloatTensor(next_states).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action_batch)*self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            target_action = (self.actor_target(next_states_batch) + noise).clamp(-self.actor.max_action, self.actor.max_action)

            target_Q1 = self.critic_1_target(next_states_batch, target_action)
            target_Q2 = self.critic_2_target(next_states_batch, target_action)
            target_Q = torch.min(target_Q1, target_Q2)

            target_Q = reward_batch + (1 -dones) * self.gamma * target_Q

        current_Q1 = self.critic_1(state_batch, action_batch)
        current_Q2 = self.critic_2(state_batch, action_batch)

        critic_1_loss = self.criterion(current_Q1, target_Q)
        critic_2_loss = self.criterion(current_Q2, target_Q)

        # Atualiza critic_1
        self.critic_optimizer_1.zero_grad()
        critic_1_loss.backward()
        self.critic_optimizer_1.step()


        # Atualiza critic_2
        self.critic_optimizer_2.zero_grad()
        critic_2_loss.backward()
        self.critic_optimizer_2.step()

        if self.total_it % self.policy_freq == 0:

            actor_loss = -self.critic_1(state_batch, self.actor(state_batch)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.critic_1, self.critic_1_target)
            self._soft_update(self.critic_2, self.critic_2_target)
            self._soft_update(self.actor, self.actor_target)

    def _soft_update(self, net, target_net):
        for current_params, target_params in zip(net.parameters(), target_net.parameters()):
            target_params.data.copy_(current_params.data*self.tau + (1-self.tau)*target_params.data)