import copy
import torch
import numpy as np

from core.networks import Actor, Critic
from core.replay_memory import ReplayMemory
from core.exploration import ExplorationStrategy
from core.off_policy_algorithm import OffPolicyAlgorithm

class DDPGAgent(OffPolicyAlgorithm):
    def __init__(self, env, actor, critic, actor_optimizer, critic_optimizer,
                 criterion, exploration_strategy: ExplorationStrategy,
                 memory_capacity=1000000, gamma=0.99, tau=0.005, 
                 batch_size=128, device='cuda'):

        if isinstance(device, str):
            if device == 'cuda' and not torch.cuda.is_available():
                device = 'cpu'
            device = torch.device(device)

        self.device = device
        super().__init__(env, self.device, warmup_steps=0)

        self.actor = actor.to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = actor_optimizer
        
        self.critic = critic.to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = critic_optimizer

        self.criterion = criterion

        self.memory = ReplayMemory(memory_capacity)
        self.exploration_strategy = exploration_strategy
        
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

    
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

    def get_info(self):
        return {
            "exploration": self.exploration_strategy.get_value()
        }

    def on_train_start(self):
        self.exploration_strategy.reset()

    def _update(self):
        if len(self.memory) < self.batch_size:
            return

        self.exploration_strategy.decay()  # Decai o valor de exploração a cada atualização
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
            target_action = self.actor_target(next_states_batch)
            target_Q = self.critic_target(next_states_batch, target_action)
            target_Q = reward_batch + (1 -dones) * self.gamma * target_Q

        current_Q = self.critic(state_batch, action_batch)
        critic_loss = self.criterion(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, net, target_net):
        for current_params, target_params in zip(net.parameters(), target_net.parameters()):
            target_params.data.copy_(current_params.data*self.tau + (1-self.tau)*target_params.data)
            
if __name__ == "__main__":
    
    agent = DDPGAgent(
        actor=Actor(3, 1, 2).to('cpu'),
        critic=Critic(3, 1, 2).to('cpu')
    )
    agent._soft_update()