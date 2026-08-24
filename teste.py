import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
import copy
from collections import deque, namedtuple
from abc import ABC, abstractmethod

# --- 1. Definições de Dados ---
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- 2. Estratégias de Exploração ---
class ExplorationStrategy(ABC):
    @abstractmethod
    def get_value(self, action_dim=None):
        pass

    @abstractmethod
    def decay(self):
        pass

    @abstractmethod
    def reset(self):
        pass

class GaussianNoise(ExplorationStrategy):
    """
    TD3 geralmente usa ruído Gaussiano fixo (sem decay complexo), 
    mas mantive a estrutura compatível caso queira usar decay.
    """
    def __init__(self, std=0.1):
        self.std = std

    def get_value(self, action_dim=1):
        return np.random.normal(0, self.std, size=action_dim)

    def decay(self):
        pass 

    def reset(self):
        pass

# --- 3. Redes Neurais ---
class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

class Actor(BaseNetwork):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim1=400, hidden_dim2=300):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.net(x)

class Critic(BaseNetwork):
    def __init__(self, state_dim, action_dim, hidden_dim1=400, hidden_dim2=300):
        super(Critic, self).__init__()
        # Arquitetura padrão TD3 costuma concatenar estado e ação na entrada
        # Ou processar estado primeiro. Aqui mantemos a concatenação simples.
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.net(x)

# --- 4. O Agente TD3 ---
class TD3Agent:
    def __init__(
        self, 
        actor: Actor, 
        critic_1: Critic, 
        critic_2: Critic,
        actor_optim: optim.Optimizer, 
        critic_1_optim: optim.Optimizer,
        critic_2_optim: optim.Optimizer,
        replay_memory: ReplayMemory,
        exploration_strategy: ExplorationStrategy,
        criterion=nn.MSELoss(), 
        device='cpu',
        gamma=0.99, 
        tau=0.005,
        batch_size=256,
        policy_noise=0.2,   # Desvio padrão do ruído de suavização do alvo
        noise_clip=0.5,     # Limite do ruído de suavização
        policy_freq=2       # Frequência de atualização do ator (Delayed Update)
    ):
        self.device = device
        
        # Atores
        self.actor = actor.to(device)
        self.actor_target = copy.deepcopy(actor).to(device)
        self.actor_optimizer = actor_optim

        # Críticos Gêmeos (Twin Critics)
        self.critic_1 = critic_1.to(device)
        self.critic_1_target = copy.deepcopy(critic_1).to(device)
        self.critic_1_optimizer = critic_1_optim

        self.critic_2 = critic_2.to(device)
        self.critic_2_target = copy.deepcopy(critic_2).to(device)
        self.critic_2_optimizer = critic_2_optim

        self.memory = replay_memory
        self.exploration = exploration_strategy
        self.criterion = criterion
        
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Hiperparâmetros específicos do TD3
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        self.total_it = 0 # Contador de atualizações

    def select_action(self, state, training=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()
        
        if training:
            exploration_value = self.exploration.get_value(action_dim=len(action))
            action = action + exploration_value
            
        return np.clip(action, -self.actor.max_action, self.actor.max_action)

    def update(self):
        self.total_it += 1

        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action = torch.FloatTensor(np.array(batch.action)).to(self.device)
        next_state = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        reward = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.array(batch.done)).unsqueeze(1).to(self.device)

        with torch.no_grad():
            # --- 1. Target Policy Smoothing ---
            # Seleciona ação alvo base
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (self.actor_target(next_state) + noise).clamp(-self.actor.max_action, self.actor.max_action)

            # --- 2. Twin Critics (Clipped Double Q-Learning) ---
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            
            # Pega o mínimo entre os dois críticos para evitar superestimativa
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # --- Atualização dos Críticos ---
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        critic_1_loss = self.criterion(current_Q1, target_Q)
        critic_2_loss = self.criterion(current_Q2, target_Q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # --- 3. Delayed Policy Updates ---
        # Só atualiza o Ator e as redes alvo a cada 'policy_freq' passos
        if self.total_it % self.policy_freq == 0:
            
            # A perda do ator é calculada baseada apenas no Q1
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft updates
            self._soft_update(self.critic_1, self.critic_1_target)
            self._soft_update(self.critic_2, self.critic_2_target)
            self._soft_update(self.actor, self.actor_target)

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

# --- 5. O Treinador ---
class Trainer:
    def __init__(self, env, agent, warm_up_steps=1000):
        self.env = env
        self.agent = agent
        self.warm_up_steps = warm_up_steps

    def train(self, num_episodes):
        history = []
        self.agent.exploration.reset()
        total_steps = 0 
        
        for i in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done, trunc = False, False
            
            while not (done or trunc):
                if total_steps < self.warm_up_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(state, training=True)
                
                next_state, reward, done, trunc, _ = self.env.step(action)
                
                # Armazenar float(done) é crucial, mas cuidado com truncamento (limite de tempo)
                # Se for truncation, done_bool deve ser False para o bootstrap não ser zerado incorretamente
                done_bool = float(done) if not trunc else 0.0
                
                self.agent.memory.push(state, action, next_state, reward, done_bool)
                
                if total_steps >= self.warm_up_steps:
                    self.agent.update()
                
                state = next_state
                episode_reward += reward
                total_steps += 1
            
            self.agent.exploration.decay()
            history.append(episode_reward)
            
            status = "Warm Up" if total_steps < self.warm_up_steps else "Treinando"
            # Opcional: imprimir steps totais para monitorar warm-up
            print(f"Episódio {i}: Recompensa {episode_reward:.2f} | Status: {status} | Total Steps: {total_steps}")
            
        return history

# --- 6. Composition Root ---
if __name__ == "__main__":
    # Nota: Certifique-se de ter instalado: pip install gymnasium[box2d]
    env_name = "BipedalWalker-v3" 
    env = gym.make(env_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Instanciação das Redes (1 Ator, 2 Críticos)
    # Arquitetura 400-300 é comum para BipedalWalker no paper original do TD3
    actor = Actor(state_dim, action_dim, max_action, hidden_dim1=400, hidden_dim2=300).to(device)
    critic_1 = Critic(state_dim, action_dim, hidden_dim1=400, hidden_dim2=300).to(device)
    critic_2 = Critic(state_dim, action_dim, hidden_dim1=400, hidden_dim2=300).to(device)
    
    actor_optim = optim.Adam(actor.parameters(), lr=3e-4)
    critic_1_optim = optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2_optim = optim.Adam(critic_2.parameters(), lr=3e-4)
    
    # Aumentando memória para 1M (padrão em papers)
    memory = ReplayMemory(1000000)
    
    # TD3 usa exploração Gaussiana simples
    noise_strategy = GaussianNoise(std=0.1)
    
    agent = TD3Agent(
        actor=actor,
        critic_1=critic_1,
        critic_2=critic_2,
        actor_optim=actor_optim,
        critic_1_optim=critic_1_optim,
        critic_2_optim=critic_2_optim,
        replay_memory=memory,
        exploration_strategy=noise_strategy,
        device=device,
        batch_size=256,
        policy_noise=0.2, 
        noise_clip=0.5,   
        policy_freq=2     
    )

    # BipedalWalker precisa de mais warm up que Pendulum
    trainer = Trainer(env, agent, warm_up_steps=10000)
    
    print(f"Iniciando treino TD3 no {env_name}...")
    # BipedalWalker geralmente precisa de >1000 episódios para resolver bem
    # Coloquei 500 para teste, ajuste conforme necessário.
    trainer.train(num_episodes=500)
    env.close()