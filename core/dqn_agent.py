import torch
import random
import copy

from core.replay_memory import ReplayMemory
from core.exploration import ExplorationStrategy
from core.off_policy_algorithm import OffPolicyAlgorithm

class DQNAgent(OffPolicyAlgorithm):
    def __init__(self,
                 env, 
                 policy_net,
                 optimizer,
                 criterion,
                 memory_capacity: int,
                 exploration_strategy: ExplorationStrategy,
                 gamma=0.99,
                 warmup_steps=1000,
                 tau=0.005,
                 batch_size=128,
                 device= 'cuda'
                 ):

        super().__init__(env, device, warmup_steps=warmup_steps)
        self.device = device

        self.policy_net = policy_net
        self.target_net = copy.deepcopy(policy_net).to(device)  # Cria uma nova instância da mesma classe
        self.target_net.eval()

        self.optimizer = optimizer
        self.criterion = criterion
        self.memory = ReplayMemory(memory_capacity)
        self.exploration_strategy = exploration_strategy
        
        self.gamma = gamma # Fator de desconto
        self.tau = tau  # Taxa de atualização da target_net
        self.batch_size = batch_size
        self.steps_done = 0
        self.n_actions = policy_net.output_size

    def get_info(self):
        return {
            "exploration": self.exploration_strategy.get_value()
        }
    
    def on_step(self):
        """Method called at the end of each step in the environment. Can be used to decay exploration strategies or other stateful components."""
        self.steps_done += 1
        # if self.steps_done >= 20000:
        #     self.exploration_strategy.reset()
        #     self.steps_done = 0
    def on_train_start(self):
        """Method called at the beginning of the training loop. Can be used to reset exploration strategies or other stateful components."""
        self.exploration_strategy.reset()


    def select_action(self, obs, training=True):
        """Seleciona uma ação com base na política epsilon-greedy."""
        eps_threshold = self.exploration_strategy.get_value(action_dim=self.n_actions)
        if random.random() < eps_threshold and training:
            return random.randrange(self.n_actions)  
        
        with torch.no_grad():
            # Converte observation de array para tensor 2D 
            obs_tensor =  torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            return self.policy_net(obs_tensor).argmax().item()

    def _update(self):
        """Função para otimizar o modelo de rede neural.
        Usa amostras da memória de replay para atualizar os pesos da rede.
        Os pesos são atualizados minimizando a diferença entre os valores Q previstos e os valores Q esperados.
        Os Q previstos são obtidos da rede de política (policy_net) e os Q esperados são calculados usando a rede alvo (target_net).
        """
        if len(self.memory) < self.batch_size :
            return
        
        self.exploration_strategy.decay()  # Decai o valor de exploração a cada atualização
        # Captura uma amostra de transições da memória de replay
        state_batch, action_batch, reward_batch, next_states, dones = self.memory.sample(self.batch_size)


        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        # Calcula os valores Q previstos para o estado atual e ação tomada usando a rede de política.
        # É como se estivéssemos consultando a rede para saber "qual é o valor esperado se eu fizer essa ação nesse estado?"
        # predicted_values = self.policy_net(state_batch).gather(dim = 1, index = action_batch)
        predicted_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        with torch.no_grad():
            # Calcula os valores Q esperados para os próximos estados não terminais usando a rede alvo.
            # Aqui, estamos perguntando "qual é o melhor valor que eu posso esperar no próximo estado?"
            next_state_values = self.target_net(next_states).max(1)[0]
            # Calcula os valores Q esperados usando a fórmula do Bellman.  
            expected_values = reward_batch + (1-dones)* self.gamma * next_state_values

        # Calcula a perda entre os valores previstos e esperados
        loss = self.criterion(predicted_values, expected_values.unsqueeze(1))
        
        # Zera os gradientes acumulados
        self.optimizer.zero_grad()
        # Propaga o erro para calcular os gradientes
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        # Atualiza os pesos da rede neural
        self.optimizer.step()
        # Soft update da rede alvo
        self._soft_update()
        return loss.item()

    def _soft_update(self):
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

    def save(self, path="./policy_net.pt"):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Modelo salvo em {path}")

    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict()) 
        self.target_net.eval()

if __name__ == "__main__":
    from dqn_model import DQN
    import gymnasium as gym

    env = gym.make('Acrobot-v1')
    obs, info = env.reset()

    # num_obs = len(obs)
    # num_actions = env.action_space.n
    # device = 'cpu'

    # obs = torch.tensor(obs).unsqueeze(0)
    # policy_net = DQN(num_obs, num_actions).to(device=device)
    # agent  =  Agent(policy_net=policy_net, n_actions=num_actions, device=device)
    # action = agent.select_action(obs)
    # print(action)