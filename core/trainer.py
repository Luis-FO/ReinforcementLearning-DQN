import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from info_overlay import InfoOverlay

from replay_memory import ReplayMemory

from logtrigger import segmented_limit_trigger
from agent import Agent
from dqn_model import DQN


class DQNTrainer():

    def __init__(self, env_name, env, agent, policy_net, target_net, optimizer, criterion, memory_capacity=10000, device= 'cuda',  batch_size=128, gamma=0.99, tau = 0.005, eps_start = 0.9, eps_end=0.05, eps_decay=0.95):

        self.env_name = env_name
        self.env = env

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

        self.EPSILON = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        
        self.steps = 0 
    
    def decay_epsilon(self):
        self.EPSILON = max(self.EPS_END, self.EPSILON * self.EPS_DECAY)
        return self.EPSILON
    
    def set_env(self, env):
        """Define um novo ambiente para o treinador.
        Fecha o ambiente atual antes de definir o novo.
        Args:
            env: Novo ambiente Gymnasium a ser usado.
        """
        self.env.close()
        self.env_name = env.spec.id
        self.env = env


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
        episode_rewards = []
        # Cria o ambiente caso não tenha sido criado ainda (cria sem renderização visual)
        if not hasattr(self, 'env'):
            self.env = gym.make(self.env_name, render_mode="rgb_array")

        # Training loop
        for episode in range(num_episodes):
            if show_train_after >= 0 and episode >= show_train_after:
                self.env.close()
                self.env = gym.make(self.env_name, render_mode="human")
                show_train_after = -1 
            
            # Resetando o ambiente no início de cada episódio
            obs, info = self.env.reset()
            # Convertendo a observação para tensor

            obs = torch.tensor(obs, device=self.device).unsqueeze(0)

            done = False
            total_reward = 0
            # Loop até o episódio terminar
            while not done:

                # Epsilon decai ao longo do tempo. 
                epsilon = self.EPSILON
                # Selecionar ação com política epsilon-greedy.
                action = self.agent.select_action(obs, epsilon)
                self.steps+=1

                # Executar ação no ambiente
                next_obs, reward, terminate, truncate, info = self.env.step(action.item())
                total_reward += reward
                # Convertendo o reward para tensor
                reward = torch.tensor([reward], device=self.device)
                done = terminate or truncate

                # Converter a próxima observação para tensor caso o episódio não tenha terminado
                if terminate:
                    next_obs = None
                else:
                    next_obs = torch.tensor(next_obs, device=self.device).unsqueeze(0)
                
                # Armazenar a transição na memória de replay
                self.memory.push(obs, action, next_obs, reward)

                obs = next_obs
                # Chamar o método de otimização do modelo
                self.optimize_model()

                # Atualizar a rede alvo periodicamente
                self.update_target_net()
            
            self.decay_epsilon()
            episode_rewards.append(total_reward)
            print(f"Episódio {episode}: Recompensa Total = {total_reward}, Epsilon = {epsilon:.4f}")
        self.env.close()
        
        if len(episode_rewards) >= 100:
            final_metric = sum(episode_rewards[-100:]) / 100
        else:
            # Caso o treino seja muito curto ou interrompido
            final_metric = sum(episode_rewards) / len(episode_rewards) if len(episode_rewards) > 0 else -1000

        print(f"Treino concluído. Métrica final (média de 100): {final_metric}")
        return final_metric

    def test_agent(self, num_episodes, render=False):
        """Função para testar o agente treinado em um número especificado de episódios.
        O agente executa ações no ambiente sem exploração (epsilon = 0) e a recompensa total por episódio é registrada.
        No final, a recompensa média sobre todos os episódios de teste é calculada e retornada.
        """
        test_env = gym.make(self.env_name, render_mode="human")
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs, info = test_env.reset()
            obs = torch.tensor(obs, device=self.device).unsqueeze(0)

            done = False
            total_reward = 0

            while not done:
                # Selecionar ação sem exploração (epsilon = 0)
                action = self.agent.select_action(obs, eps_threshold=0.0)

                # Executar ação no ambiente
                next_obs, reward, terminate, truncate, info = test_env.step(action.item())
                total_reward += reward
                done = terminate or truncate

                # Converter a próxima observação para tensor caso o episódio não tenha terminado
                if terminate:
                    next_obs = None
                else:
                    next_obs = torch.tensor(next_obs, device=self.device).unsqueeze(0)

                obs = next_obs

            episode_rewards.append(total_reward)
            print(f"Episódio de Teste {episode}: Recompensa Total = {total_reward}")

        test_env.close()
        average_reward = sum(episode_rewards) / num_episodes
        print(f"Média de Recompensa nos Testes: {average_reward}")
        return average_reward
        
    def optimize_model(self):
        """Função para otimizar o modelo de rede neural.
        Usa amostras da memória de replay para atualizar os pesos da rede.
        Os pesos são atualizados minimizando a diferença entre os valores Q previstos e os valores Q esperados.
        Os Q previstos são obtidos da rede de política (policy_net) e os Q esperados são calculados usando a rede alvo (target_net).
        """
        if len(self.memory) < self.batch_size :
            return
        
        # Captura uma amostra de transições da memória de replay
        state_batch, action_batch, reward_batch, next_states = self.memory.sample(self.batch_size)
        """
        1. *samples desempacota a lista de transições em colunas separadas.
        2. zip(*samples) agrupa os elementos correspondentes de cada transição juntos:
            Antes: [(s1, a1, s1', r1), (s2, a2, s2', r2), ...]
            Depois: [(s1, s2, ...), (a1, a2, ...), (s1', s2', ...), (r1, r2, ...)]
        3. Transition(*zip(*samples)) cria uma nova namedtuple Transition onde cada campo contém uma tupla de todos os valores correspondentes:
            Transition(state=(s1, s2, ...), action=(a1, a2, ...), next_state=(s1', s2', ...), reward=(r1, r2, ...))
        """

        # Filtrar os próximos estados que não são terminais. Porque em estados terminais não há próximo estado.
        non_final_next_states = torch.cat([s for s in next_states \
                                                if s is not None])
        
        # Máscara booleana para identificar quais próximos estados não são terminais
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, \
                                            next_states)), device=self.device, dtype = torch.bool)

        # Calcula os valores Q previstos para o estado atual e ação tomada usando a rede de política.
        # É como se estivéssemos consultando a rede para saber "qual é o valor esperado se eu fizer essa ação nesse estado?"
        predicted_values = self.policy_net(state_batch).gather(dim = 1, index = action_batch)

        next_state_values = torch.zeros(self.batch_size , device = self.device)

        with torch.no_grad():
            # Calcula os valores Q esperados para os próximos estados não terminais usando a rede alvo.
            # Aqui, estamos perguntando "qual é o melhor valor que eu posso esperar no próximo estado?"
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(dim = 1).values
        
        # Calcula os valores Q esperados usando a fórmula do Bellman.  
        expected_values = next_state_values * self.gamma + reward_batch

        # Calcula a perda entre os valores previstos e esperados
        loss = self.criterion(predicted_values, expected_values.unsqueeze(1))
        
        # Zera os gradientes acumulados
        self.optimizer.zero_grad()
        # Propaga o erro para calcular os gradientes
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        # Atualiza os pesos da rede neural
        self.optimizer.step()

    def save_policy_net(self, path="./policy_net.pt"):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Modelo salvo em {path}")
    
    def load_pretmodel(self, path: str):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict()) 
        self.target_net.eval()
        

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