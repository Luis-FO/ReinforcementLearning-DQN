import numpy as np
import random

class SumTree:
    """
    Estrutura de dados auxiliar para amostragem eficiente baseada em prioridade.
    Complexidade: O(log N) para amostragem e atualização.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        # Árvore binária para soma das prioridades
        # Tamanho: 2 * capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        # Dados reais (transitions)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        # Índice na folha da árvore
        tree_index = self.data_pointer + self.capacity - 1
        
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)

        self.data_pointer += 1
        # Se ultrapassar a capacidade, volta ao início (sobrescreve o mais antigo)
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # Propaga a mudança para os nós pais até a raiz
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        """
        Encontra o índice da folha correspondente ao valor v acumulado.
        """
        parent_index = 0
        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # Se chegou no final da árvore (folha)
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            if v <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                v -= self.tree[left_child_index]
                parent_index = right_child_index
        
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # O valor na raiz é a soma total


class PrioritizedReplayMemory:
    """
    Buffer de Experiência Priorizada (PER).
    
    alpha: Quanto de priorização usar (0 = uniforme, 1 = prioridade total).
    beta: Correção de viés (Importance Sampling). Começa baixo e sobe até 1.
    """
    def __init__(self, size, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(size)
        self.max_priority = 1.0  # Prioridade inicial para novas experiências
        self.alpha = alpha
        
        # Hiperparâmetros para beta annealing
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        # Novas experiências entram com prioridade máxima para garantir que sejam vistas
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size):
        batch_items = []
        idxs = []
        segment = self.tree.total_priority / batch_items
        priorities = []

        # Calcula o beta atual (annealing linear até 1.0)
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            # Pega o índice na árvore, a prioridade e os dados
            idx, p, data = self.tree.get_leaf(s)
            
            priorities.append(p)
            batch_items.append(data)
            idxs.append(idx)

        # Processamento dos Pesos de Importance Sampling (IS Weights)
        sampling_probabilities = np.array(priorities) / self.tree.total_priority
        # w_i = (N * P(i)) ^ -beta
        is_weights = np.power(self.tree.capacity * sampling_probabilities, -beta)
        is_weights /= is_weights.max()  # Normaliza para estabilidade (max 1.0)

        # Desempacota os dados
        states, actions, rewards, next_states, dones = zip(*batch_items)
        
        return (
            np.array(states), 
            np.array(actions), 
            np.array(rewards), 
            np.array(next_states), 
            np.array(dones), 
            np.array(idxs),       # Importante: Retorna índices para atualizar depois
            np.array(is_weights, dtype=np.float32) # Pesos para multiplicar na Loss
        )

    def update_priorities(self, idxs, td_errors):
        """
        Atualiza as prioridades na árvore com base nos novos erros TD calculados.
        """
        # Adiciona um epsilon pequeno para garantir probabilidade não-zero
        epsilon = 1e-5
        td_errors = np.abs(td_errors) + epsilon
        clipped_errors = np.minimum(td_errors, 1.0) # Clipping para estabilidade
        
        for idx, error in zip(idxs, clipped_errors):
            p = np.power(error, self.alpha)
            self.tree.update(idx, p)
        
        # Atualiza a max_priority para as próximas inserções
        self.max_priority = max(self.max_priority, np.max(np.power(clipped_errors, self.alpha)))

    def __len__(self):
        # Apenas uma aproximação baseada no ponteiro se não estiver cheio
        if self.tree.data_pointer == 0 and self.tree.tree[0] == 0:
            return 0
        return self.tree.capacity # Assumindo cheio, ou controle separado de contagem

# --- Exemplo de Uso ---
if __name__ == "__main__":
    mem = PrioritizedReplayMemory(size=100)
    
    # 1. Adicionando dados
    mem.push(np.array([1]), 0, 1, np.array([2]), False)
    mem.push(np.array([2]), 1, -1, np.array([3]), True)
    
    # 2. Amostrando (Note que retorna indices e pesos agora)
    states, actions, rewards, next_states, dones, idxs, weights = mem.sample(1)
    
    print(f"Estado amostrado: {states}")
    print(f"Indices na árvore: {idxs}")
    print(f"Pesos IS: {weights}")
    
    # 3. Simulando um treino onde o erro foi ALTO (surpresa!)
    # O agente calculou a loss e descobriu que errou feio nessa transição
    fake_td_errors = np.array([10.0]) 
    
    # 4. Atualizando a prioridade
    mem.update_priorities(idxs, fake_td_errors)
    print("Prioridade atualizada! Essa transição será escolhida com mais frequência agora.")