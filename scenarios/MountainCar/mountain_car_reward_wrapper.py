import gymnasium as gym

class MountainCarRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, reward):
        # O 'reward' aqui é o original (-1)
        posicao, velocidade = self.env.unwrapped.state
        
        # Lógica de recompensa baseada na altura
        nova_recompensa = reward + abs(posicao - (-0.5))
        
        # Penaliza se ficar parado (velocidade muito baixa) para forçar movimento
        if abs(velocidade) < 0.001:
            nova_recompensa -= 0.1
            
        return nova_recompensa

if __name__ == "__main__":

    env = gym.make('MountainCar-v0')
    env = MountainCarRewardWrapper(env) 
