import gymnasium as gym

class MountainCarRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, reward):
        # O 'reward' aqui é o original (-1)
        pos, velocity = self.env.unwrapped.state
        
        # Lógica de recompensa baseada na altura
        new_reward = reward + abs(pos - (-0.5))
        
        # Penaliza se ficar parado (velocidade muito baixa) para forçar movimento
        if abs(velocity) < 0.001:
            new_reward -= 0.1
            
        return new_reward

if __name__ == "__main__":

    env = gym.make('MountainCar-v0')
    env = MountainCarRewardWrapper(env)
