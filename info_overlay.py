import gymnasium as gym
import numpy as np
import cv2

class InfoOverlay(gym.Wrapper):
    """Wrapper com painel de informações customizado"""
    
    def __init__(self, env, panel_height=100):
        super().__init__(env)
        self.episode = 0
        self.step_count = 0
        self.episode_reward = 0
        self.last_reward = 0
        self.panel_height = panel_height
        
    def reset(self, **kwargs):
        self.episode += 1
        self.step_count = 0
        self.episode_reward = 0
        self.last_reward = 0
        return super().reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.step_count += 1
        self.episode_reward += reward
        self.last_reward = reward
        return obs, reward, terminated, truncated, info
    
    def render(self):
        frame = super().render()
        
        if frame is None:
            return frame
        
        # Criar frame expandido com painel
        h, w = frame.shape[:2]
        frame_with_panel = np.zeros((h + self.panel_height, w, 3), dtype=np.uint8)
        frame_with_panel[self.panel_height:] = frame
        
        # Adicionar informações no painel
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        
        cv2.putText(frame_with_panel, f'Episode: {self.episode}', 
                    (10, 30), font, 0.7, color, 2)
        cv2.putText(frame_with_panel, f'Step: {self.step_count}', 
                    (10, 65), font, 0.7, color, 2)
        cv2.putText(frame_with_panel, f'Last Reward: {self.last_reward:.2f}', 
                    (250, 30), font, 0.7, color, 2)
        cv2.putText(frame_with_panel, f'Total Reward: {self.episode_reward:.2f}', 
                    (250, 65), font, 0.7, color, 2)
        
        return frame_with_panel