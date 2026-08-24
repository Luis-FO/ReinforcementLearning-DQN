import gymnasium as gym
import cv2
import numpy as np

class InfoOverlay(gym.Wrapper):
    def __init__(self, env, panel_height=100, target_size=None, format_type='stories'):
        """
        Args:
            env: Gymnasium Environment
            panel_height: Height of the information panel in pixels
            target_size: Tuple (width, height) for resizing the output frame to a specific size. 
                         If None, no resizing is done.
            format_type: 'stories' (9:16), 'feed' (1:1), 'portrait' (4:5), or None
        """
        super().__init__(env)
        self.episode = 0
        self.step_count = 0
        self.episode_reward = 0
        self.panel_height = panel_height
        
        if target_size:
            self.target_size = target_size
        elif format_type == 'stories':
            self.target_size = (1080, 1920) 
        elif format_type == 'feed':
            self.target_size = (1080, 1080)  
        elif format_type == 'portrait':
            self.target_size = (1080, 1350)
        else:
            self.target_size = None
        
    def reset(self, **kwargs):
        self.episode += 1
        self.step_count = 0
        self.episode_reward = 0
        return super().reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.step_count += 1
        self.episode_reward += reward
        return obs, reward, terminated, truncated, info
    
    def render(self):
        frame = super().render()
        if frame is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Create frame with information panel
        frame_with_panel = np.zeros((h + self.panel_height, w, 3), dtype=np.uint8)
        frame_with_panel[self.panel_height:] = frame
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        
        cv2.putText(frame_with_panel, f'Episode: {self.episode}', 
                    (50, 35), font, 0.8, color, 2)

        # Calcular posição para o reward à direita
        reward_text = f'Reward: {self.episode_reward:.1f}'
        text_size = cv2.getTextSize(reward_text, font, 0.8, 2)[0]
        text_x = w - text_size[0] - 50
        cv2.putText(frame_with_panel, reward_text, 
                    (text_x, 35), font, 0.8, color, 2)
        
        # Redimensionar para formato Instagram se especificado
        if self.target_size:
            frame_with_panel = self._resize_for_instagram(frame_with_panel)
        
        return frame_with_panel
    
    def _resize_for_instagram(self, frame):
        """Resize the frame to the Instagram format while maintaining aspect ratio"""
        h, w = frame.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate aspect ratios
        frame_aspect = w / h
        target_aspect = target_w / target_h
        
        if frame_aspect > target_aspect:
            # Frame is wider - add bars on top and bottom
            new_w = target_w
            new_h = int(target_w / frame_aspect)
        else:
            # Frame is taller - add bars on the sides
            new_h = target_h
            new_w = int(target_h * frame_aspect)
        
        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create canvas at target size with black background
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Center resized frame on canvas
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas