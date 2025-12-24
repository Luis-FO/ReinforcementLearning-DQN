import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import cv2
import numpy as np

class InfoOverlay(gym.Wrapper):
    def __init__(self, env, panel_height=100, target_size=None, format_type='stories'):
        """
        Args:
            env: Ambiente do Gymnasium
            panel_height: Altura do painel de informações
            target_size: Tupla (width, height) customizada, ou None para usar format_type
            format_type: 'stories' (9:16), 'feed' (1:1), 'portrait' (4:5), ou None
        """
        super().__init__(env)
        self.episode = 0
        self.step_count = 0
        self.episode_reward = 0
        self.panel_height = panel_height
        
        # Definir tamanho alvo baseado no formato
        if target_size:
            self.target_size = target_size
        elif format_type == 'stories':
            self.target_size = (1080, 1920)  # 9:16 - Stories/Reels
        elif format_type == 'feed':
            self.target_size = (1080, 1080)  # 1:1 - Feed quadrado
        elif format_type == 'portrait':
            self.target_size = (1080, 1350)  # 4:5 - Feed retrato
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
        
        # Criar frame com painel de informações
        frame_with_panel = np.zeros((h + self.panel_height, w, 3), dtype=np.uint8)
        frame_with_panel[self.panel_height:] = frame
        
        # Adicionar texto
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
        """Redimensiona o frame para o formato Instagram mantendo aspect ratio"""
        h, w = frame.shape[:2]
        target_w, target_h = self.target_size
        
        # Calcular aspect ratios
        frame_aspect = w / h
        target_aspect = target_w / target_h
        
        if frame_aspect > target_aspect:
            # Frame é mais largo - adicionar barras em cima/embaixo
            new_w = target_w
            new_h = int(target_w / frame_aspect)
        else:
            # Frame é mais alto - adicionar barras nas laterais
            new_h = target_h
            new_w = int(target_h * frame_aspect)
        
        # Redimensionar frame
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Criar canvas no tamanho alvo com fundo preto
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Centralizar frame redimensionado no canvas
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas

if __name__ == "__main__":

    # Uso para Instagram Stories/Reels (vertical):
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = InfoOverlay(env, format_type='stories')  # 9:16
    env = RecordVideo(env, 'videos/stories/', name_prefix='cartpole-stories')

    # Uso para Instagram Feed (quadrado):
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = InfoOverlay(env, format_type='feed')  # 1:1
    env = RecordVideo(env, 'videos/feed/', name_prefix='cartpole-feed')

    # Uso para Instagram Feed (retrato):
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = InfoOverlay(env, format_type='portrait')  # 4:5
    env = RecordVideo(env, 'videos/portrait/', name_prefix='cartpole-portrait')

    # Ou tamanho customizado:
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = InfoOverlay(env, target_size=(720, 1280))  # Customizado
    env = RecordVideo(env, 'videos/custom/')