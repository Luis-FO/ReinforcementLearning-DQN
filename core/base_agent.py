"""
Base Agent class providing common functionality for all RL agents.
This class defines the interface and shared methods for DQN, DDPG, TD3, and PPO agents.
"""

from abc import ABC, abstractmethod
import torch
import os


class BaseAgent(ABC):
    """
    Abstract base class for reinforcement learning agents.
    
    Subclasses should implement:
    - select_action(): Choose action based on current policy
    - update(): Update agent parameters based on experience
    """
    
    def __init__(self, device='cuda', gamma=0.99, tau=0.005, batch_size=128):
        """
        Initialize base agent.
        
        Args:
            device: torch device (e.g., 'cuda' or 'cpu')
            gamma: discount factor
            tau: soft update coefficient for target networks
            batch_size: batch size for training
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memory = None  # To be set by subclasses
        
    @abstractmethod
    def select_action(self, obs, training=True):
        """
        Select an action based on the current observation.
        
        Args:
            obs: Current observation/state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Action to take
        """
        pass
    
    @abstractmethod
    def update(self):
        """
        Update agent parameters based on experience.
        Should use self.memory to sample experiences.
        """
        pass
    
    def soft_update(self, target_net, source_net, tau=None):
        """
        Perform soft update of target network parameters.
        θ_target = τ * θ_source + (1 - τ) * θ_target
        
        Args:
            target_net: Target network to update
            source_net: Source network to copy from
            tau: Soft update coefficient (uses self.tau if None)
        """
        if tau is None:
            tau = self.tau
            
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )
    
    def hard_update(self, target_net, source_net):
        """
        Perform hard update (complete copy) of target network parameters.
        
        Args:
            target_net: Target network to update
            source_net: Source network to copy from
        """
        target_net.load_state_dict(source_net.state_dict())
    
    def save(self, filepath, networks_dict=None, optimizers_dict=None):
        """
        Save agent state to file.
        
        Args:
            filepath: Path to save checkpoint
            networks_dict: Dictionary of network names to network objects
            optimizers_dict: Dictionary of optimizer names to optimizer objects
        """
        checkpoint = {
            'device': str(self.device),
            'gamma': self.gamma,
            'tau': self.tau,
            'batch_size': self.batch_size,
        }
        
        if networks_dict:
            for name, network in networks_dict.items():
                checkpoint[f'{name}_state'] = network.state_dict()
        
        if optimizers_dict:
            for name, optimizer in optimizers_dict.items():
                checkpoint[f'{name}_state'] = optimizer.state_dict()
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)

        
    def load(self, filepath, networks_dict=None, optimizers_dict=None):
        """
        Load agent state from file.
        
        Args:
            filepath: Path to checkpoint file
            networks_dict: Dictionary of network names to network objects
            optimizers_dict: Dictionary of optimizer names to optimizer objects
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if networks_dict:
            for name, network in networks_dict.items():
                state_key = f'{name}_state'
                if state_key in checkpoint:
                    network.load_state_dict(checkpoint[state_key])
        
        if optimizers_dict:
            for name, optimizer in optimizers_dict.items():
                state_key = f'{name}_state'
                if state_key in checkpoint:
                    optimizer.load_state_dict(checkpoint[state_key])
