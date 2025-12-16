"""Q-Network Module

Implements Q-value networks for reinforcement learning-based
resource allocation decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Deep Q-Network for action value estimation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        """Initialize Q-Network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Build network layers
        layers = []
        current_dim = state_dim
        
        for i in range(num_layers):
            next_dim = hidden_dim
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            current_dim = next_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for given state.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values tensor of shape (batch_size, action_dim)
        """
        return self.network(state)


class DuelingQNetwork(nn.Module):
    """Dueling Q-Network architecture for improved learning."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        """Initialize Dueling Q-Network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Feature extraction layers
        feature_layers = []
        current_dim = state_dim
        
        for i in range(num_layers):
            feature_layers.append(nn.Linear(current_dim, hidden_dim))
            feature_layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*feature_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values using dueling architecture.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values tensor of shape (batch_size, action_dim)
        """
        features = self.feature_extractor(state)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
