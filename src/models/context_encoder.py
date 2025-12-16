"""Context Encoder Module

Responsible for encoding contextual information from the network state
and system parameters into dense representations.
"""

import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """Encodes network context into dense representations."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        """Initialize context encoder.
        
        Args:
            input_dim: Dimension of input context features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output encoding
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Build encoder layers
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(current_dim, next_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
            current_dim = next_dim
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Encode context features.
        
        Args:
            context: Input context tensor of shape (batch_size, input_dim)
            
        Returns:
            Encoded context tensor of shape (batch_size, output_dim)
        """
        return self.encoder(context)
