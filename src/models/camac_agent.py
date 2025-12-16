"""CAMAC Agent Module

Main agent implementation combining context encoding, graph neural networks,
attention mechanisms, and Q-learning for collaborative multi-agent
resource allocation in distributed systems.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .context_encoder import ContextEncoder
from .gnn import GNNEncoder
from .attention import MultiHeadAttention, TransformerBlock
from .q_network import DuelingQNetwork


class CAMACAgent(nn.Module):
    """Collaborative Multi-Agent Resource Allocation Agent."""
    
    def __init__(self, 
                 context_dim: int,
                 node_feature_dim: int,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int = 128,
                 num_gnn_layers: int = 2,
                 num_attention_heads: int = 4,
                 num_transformer_layers: int = 2,
                 hidden_dim: int = 256):
        """Initialize CAMAC Agent.
        
        Args:
            context_dim: Dimension of context features
            node_feature_dim: Dimension of node features for GNN
            state_dim: Dimension of state representation
            action_dim: Dimension of action space
            embedding_dim: Dimension of embeddings
            num_gnn_layers: Number of GNN layers
            num_attention_heads: Number of attention heads
            num_transformer_layers: Number of transformer blocks
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        self.context_dim = context_dim
        self.node_feature_dim = node_feature_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        
        # Context encoder
        self.context_encoder = ContextEncoder(
            input_dim=context_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=2
        )
        
        # Graph neural network
        self.gnn_encoder = GNNEncoder(
            node_input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_gnn_layers
        )
        
        # Attention mechanism
        self.attention = MultiHeadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=0.1
        )
        
        # Transformer blocks for feature fusion
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embedding_dim,
                num_heads=num_attention_heads,
                feed_forward_dim=hidden_dim,
                dropout=0.1
            )
            for _ in range(num_transformer_layers)
        ])
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Q-network for action selection
        self.q_network = DuelingQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=2
        )
    
    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        """Encode context information.
        
        Args:
            context: Context tensor of shape (batch_size, context_dim)
            
        Returns:
            Encoded context of shape (batch_size, embedding_dim)
        """
        return self.context_encoder(context)
    
    def encode_graph(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                     batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode graph topology.
        
        Args:
            node_features: Node feature matrix of shape (num_nodes, node_feature_dim)
            edge_index: Graph edges of shape (2, num_edges)
            batch: Batch indices (optional)
            
        Returns:
            Graph encoding of shape (batch_size, embedding_dim)
        """
        return self.gnn_encoder(node_features, edge_index, batch)
    
    def forward(self, context: torch.Tensor, node_features: torch.Tensor,
                edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through CAMAC agent.
        
        Args:
            context: Context tensor of shape (batch_size, context_dim)
            node_features: Node features of shape (num_nodes, node_feature_dim)
            edge_index: Graph edges of shape (2, num_edges)
            batch: Batch indices (optional)
            
        Returns:
            Tuple of (state_representation, q_values)
        """
        # Encode context
        context_embedding = self.encode_context(context)
        
        # Encode graph
        graph_embedding = self.encode_graph(node_features, edge_index, batch)
        
        # Apply attention
        # Reshape for attention (add sequence dimension if needed)
        context_seq = context_embedding.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        graph_seq = graph_embedding.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        combined = torch.cat([context_seq, graph_seq], dim=1)  # (batch_size, 2, embedding_dim)
        
        attended, _ = self.attention(combined, combined, combined)
        
        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            attended = transformer(attended)
        
        # Fuse features
        fused = attended.view(attended.size(0), -1)  # Flatten
        state_representation = self.fusion_layer(fused)
        
        # Compute Q-values
        q_values = self.q_network(state_representation)
        
        return state_representation, q_values
    
    def select_action(self, context: torch.Tensor, node_features: torch.Tensor,
                      edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None,
                      epsilon: float = 0.0) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            context: Context tensor
            node_features: Node features
            edge_index: Graph edges
            batch: Batch indices (optional)
            epsilon: Exploration rate
            
        Returns:
            Selected action index
        """
        import numpy as np
        
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            _, q_values = self.forward(context, node_features, edge_index, batch)
            action = q_values.argmax(dim=1).item()
        
        return action
