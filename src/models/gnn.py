"""Graph Neural Network Module

Implements a Graph Neural Network for processing network topology
and node relationships in the distributed resource allocation system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, global_add_pool


class GNNEncoder(nn.Module):
    """Graph Neural Network for encoding network topology."""
    
    def __init__(self, node_input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        """Initialize GNN encoder.
        
        Args:
            node_input_dim: Dimension of node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output embeddings
            num_layers: Number of GNN layers
        """
        super().__init__()
        
        self.node_input_dim = node_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # GNN layers
        self.conv_layers = nn.ModuleList()
        current_dim = node_input_dim
        
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else output_dim
            self.conv_layers.append(GCNConv(current_dim, next_dim))
            current_dim = next_dim
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through GNN.
        
        Args:
            x: Node feature matrix of shape (num_nodes, node_input_dim)
            edge_index: Graph connectivity tensor of shape (2, num_edges)
            batch: Batch indices for graph-level output (optional)
            
        Returns:
            Graph encoding tensor or node embeddings
        """
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
        
        # Global pooling if batch indices provided
        if batch is not None:
            x = global_add_pool(x, batch)
        
        return x
