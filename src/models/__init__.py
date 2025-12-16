"""
Models module for CAMAC-DRA.

This module exports core model classes and their configurations:
- BaseModel: Base model class for all neural network models
- DQNAgent: Deep Q-Network agent implementation
- PolicyNetwork: Policy network for actor-critic methods
- ValueNetwork: Value network for actor-critic methods
- AttentionNetwork: Attention-based network architecture

Each model has an associated configuration dataclass for hyperparameter management.
"""

from .base_model import BaseModel, BaseModelConfig
from .dqn_agent import DQNAgent, DQNAgentConfig
from .policy_network import PolicyNetwork, PolicyNetworkConfig
from .value_network import ValueNetwork, ValueNetworkConfig
from .attention_network import AttentionNetwork, AttentionNetworkConfig

__all__ = [
    # Base Model
    "BaseModel",
    "BaseModelConfig",
    # DQN Agent
    "DQNAgent",
    "DQNAgentConfig",
    # Policy Network
    "PolicyNetwork",
    "PolicyNetworkConfig",
    # Value Network
    "ValueNetwork",
    "ValueNetworkConfig",
    # Attention Network
    "AttentionNetwork",
    "AttentionNetworkConfig",
]
