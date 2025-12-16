"""
Training module for CAMAC-DRA

This module provides core training components including:
- ReplayBuffer: Experience storage and sampling
- RewardComputer: Reward computation utilities
- Trainer: Main training orchestrator
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Replay buffer for storing and sampling training experience.
    
    Implements a fixed-size circular buffer for storing transitions
    (state, action, reward, next_state, done) from agent-environment interactions.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: Any, action: Any, reward: float, 
            next_state: Any, done: bool) -> None:
        """
        Add a transition to the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting next state
            done: Whether the episode is finished
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[List, List, List, List, List]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer size ({len(self.buffer)}) is less than batch size ({batch_size})")
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return list(states), list(actions), list(rewards), list(next_states), list(dones)
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if the buffer is at capacity."""
        return len(self.buffer) == self.capacity
    
    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self.buffer.clear()


class RewardComputer:
    """
    Compute and process rewards for training.
    
    Handles reward calculations, normalization, and preprocessing
    for training algorithms.
    """
    
    def __init__(self, discount_factor: float = 0.99, 
                 normalize: bool = True):
        """
        Initialize the reward computer.
        
        Args:
            discount_factor: Discount factor (gamma) for future rewards
            normalize: Whether to normalize rewards
        """
        self.discount_factor = discount_factor
        self.normalize = normalize
        self.reward_history = deque(maxlen=1000)
    
    def compute_return(self, rewards: List[float], 
                      dones: List[bool]) -> List[float]:
        """
        Compute discounted cumulative returns from rewards.
        
        Args:
            rewards: List of rewards for a trajectory
            dones: List of done flags for each step
            
        Returns:
            List of discounted returns for each step
        """
        returns = []
        cumulative_return = 0.0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            cumulative_return = reward + self.discount_factor * cumulative_return * (1 - done)
            returns.insert(0, cumulative_return)
        
        return returns
    
    def compute_advantages(self, rewards: List[float], 
                          values: List[float], 
                          dones: List[bool]) -> List[float]:
        """
        Compute advantages using rewards and value estimates.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            
        Returns:
            List of advantages for each step
        """
        advantages = []
        gae = 0.0
        lambda_param = 0.95
        
        for t in range(len(rewards) - 1, -1, -1):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.discount_factor * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.discount_factor * lambda_param * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def normalize_rewards(self, rewards: List[float]) -> List[float]:
        """
        Normalize rewards using running statistics.
        
        Args:
            rewards: List of rewards to normalize
            
        Returns:
            List of normalized rewards
        """
        if not self.normalize or len(self.reward_history) == 0:
            return rewards
        
        mean = np.mean(list(self.reward_history))
        std = np.std(list(self.reward_history)) + 1e-8
        
        normalized = [(r - mean) / std for r in rewards]
        return normalized
    
    def update_history(self, rewards: List[float]) -> None:
        """
        Update reward history for normalization.
        
        Args:
            rewards: List of rewards to add to history
        """
        self.reward_history.extend(rewards)


class Trainer:
    """
    Main training orchestrator for CAMAC-DRA agents.
    
    Coordinates the training loop, manages replay buffer,
    computes rewards, and updates model parameters.
    """
    
    def __init__(self, replay_buffer: Optional[ReplayBuffer] = None,
                 reward_computer: Optional[RewardComputer] = None,
                 batch_size: int = 32,
                 learning_rate: float = 1e-4):
        """
        Initialize the trainer.
        
        Args:
            replay_buffer: ReplayBuffer instance (creates default if None)
            reward_computer: RewardComputer instance (creates default if None)
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
        """
        self.replay_buffer = replay_buffer or ReplayBuffer()
        self.reward_computer = reward_computer or RewardComputer()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.training_step = 0
        self.episode_count = 0
        self.total_reward = 0.0
    
    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Perform a single training step.
        
        Samples from replay buffer and performs model update.
        
        Returns:
            Dictionary of training metrics or None if buffer is too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Update reward history for normalization
        self.reward_computer.update_history(rewards)
        
        # Normalize rewards
        normalized_rewards = self.reward_computer.normalize_rewards(rewards)
        
        self.training_step += 1
        
        metrics = {
            "training_step": self.training_step,
            "batch_size": self.batch_size,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
        }
        
        return metrics
    
    def add_experience(self, state: Any, action: Any, reward: float,
                      next_state: Any, done: bool) -> None:
        """
        Add experience to the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting next state
            done: Whether episode is done
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.total_reward += reward
    
    def reset_episode(self) -> None:
        """Reset episode counter and total reward."""
        self.episode_count += 1
        self.total_reward = 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current training metrics.
        
        Returns:
            Dictionary of training metrics
        """
        return {
            "training_steps": self.training_step,
            "episodes": self.episode_count,
            "buffer_size": len(self.replay_buffer),
            "buffer_capacity": self.replay_buffer.capacity,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
        }


__all__ = [
    "ReplayBuffer",
    "RewardComputer",
    "Trainer",
]
