"""
Base environment module for multi-agent coordination and resource allocation.

This module provides abstract base classes and concrete implementations for
managing multi-agent environments with resource allocation, state management,
and inter-agent coordination.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Enumeration of available resource types."""
    COMPUTE = "compute"
    MEMORY = "memory"
    BANDWIDTH = "bandwidth"
    STORAGE = "storage"
    ENERGY = "energy"


@dataclass
class Resource:
    """Represents a resource with capacity and current usage."""
    resource_type: ResourceType
    capacity: float
    current_usage: float = 0.0
    
    @property
    def available(self) -> float:
        """Return available resource capacity."""
        return max(0.0, self.capacity - self.current_usage)
    
    @property
    def utilization(self) -> float:
        """Return utilization ratio (0.0 to 1.0)."""
        return min(1.0, self.current_usage / self.capacity) if self.capacity > 0 else 0.0


@dataclass
class AgentState:
    """Represents the state of a single agent."""
    agent_id: str
    position: np.ndarray
    resources: Dict[ResourceType, Resource] = field(default_factory=dict)
    allocated_resources: Dict[str, float] = field(default_factory=dict)
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentConfig:
    """Configuration for the environment."""
    num_agents: int = 4
    num_resource_types: int = 5
    max_episode_steps: int = 1000
    grid_size: int = 10
    resource_capacities: Dict[ResourceType, float] = field(default_factory=lambda: {
        ResourceType.COMPUTE: 100.0,
        ResourceType.MEMORY: 200.0,
        ResourceType.BANDWIDTH: 150.0,
        ResourceType.STORAGE: 500.0,
        ResourceType.ENERGY: 300.0,
    })
    observation_space_size: int = 64
    action_space_size: int = 32


class BaseEnvironment(ABC):
    """
    Abstract base class for multi-agent environments with resource allocation.
    
    This class defines the interface for environments supporting:
    - Multi-agent coordination
    - Resource allocation and management
    - State observation and action execution
    - Episode management (reset/step)
    - Visualization and cleanup
    """
    
    def __init__(self, config: Optional[EnvironmentConfig] = None):
        """
        Initialize the base environment.
        
        Args:
            config: Environment configuration. If None, uses default config.
        """
        self.config = config or EnvironmentConfig()
        self.agents: Dict[str, AgentState] = {}
        self.current_step: int = 0
        self.is_closed: bool = False
        self._initialize_environment()
    
    @abstractmethod
    def _initialize_environment(self) -> None:
        """Initialize environment-specific components."""
        pass
    
    @abstractmethod
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment to initial state.
        
        Returns:
            Dictionary mapping agent IDs to initial observations.
        """
        pass
    
    @abstractmethod
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],  # rewards
        Dict[str, bool],  # dones
        Dict[str, Dict[str, Any]]  # info
    ]:
        """
        Execute one step of the environment.
        
        Args:
            actions: Dictionary mapping agent IDs to action arrays.
        
        Returns:
            Tuple of (observations, rewards, dones, info).
        """
        pass
    
    @abstractmethod
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment state.
        
        Args:
            mode: Rendering mode ("human", "rgb_array", etc.).
        
        Returns:
            Rendered output (e.g., RGB array for "rgb_array" mode).
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean up environment resources."""
        pass
    
    @abstractmethod
    def allocate_resources(self, agent_id: str, resources: Dict[ResourceType, float]) -> bool:
        """
        Allocate resources to an agent.
        
        Args:
            agent_id: ID of the agent.
            resources: Dictionary mapping ResourceType to requested amount.
        
        Returns:
            True if allocation successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def deallocate_resources(self, agent_id: str, resources: Dict[ResourceType, float]) -> bool:
        """
        Deallocate resources from an agent.
        
        Args:
            agent_id: ID of the agent.
            resources: Dictionary mapping ResourceType to amount to deallocate.
        
        Returns:
            True if deallocation successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_observation(self, agent_id: str) -> np.ndarray:
        """
        Get observation for a specific agent.
        
        Args:
            agent_id: ID of the agent.
        
        Returns:
            Observation as a numpy array.
        """
        pass
    
    @abstractmethod
    def get_resource_state(self) -> Dict[ResourceType, Dict[str, float]]:
        """
        Get current global resource state.
        
        Returns:
            Dictionary containing resource statistics.
        """
        pass
    
    def is_done(self) -> bool:
        """Check if episode is complete."""
        return self.current_step >= self.config.max_episode_steps
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class Environment(BaseEnvironment):
    """
    Concrete implementation of multi-agent environment with resource allocation.
    
    Features:
    - Multi-agent state management
    - Resource allocation and tracking
    - Reward computation for coordination
    - Observation generation
    - Episode management
    """
    
    def __init__(self, config: Optional[EnvironmentConfig] = None):
        """
        Initialize the environment.
        
        Args:
            config: Environment configuration.
        """
        super().__init__(config)
    
    def _initialize_environment(self) -> None:
        """Initialize environment components."""
        logger.info(f"Initializing environment with {self.config.num_agents} agents")
        
        # Initialize agents
        for i in range(self.config.num_agents):
            agent_id = f"agent_{i}"
            
            # Initialize resources for each agent
            resources = {}
            for resource_type, capacity in self.config.resource_capacities.items():
                resources[resource_type] = Resource(
                    resource_type=resource_type,
                    capacity=capacity
                )
            
            # Create agent state
            agent_state = AgentState(
                agent_id=agent_id,
                position=np.random.rand(2) * self.config.grid_size,
                resources=resources
            )
            self.agents[agent_id] = agent_state
        
        self.current_step = 0
        self.global_resources: Dict[ResourceType, float] = {
            rt: self.config.resource_capacities[rt] * self.config.num_agents
            for rt in ResourceType
        }
        logger.info("Environment initialization complete")
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment to initial state.
        
        Returns:
            Dictionary mapping agent IDs to initial observations.
        """
        logger.info("Resetting environment")
        
        # Reset step counter
        self.current_step = 0
        
        # Reset all agents
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            
            # Reset position
            agent.position = np.random.rand(2) * self.config.grid_size
            
            # Reset resources
            for resource_type in agent.resources:
                agent.resources[resource_type].current_usage = 0.0
            
            # Reset state
            agent.reward = 0.0
            agent.done = False
            agent.allocated_resources = {}
            agent.info = {}
        
        # Generate initial observations
        observations = {
            agent_id: self.get_observation(agent_id)
            for agent_id in self.agents
        }
        
        return observations
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Dict[str, Any]]
    ]:
        """
        Execute one step of the environment.
        
        Args:
            actions: Dictionary mapping agent IDs to action arrays.
        
        Returns:
            Tuple of (observations, rewards, dones, info).
        """
        if self.is_closed:
            raise RuntimeError("Cannot step a closed environment")
        
        self.current_step += 1
        
        # Process actions for each agent
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        for agent_id, action in actions.items():
            if agent_id not in self.agents:
                logger.warning(f"Unknown agent: {agent_id}")
                continue
            
            # Update agent state based on action
            self._process_agent_action(agent_id, action)
            
            # Get observation
            observations[agent_id] = self.get_observation(agent_id)
            
            # Calculate reward
            rewards[agent_id] = self._compute_reward(agent_id)
            
            # Check if done
            dones[agent_id] = self.is_done()
            
            # Additional info
            infos[agent_id] = {
                "step": self.current_step,
                "position": self.agents[agent_id].position.copy(),
                "resource_utilization": self._get_agent_resource_utilization(agent_id)
            }
        
        return observations, rewards, dones, infos
    
    def _process_agent_action(self, agent_id: str, action: np.ndarray) -> None:
        """
        Process action for an agent.
        
        Args:
            agent_id: ID of the agent.
            action: Action array.
        """
        agent = self.agents[agent_id]
        
        # Interpret action as resource allocation request
        # First part: movement, rest: resource allocation
        if len(action) >= 2:
            # Update position (with bounds checking)
            delta_pos = action[:2] * 0.1  # Scale movement
            new_position = agent.position + delta_pos
            agent.position = np.clip(new_position, 0, self.config.grid_size)
        
        # Process resource requests if action vector is long enough
        if len(action) > 2:
            resource_requests = {}
            resource_types = list(ResourceType)
            
            for i, resource_type in enumerate(resource_types):
                if i + 2 < len(action):
                    requested_amount = action[i + 2] * self.config.resource_capacities[resource_type]
                    resource_requests[resource_type] = max(0.0, requested_amount)
            
            # Try to allocate requested resources
            self.allocate_resources(agent_id, resource_requests)
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment state.
        
        Args:
            mode: Rendering mode ("human", "rgb_array").
        
        Returns:
            Rendered output (RGB array for "rgb_array" mode).
        """
        if mode == "human":
            self._render_human()
            return None
        elif mode == "rgb_array":
            return self._render_rgb_array()
        else:
            logger.warning(f"Unknown render mode: {mode}")
            return None
    
    def _render_human(self) -> None:
        """Render environment to console."""
        print(f"\n{'='*60}")
        print(f"Step: {self.current_step}/{self.config.max_episode_steps}")
        print(f"{'='*60}")
        
        for agent_id, agent in self.agents.items():
            print(f"\n{agent_id}:")
            print(f"  Position: {agent.position}")
            print(f"  Resources:")
            for resource_type, resource in agent.resources.items():
                print(f"    {resource_type.value}: {resource.current_usage:.2f}/{resource.capacity:.2f} "
                      f"(util: {resource.utilization*100:.1f}%)")
            print(f"  Reward: {agent.reward:.4f}")
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render environment as RGB array."""
        # Create a simple 2D visualization
        canvas_size = 256
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        
        # Draw agents
        for agent_id, agent in self.agents.items():
            # Scale position to canvas
            x = int((agent.position[0] / self.config.grid_size) * canvas_size)
            y = int((agent.position[1] / self.config.grid_size) * canvas_size)
            
            # Draw agent as a circle
            x = np.clip(x, 5, canvas_size - 5)
            y = np.clip(y, 5, canvas_size - 5)
            
            # Color based on resource utilization
            utilization = self._get_agent_resource_utilization(agent_id)
            color = int(utilization * 255)
            canvas[max(0, y-5):min(canvas_size, y+5),
                   max(0, x-5):min(canvas_size, x+5)] = [color, 0, 255-color]
        
        return canvas
    
    def close(self) -> None:
        """Close the environment and clean up resources."""
        if not self.is_closed:
            logger.info("Closing environment")
            self.is_closed = True
            # Additional cleanup if needed
    
    def allocate_resources(self, agent_id: str, resources: Dict[ResourceType, float]) -> bool:
        """
        Allocate resources to an agent.
        
        Args:
            agent_id: ID of the agent.
            resources: Dictionary mapping ResourceType to requested amount.
        
        Returns:
            True if allocation successful, False otherwise.
        """
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found")
            return False
        
        agent = self.agents[agent_id]
        
        # Check if requested resources are available
        for resource_type, requested_amount in resources.items():
            if resource_type not in agent.resources:
                logger.warning(f"Unknown resource type: {resource_type}")
                return False
            
            resource = agent.resources[resource_type]
            if resource.available < requested_amount:
                logger.debug(f"Insufficient {resource_type.value} for agent {agent_id}")
                return False
        
        # Allocate resources
        for resource_type, requested_amount in resources.items():
            agent.resources[resource_type].current_usage += requested_amount
            agent.allocated_resources[resource_type.value] = requested_amount
        
        return True
    
    def deallocate_resources(self, agent_id: str, resources: Dict[ResourceType, float]) -> bool:
        """
        Deallocate resources from an agent.
        
        Args:
            agent_id: ID of the agent.
            resources: Dictionary mapping ResourceType to amount to deallocate.
        
        Returns:
            True if deallocation successful, False otherwise.
        """
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found")
            return False
        
        agent = self.agents[agent_id]
        
        # Deallocate resources
        for resource_type, amount in resources.items():
            if resource_type not in agent.resources:
                logger.warning(f"Unknown resource type: {resource_type}")
                return False
            
            resource = agent.resources[resource_type]
            resource.current_usage = max(0.0, resource.current_usage - amount)
            
            # Update allocated resources
            if resource_type.value in agent.allocated_resources:
                agent.allocated_resources[resource_type.value] = max(
                    0.0, agent.allocated_resources[resource_type.value] - amount
                )
        
        return True
    
    def get_observation(self, agent_id: str) -> np.ndarray:
        """
        Get observation for a specific agent.
        
        Args:
            agent_id: ID of the agent.
        
        Returns:
            Observation as a numpy array.
        """
        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent: {agent_id}")
        
        agent = self.agents[agent_id]
        
        # Construct observation
        observation = []
        
        # Add agent's own state
        observation.extend(agent.position)
        observation.append(self.current_step / self.config.max_episode_steps)
        
        # Add resource states
        for resource_type in ResourceType:
            if resource_type in agent.resources:
                resource = agent.resources[resource_type]
                observation.append(resource.utilization)
        
        # Add other agents' relative positions and states
        for other_id, other_agent in self.agents.items():
            if other_id != agent_id:
                relative_pos = other_agent.position - agent.position
                observation.extend(relative_pos)
                
                # Add other agent's resource utilization
                for resource_type in ResourceType:
                    if resource_type in other_agent.resources:
                        observation.append(other_agent.resources[resource_type].utilization)
        
        # Pad or truncate to observation space size
        observation = np.array(observation, dtype=np.float32)
        if len(observation) < self.config.observation_space_size:
            observation = np.pad(observation, (0, self.config.observation_space_size - len(observation)))
        else:
            observation = observation[:self.config.observation_space_size]
        
        return observation
    
    def get_resource_state(self) -> Dict[ResourceType, Dict[str, float]]:
        """
        Get current global resource state.
        
        Returns:
            Dictionary containing resource statistics.
        """
        state = {}
        
        for resource_type in ResourceType:
            total_capacity = 0.0
            total_usage = 0.0
            
            for agent in self.agents.values():
                if resource_type in agent.resources:
                    resource = agent.resources[resource_type]
                    total_capacity += resource.capacity
                    total_usage += resource.current_usage
            
            state[resource_type] = {
                "total_capacity": total_capacity,
                "total_usage": total_usage,
                "total_available": max(0.0, total_capacity - total_usage),
                "utilization": min(1.0, total_usage / total_capacity) if total_capacity > 0 else 0.0
            }
        
        return state
    
    def _compute_reward(self, agent_id: str) -> float:
        """
        Compute reward for an agent.
        
        Reward based on:
        - Resource efficiency (low utilization penalty)
        - Coordination with other agents
        - Task completion
        
        Args:
            agent_id: ID of the agent.
        
        Returns:
            Reward value.
        """
        agent = self.agents[agent_id]
        reward = 0.0
        
        # Reward for efficient resource utilization
        utilization = self._get_agent_resource_utilization(agent_id)
        reward += utilization * 0.3  # Reward for resource usage
        
        # Penalty for high utilization (avoid overuse)
        if utilization > 0.9:
            reward -= 0.1
        
        # Reward for cooperation (distance to other agents)
        cooperation_reward = 0.0
        for other_id, other_agent in self.agents.items():
            if other_id != agent_id:
                distance = np.linalg.norm(agent.position - other_agent.position)
                # Reward for being close but not too close (optimal distance ~ grid_size/2)
                optimal_distance = self.config.grid_size / 2
                cooperation_reward += 1.0 / (1.0 + abs(distance - optimal_distance))
        
        cooperation_reward /= max(1, len(self.agents) - 1)
        reward += cooperation_reward * 0.3
        
        # Small penalty for each step (encourage efficiency)
        reward -= 0.01
        
        agent.reward = reward
        return reward
    
    def _get_agent_resource_utilization(self, agent_id: str) -> float:
        """
        Get average resource utilization for an agent.
        
        Args:
            agent_id: ID of the agent.
        
        Returns:
            Utilization ratio (0.0 to 1.0).
        """
        if agent_id not in self.agents:
            return 0.0
        
        agent = self.agents[agent_id]
        if not agent.resources:
            return 0.0
        
        utilization_sum = sum(resource.utilization for resource in agent.resources.values())
        return utilization_sum / len(agent.resources)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create environment with default configuration
    config = EnvironmentConfig(num_agents=4, max_episode_steps=100)
    
    with Environment(config) as env:
        # Reset environment
        observations = env.reset()
        print(f"Initial observations received for {len(observations)} agents")
        
        # Run a few steps
        for step in range(10):
            # Generate random actions for all agents
            actions = {
                agent_id: np.random.randn(env.config.action_space_size)
                for agent_id in observations.keys()
            }
            
            # Execute step
            observations, rewards, dones, infos = env.step(actions)
            
            # Render
            env.render(mode="human")
            
            # Print resource state
            resource_state = env.get_resource_state()
            print(f"\nGlobal Resource State (Step {step + 1}):")
            for resource_type, stats in resource_state.items():
                print(f"  {resource_type.value}: {stats['utilization']*100:.1f}% utilized")
