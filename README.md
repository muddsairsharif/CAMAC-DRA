# CAMAC-DRA: Comprehensive Adaptive Multi-Agent Coordination in Distributed Resource Allocation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
  - [Environment Module](#environment-module)
  - [Models Module](#models-module)
  - [Training Module](#training-module)
  - [Coordination Module](#coordination-module)
  - [Utils Module](#utils-module)
- [Installation Guide](#installation-guide)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Data Files Documentation](#data-files-documentation)
- [Configuration Guide](#configuration-guide)
- [Testing Procedures](#testing-procedures)
- [Troubleshooting Guide](#troubleshooting-guide)
- [Contributing Guidelines](#contributing-guidelines)
- [License](#license)
- [Contact](#contact)

---

## Overview

CAMAC-DRA (Comprehensive Adaptive Multi-Agent Coordination in Distributed Resource Allocation) is a sophisticated framework designed for coordinating multiple autonomous agents in complex distributed resource allocation scenarios. The system leverages advanced reinforcement learning, multi-agent coordination protocols, and adaptive algorithms to optimize resource distribution across heterogeneous environments.

### Key Objectives

- Enable seamless coordination between multiple autonomous agents
- Optimize resource allocation in distributed systems
- Provide adaptive learning mechanisms for dynamic environments
- Support scalability to large-scale multi-agent systems
- Facilitate efficient communication and synchronization between agents

### Use Cases

- Cloud resource management and VM allocation
- IoT network resource optimization
- Edge computing task distribution
- Supply chain coordination
- Manufacturing process optimization
- Smart grid energy distribution

---

## Features

### 🎯 Core Capabilities

- **Multi-Agent Coordination**: Sophisticated protocols for agent-to-agent communication and synchronization
- **Adaptive Resource Allocation**: Dynamic algorithm adjustments based on environmental conditions
- **Reinforcement Learning Integration**: State-of-the-art RL models for policy optimization
- **Distributed Architecture**: Support for geographically distributed agent networks
- **Real-time Monitoring**: Comprehensive logging and performance metrics tracking
- **Scalability**: Efficient handling of hundreds to thousands of agents
- **Fault Tolerance**: Resilience to agent failures and network disruptions
- **Configuration Flexibility**: Extensive customization options for different deployment scenarios

### 🔧 Technical Features

- Asynchronous message passing for non-blocking communication
- Weighted graph-based topology representation
- Multi-objective optimization support
- Customizable reward functions
- Pluggable model architectures
- Extensive data serialization support (JSON, CSV, HDF5)
- Comprehensive logging and debugging tools

---

## Project Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CAMAC-DRA Framework                      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Environment  │  │   Models     │  │   Training   │      │
│  │   Module     │  │   Module     │  │   Module     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│        │                 │                  │               │
│  ┌─────────────────────────────────────────────────┐        │
│  │       Coordination Module (Message Bus)         │        │
│  └─────────────────────────────────────────────────┘        │
│        │                                                    │
│  ┌──────────────────────────────────────────────┐          │
│  │      Utils Module (Data, Logging, Config)     │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Environment Module

The Environment Module provides the simulation and interaction layer for agents.

**Key Components:**
- `Environment`: Base environment class managing agent interactions and state transitions
- `ResourcePool`: Manages available resources and allocation constraints
- `Topology`: Represents network topology and agent connectivity
- `Dynamics`: Models environment dynamics and state evolution
- `Constraints`: Defines resource constraints and limits

**Responsibilities:**
- Initialize and manage agent environments
- Process agent actions and generate observations
- Track resource availability and constraints
- Handle episode management and reset procedures
- Generate reward signals based on allocation efficiency

**Key Classes and Methods:**
```python
class Environment:
    def __init__(self, config: Dict)
    def reset(self) -> Dict[str, np.ndarray]
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, bool]
    def render(self) -> None
    def close(self) -> None

class ResourcePool:
    def __init__(self, resources: Dict[str, float])
    def allocate(self, resource_id: str, amount: float) -> bool
    def deallocate(self, resource_id: str, amount: float) -> float
    def get_available(self, resource_id: str) -> float
    def get_utilization(self) -> Dict[str, float]

class Topology:
    def __init__(self, agents: List[str], connections: List[Tuple])
    def add_edge(self, agent1: str, agent2: str, weight: float = 1.0) -> None
    def get_neighbors(self, agent_id: str) -> List[str]
    def is_connected(self) -> bool
    def get_diameter(self) -> int
```

---

### Models Module

The Models Module contains neural network architectures and value function approximators.

**Key Components:**
- `BaseModel`: Abstract base class for all model architectures
- `DQNAgent`: Deep Q-Network implementation for single agents
- `PolicyNetwork`: Actor-critic architecture for policy-based learning
- `ValueNetwork`: Value function approximator
- `AttentionNetwork`: Multi-head attention mechanism for agent interactions
- `GraphNeuralNetwork`: GNN for topology-aware learning

**Responsibilities:**
- Define neural network architectures
- Handle model initialization and weight management
- Support different learning paradigms (value-based, policy-based, hybrid)
- Enable model serialization and deserialization
- Provide inference interfaces for decision-making

**Key Classes and Methods:**
```python
class BaseModel(nn.Module):
    def __init__(self, input_size: int, output_size: int)
    def forward(self, state: torch.Tensor) -> torch.Tensor
    def save(self, path: str) -> None
    def load(self, path: str) -> None
    def get_config(self) -> Dict

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 1e-3)
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> float
    def get_q_values(self, state: np.ndarray) -> np.ndarray

class AttentionNetwork(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 8)
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor
    def get_attention_weights(self) -> torch.Tensor
```

---

### Training Module

The Training Module orchestrates the learning process and optimization.

**Key Components:**
- `Trainer`: Main training loop coordinator
- `ExperienceBuffer`: Replay buffer for experience sampling
- `Optimizer`: Optimization algorithms (SGD, Adam, RMSprop)
- `LearningScheduler`: Learning rate and exploration scheduling
- `MetricsTracker`: Performance metrics collection and analysis
- `EpisodeManager`: Episode lifecycle management

**Responsibilities:**
- Manage training loops and iterations
- Handle experience collection and replay
- Execute parameter updates
- Track training metrics and statistics
- Support distributed training scenarios
- Implement early stopping and checkpointing

**Key Classes and Methods:**
```python
class Trainer:
    def __init__(self, env: Environment, model: BaseModel, config: Dict)
    def train(self, num_episodes: int, num_steps: int) -> Dict[str, List]
    def train_episode(self) -> Dict[str, float]
    def evaluate(self, num_episodes: int) -> Dict[str, float]
    def save_checkpoint(self, path: str) -> None
    def load_checkpoint(self, path: str) -> None

class ExperienceBuffer:
    def __init__(self, max_size: int = 100000)
    def add(self, experience: Tuple) -> None
    def sample(self, batch_size: int) -> List[Tuple]
    def clear(self) -> None
    def is_ready(self, batch_size: int) -> bool

class MetricsTracker:
    def __init__(self)
    def record(self, metric_name: str, value: float) -> None
    def get_summary(self) -> Dict[str, float]
    def reset(self) -> None
    def export(self, path: str) -> None
```

---

### Coordination Module

The Coordination Module manages inter-agent communication and synchronization.

**Key Components:**
- `MessageBus`: Central message routing and delivery system
- `Agent`: Base agent class with communication interface
- `Protocol`: Coordination protocol definitions
- `StateSync`: State synchronization mechanisms
- `ConflictResolver`: Handles resource conflicts between agents
- `Scheduler`: Agent action scheduling and ordering

**Responsibilities:**
- Route messages between agents
- Manage agent state consistency
- Resolve resource conflicts fairly
- Synchronize distributed operations
- Monitor coordination health
- Implement consensus algorithms

**Key Classes and Methods:**
```python
class MessageBus:
    def __init__(self, timeout: float = 5.0)
    def send(self, sender_id: str, recipient_id: str, message: Dict) -> bool
    def broadcast(self, sender_id: str, message: Dict) -> List[str]
    def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Dict]
    def get_message_queue(self, agent_id: str) -> List[Dict]

class Agent:
    def __init__(self, agent_id: str, env: Environment, model: BaseModel)
    def act(self, observation: np.ndarray) -> int
    def receive_message(self, message: Dict) -> None
    def send_message(self, recipient_id: str, content: Dict) -> bool
    def update_state(self, new_state: np.ndarray) -> None
    def get_status(self) -> Dict

class Protocol:
    def initiate(self) -> bool
    def coordinate(self, agents: List[Agent]) -> Dict
    def resolve_conflicts(self, conflicts: List[Tuple]) -> Dict
    def synchronize(self) -> bool

class ConflictResolver:
    def __init__(self, strategy: str = "weighted_random")
    def resolve(self, conflicts: List[Tuple[str, str, float, float]]) -> Dict[str, float]
    def get_available_strategies(self) -> List[str]
```

---

### Utils Module

The Utils Module provides utility functions and tools for the entire framework.

**Key Components:**
- `ConfigLoader`: Configuration file parsing and management
- `DataLoader`: Data loading and preprocessing
- `Logger`: Logging and debugging utilities
- `Visualizer`: Result visualization and plotting
- `FileHandler`: Data serialization and storage
- `MetricsExporter`: Metrics export and reporting
- `EnvironmentValidator`: Configuration validation

**Responsibilities:**
- Load and validate configurations
- Handle data I/O operations
- Provide logging infrastructure
- Generate visualizations and reports
- Validate inputs and environments
- Export results in various formats

**Key Classes and Methods:**
```python
class ConfigLoader:
    @staticmethod
    def load_config(path: str) -> Dict
    @staticmethod
    def validate_config(config: Dict) -> bool
    @staticmethod
    def merge_configs(base: Dict, override: Dict) -> Dict
    @staticmethod
    def get_default_config() -> Dict

class DataLoader:
    def __init__(self, data_path: str)
    def load_csv(self, filename: str) -> pd.DataFrame
    def load_json(self, filename: str) -> Dict
    def load_hdf5(self, filename: str, dataset: str) -> np.ndarray
    def save_csv(self, data: pd.DataFrame, filename: str) -> None
    def save_json(self, data: Dict, filename: str) -> None

class Logger:
    def __init__(self, name: str, level: str = "INFO")
    def info(self, message: str) -> None
    def debug(self, message: str) -> None
    def warning(self, message: str) -> None
    def error(self, message: str) -> None
    def configure_file_handler(self, log_file: str) -> None

class Visualizer:
    @staticmethod
    def plot_training_curves(metrics: Dict, output_path: str) -> None
    @staticmethod
    def plot_agent_trajectories(trajectories: Dict[str, List]) -> None
    @staticmethod
    def plot_resource_utilization(utilization: Dict[str, List]) -> None
    @staticmethod
    def create_network_graph(topology: Topology, save_path: str) -> None

class MetricsExporter:
    def __init__(self, output_format: str = "json")
    def export(self, metrics: Dict, path: str) -> None
    def export_html_report(self, metrics: Dict, path: str) -> None
    def generate_summary_statistics(self, metrics: Dict) -> Dict
```

---

## Installation Guide

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **RAM**: Minimum 8GB (16GB recommended for large-scale experiments)
- **Storage**: At least 2GB free disk space
- **GPU**: Optional (NVIDIA GPU with CUDA support recommended for faster training)

### Step 1: Clone the Repository

```bash
git clone https://github.com/muddsairsharif/CAMAC-DRA.git
cd CAMAC-DRA
```

### Step 2: Create Virtual Environment

Using Python venv:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Or using Conda:
```bash
conda create -n camac-dra python=3.8
conda activate camac-dra
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

For GPU support (optional):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Install Development Dependencies (Optional)

For development and testing:
```bash
pip install -r requirements-dev.txt
```

### Step 5: Verify Installation

```bash
python -c "import camac_dra; print('Installation successful!')"
python -m pytest tests/ -v  # Run tests
```

### Troubleshooting Installation

**Issue**: `ModuleNotFoundError: No module named 'torch'`
- **Solution**: Ensure PyTorch is installed for your system: `pip install torch`

**Issue**: `pip install` fails with permission error
- **Solution**: Use `pip install --user` or create a virtual environment

**Issue**: GPU not detected
- **Solution**: Check CUDA compatibility and reinstall PyTorch with appropriate CUDA version

---

## Quick Start

### 1. Basic Setup

```python
from camac_dra.environment import Environment
from camac_dra.models import DQNAgent
from camac_dra.training import Trainer
from camac_dra.utils import ConfigLoader

# Load configuration
config = ConfigLoader.load_config('configs/default.yaml')

# Create environment
env = Environment(config['environment'])

# Create model
model = DQNAgent(
    state_size=config['model']['state_size'],
    action_size=config['model']['action_size']
)

# Create trainer
trainer = Trainer(env, model, config)

# Run training
results = trainer.train(num_episodes=100, num_steps=1000)
```

### 2. Multi-Agent Coordination

```python
from camac_dra.coordination import MessageBus, Agent

# Create message bus
message_bus = MessageBus()

# Create agents
agents = [
    Agent(f"agent_{i}", env, model) 
    for i in range(config['num_agents'])
]

# Coordinate actions
for step in range(1000):
    # Agents perceive environment
    observations = env.step(None)
    
    # Agents make decisions
    actions = {agent.id: agent.act(observations[agent.id]) for agent in agents}
    
    # Environment processes actions
    next_obs, rewards, dones, info = env.step(actions)
```

### 3. Evaluate Performance

```python
# Evaluate trained model
eval_metrics = trainer.evaluate(num_episodes=20)
print(f"Average Reward: {eval_metrics['avg_reward']:.2f}")
print(f"Success Rate: {eval_metrics['success_rate']:.2%}")

# Export results
trainer.metrics_tracker.export('results/metrics.json')
```

---

## Usage Examples

### Example 1: Resource Allocation in Cloud Computing

```python
from camac_dra.environment import Environment, ResourcePool
from camac_dra.utils import ConfigLoader, Logger

# Configure resource pool
resource_config = {
    'cpu': 1000,
    'memory': 500,
    'bandwidth': 10000
}

# Create environment
config = ConfigLoader.load_config('configs/cloud.yaml')
env = Environment(config)
logger = Logger('cloud_allocation')

# Training loop
for episode in range(100):
    obs = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        # Agents select actions
        actions = {
            agent_id: agent.select_action(obs[agent_id])
            for agent_id in obs.keys()
        }
        
        # Execute actions
        obs, rewards, dones, info = env.step(actions)
        episode_reward += sum(rewards.values())
        done = all(dones.values())
    
    if episode % 10 == 0:
        logger.info(f"Episode {episode}: Reward={episode_reward:.2f}")
```

### Example 2: IoT Network Optimization

```python
from camac_dra.coordination import Agent, Protocol
from camac_dra.models import AttentionNetwork
import numpy as np

# Create agents for IoT network
num_agents = 50
agents = []

for i in range(num_agents):
    agent = Agent(
        agent_id=f"iot_node_{i}",
        env=env,
        model=AttentionNetwork(input_dim=64, num_heads=8)
    )
    agents.append(agent)

# Coordination protocol
protocol = Protocol(strategy='distributed_consensus')

# Run coordination
for step in range(500):
    # Agents exchange information
    for agent in agents:
        neighbors = topology.get_neighbors(agent.agent_id)
        for neighbor_id in neighbors:
            message = {'type': 'state_update', 'data': agent.state}
            agent.send_message(neighbor_id, message)
    
    # Execute coordinated actions
    protocol.coordinate(agents)
```

### Example 3: Training with Custom Reward Function

```python
from camac_dra.environment import Environment
from camac_dra.training import Trainer, MetricsTracker

# Custom reward function
def custom_reward(agent_id, allocation, resources):
    efficiency = allocation / resources
    fairness = 1.0 - np.std([allocation])
    return 0.7 * efficiency + 0.3 * fairness

# Create environment with custom reward
config = ConfigLoader.load_config('configs/default.yaml')
config['environment']['reward_fn'] = custom_reward
env = Environment(config)

# Train with metrics tracking
trainer = Trainer(env, model, config)
metrics = trainer.train(num_episodes=50, num_steps=500)

# Analyze results
tracker = MetricsTracker()
for metric_name, values in metrics.items():
    tracker.record(metric_name, np.mean(values))

print(tracker.get_summary())
```

---

## API Reference

### Environment Module API

#### `Environment`

```python
class Environment:
    """
    Main environment class for multi-agent coordination simulation.
    
    Attributes:
        config (Dict): Environment configuration
        agents (Dict[str, Agent]): Dictionary of agents
        resources (ResourcePool): Available resources
        topology (Topology): Network topology
    """
    
    def __init__(self, config: Dict):
        """Initialize environment with configuration."""
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and return initial observations."""
        
    def step(self, actions: Dict[str, int]) -> Tuple:
        """
        Execute one step of environment.
        
        Args:
            actions: Dictionary mapping agent IDs to actions
            
        Returns:
            observations: Current observations for each agent
            rewards: Reward for each agent
            dones: Whether episode is done
            info: Additional information
        """
        
    def render(self, mode: str = 'human') -> None:
        """Render current environment state."""
        
    def close(self) -> None:
        """Close environment and clean up resources."""
```

#### `ResourcePool`

```python
class ResourcePool:
    """Manages resource allocation and availability."""
    
    def allocate(self, resource_id: str, amount: float) -> bool:
        """Allocate resource. Returns True if successful."""
        
    def deallocate(self, resource_id: str, amount: float) -> float:
        """Deallocate resource. Returns deallocated amount."""
        
    def get_available(self, resource_id: str) -> float:
        """Get available amount of resource."""
        
    def get_utilization(self) -> Dict[str, float]:
        """Get utilization percentage for each resource."""
```

### Models Module API

#### `BaseModel`

```python
class BaseModel(nn.Module):
    """Abstract base class for all models."""
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through model."""
        
    def save(self, path: str) -> None:
        """Save model to file."""
        
    def load(self, path: str) -> None:
        """Load model from file."""
```

#### `DQNAgent`

```python
class DQNAgent:
    """Deep Q-Network agent for single-agent learning."""
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action using epsilon-greedy strategy."""
        
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> float:
        """Update Q-values using TD learning."""
        
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions."""
```

### Training Module API

#### `Trainer`

```python
class Trainer:
    """Main trainer for coordinating training process."""
    
    def train(self, num_episodes: int, num_steps: int) -> Dict:
        """
        Train model for specified episodes and steps.
        
        Returns:
            Dictionary containing training metrics
        """
        
    def evaluate(self, num_episodes: int) -> Dict[str, float]:
        """
        Evaluate trained model.
        
        Returns:
            Dictionary with evaluation metrics
        """
        
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
```

#### `ExperienceBuffer`

```python
class ExperienceBuffer:
    """Replay buffer for storing and sampling experiences."""
    
    def add(self, experience: Tuple) -> None:
        """Add experience to buffer."""
        
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample batch of experiences."""
        
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has sufficient experiences."""
```

### Coordination Module API

#### `MessageBus`

```python
class MessageBus:
    """Central message routing system."""
    
    def send(self, sender_id: str, recipient_id: str, message: Dict) -> bool:
        """Send message from sender to recipient."""
        
    def broadcast(self, sender_id: str, message: Dict) -> List[str]:
        """Broadcast message to all agents."""
        
    def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Dict]:
        """Receive message for agent."""
```

#### `Agent`

```python
class Agent:
    """Base agent class with coordination capabilities."""
    
    def act(self, observation: np.ndarray) -> int:
        """Select action based on observation."""
        
    def receive_message(self, message: Dict) -> None:
        """Receive message from other agent."""
        
    def send_message(self, recipient_id: str, content: Dict) -> bool:
        """Send message to other agent."""
        
    def update_state(self, new_state: np.ndarray) -> None:
        """Update agent's internal state."""
```

### Utils Module API

#### `ConfigLoader`

```python
class ConfigLoader:
    """Configuration file handling."""
    
    @staticmethod
    def load_config(path: str) -> Dict:
        """Load configuration from file."""
        
    @staticmethod
    def validate_config(config: Dict) -> bool:
        """Validate configuration."""
        
    @staticmethod
    def merge_configs(base: Dict, override: Dict) -> Dict:
        """Merge two configurations."""
```

#### `Logger`

```python
class Logger:
    """Logging utility."""
    
    def info(self, message: str) -> None:
        """Log info message."""
        
    def debug(self, message: str) -> None:
        """Log debug message."""
        
    def error(self, message: str) -> None:
        """Log error message."""
```

---

## Data Files Documentation

### Configuration Files (YAML/JSON)

**Location**: `configs/`

#### Default Configuration (`default.yaml`)

```yaml
environment:
  num_agents: 10
  num_resources: 5
  max_episode_length: 1000
  resource_limits:
    cpu: 1000
    memory: 500
    bandwidth: 10000
  
model:
  type: "dqn"
  state_size: 64
  action_size: 10
  learning_rate: 0.001
  
training:
  num_episodes: 100
  batch_size: 32
  replay_buffer_size: 100000
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995
  
coordination:
  protocol: "distributed_consensus"
  message_timeout: 5.0
  sync_interval: 10
```

### Data Files (CSV/HDF5)

**Location**: `data/`

#### Resource Data (`resources.csv`)

| timestamp | agent_id | resource_type | amount | utilization |
|-----------|----------|---------------|--------|-------------|
| 0 | agent_0 | cpu | 100 | 0.8 |
| 0 | agent_1 | memory | 250 | 0.5 |
| ... | ... | ... | ... | ... |

#### Trajectory Data (`trajectories.hdf5`)

- **Dataset**: `/trajectories/agent_{id}`
  - Shape: (num_steps, state_dim)
  - dtype: float32
  - Description: Agent state trajectory

- **Dataset**: `/actions/agent_{id}`
  - Shape: (num_steps,)
  - dtype: int32
  - Description: Agent actions taken

#### Metrics Data (`metrics.json`)

```json
{
  "training": {
    "episode_rewards": [100, 150, 200, ...],
    "episode_losses": [0.5, 0.4, 0.3, ...],
    "success_rate": 0.95
  },
  "evaluation": {
    "avg_reward": 250.5,
    "std_reward": 15.3,
    "success_rate": 0.98
  }
}
```

### Checkpoint Files

**Location**: `checkpoints/`

- `model_weights.pt`: PyTorch model weights
- `optimizer_state.pt`: Optimizer state
- `config.yaml`: Configuration snapshot
- `metrics.json`: Training metrics at checkpoint

---

## Configuration Guide

### Environment Configuration

```yaml
environment:
  # Number of agents in the system
  num_agents: 10
  
  # Number of resource types
  num_resources: 5
  
  # Maximum steps per episode
  max_episode_length: 1000
  
  # Initial resource availability
  resource_limits:
    cpu: 1000
    memory: 500
    bandwidth: 10000
  
  # Agent connectivity type: "fully_connected", "ring", "grid", "random"
  topology: "random"
  
  # Reward function: "efficiency", "fairness", "hybrid"
  reward_function: "hybrid"
  
  # Observation type: "full", "partial", "graph"
  observation_type: "partial"
```

### Model Configuration

```yaml
model:
  # Model type: "dqn", "policy_gradient", "actor_critic", "attention"
  type: "dqn"
  
  # Input state dimensionality
  state_size: 64
  
  # Output action dimensionality
  action_size: 10
  
  # Learning rate
  learning_rate: 0.001
  
  # Discount factor
  gamma: 0.99
  
  # Target network update frequency
  target_update_freq: 1000
  
  # Hidden layer sizes
  hidden_sizes: [128, 128]
  
  # Activation function: "relu", "elu", "tanh"
  activation: "relu"
```

### Training Configuration

```yaml
training:
  # Number of training episodes
  num_episodes: 100
  
  # Batch size for mini-batch updates
  batch_size: 32
  
  # Replay buffer size
  replay_buffer_size: 100000
  
  # Exploration: initial epsilon value
  epsilon_start: 1.0
  
  # Exploration: final epsilon value
  epsilon_end: 0.01
  
  # Exploration: epsilon decay rate
  epsilon_decay: 0.995
  
  # Optimizer: "adam", "sgd", "rmsprop"
  optimizer: "adam"
  
  # Enable early stopping
  early_stopping: true
  
  # Early stopping patience (episodes)
  early_stopping_patience: 20
```

### Coordination Configuration

```yaml
coordination:
  # Protocol type: "distributed_consensus", "centralized", "gossip"
  protocol: "distributed_consensus"
  
  # Message timeout in seconds
  message_timeout: 5.0
  
  # State synchronization interval (steps)
  sync_interval: 10
  
  # Conflict resolution strategy
  conflict_resolver: "weighted_random"
  
  # Maximum message queue size per agent
  max_queue_size: 1000
```

### Customizing Configuration

**Method 1: Command-line Override**

```bash
python train.py \
  --config configs/default.yaml \
  --environment.num_agents 20 \
  --training.num_episodes 200
```

**Method 2: Python Override**

```python
config = ConfigLoader.load_config('configs/default.yaml')
config['environment']['num_agents'] = 20
config['training']['num_episodes'] = 200

# Validate updated configuration
if ConfigLoader.validate_config(config):
    env = Environment(config)
```

---

## Testing Procedures

### Unit Tests

**Location**: `tests/`

Run all unit tests:
```bash
pytest tests/ -v
```

Run specific test module:
```bash
pytest tests/test_environment.py -v
```

Run tests with coverage:
```bash
pytest tests/ --cov=camac_dra --cov-report=html
```

### Test Structure

```
tests/
├── test_environment.py       # Environment module tests
├── test_models.py            # Models module tests
├── test_training.py          # Training module tests
├── test_coordination.py       # Coordination module tests
├── test_utils.py             # Utils module tests
└── integration_tests.py       # Integration tests
```

### Example Unit Test

```python
import unittest
import numpy as np
from camac_dra.environment import Environment, ResourcePool
from camac_dra.utils import ConfigLoader

class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.config = ConfigLoader.load_config('configs/default.yaml')
        self.env = Environment(self.config)
    
    def test_reset(self):
        obs = self.env.reset()
        self.assertIsInstance(obs, dict)
        self.assertEqual(len(obs), self.config['environment']['num_agents'])
    
    def test_step(self):
        self.env.reset()
        actions = {f"agent_{i}": 0 for i in range(self.config['environment']['num_agents'])}
        obs, rewards, dones, info = self.env.step(actions)
        
        self.assertIsInstance(rewards, dict)
        self.assertGreater(len(rewards), 0)
    
    def test_resource_allocation(self):
        pool = ResourcePool({'cpu': 100, 'memory': 50})
        self.assertTrue(pool.allocate('cpu', 50))
        self.assertEqual(pool.get_available('cpu'), 50)
        self.assertFalse(pool.allocate('cpu', 100))
```

### Integration Tests

```bash
pytest tests/integration_tests.py -v
```

### Performance Tests

```bash
pytest tests/ -v --benchmark
```

### Test with Specific Configuration

```bash
pytest tests/ -v --config=configs/test.yaml
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Error

**Symptom**: `RuntimeError: CUDA out of memory` or `MemoryError`

**Solutions**:
- Reduce batch size: `--training.batch_size 16`
- Reduce replay buffer size: `--training.replay_buffer_size 50000`
- Reduce number of agents: `--environment.num_agents 5`
- Use CPU instead of GPU: `export CUDA_VISIBLE_DEVICES=""`

#### 2. Slow Training

**Symptom**: Training takes too long to complete

**Solutions**:
- Use GPU acceleration: Ensure CUDA is properly installed
- Increase batch size: `--training.batch_size 64`
- Reduce state/action dimensions
- Use faster models (smaller networks)
- Enable multi-processing: `--num_workers 4`

#### 3. Model Not Converging

**Symptom**: Reward plateaus or oscillates without improvement

**Solutions**:
- Adjust learning rate: `--model.learning_rate 0.0005`
- Increase exploration time: Modify `epsilon_decay`
- Change reward function
- Verify environment configuration
- Check data normalization

#### 4. Coordination Timeout

**Symptom**: `TimeoutError: Message delivery timeout`

**Solutions**:
- Increase message timeout: `--coordination.message_timeout 10.0`
- Reduce number of agents
- Check network connectivity
- Verify agent implementation
- Review message queue

#### 5. Configuration Validation Error

**Symptom**: `ValueError: Invalid configuration`

**Solutions**:
- Validate YAML syntax
- Check all required fields are present
- Verify parameter ranges
- Use `python -m camac_dra.utils validate-config config.yaml`

#### 6. Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'camac_dra'`

**Solutions**:
- Install package: `pip install -e .`
- Ensure you're in the correct directory
- Check Python path: `echo $PYTHONPATH`
- Reinstall dependencies: `pip install -r requirements.txt`

### Debugging

#### Enable Debug Logging

```python
from camac_dra.utils import Logger

logger = Logger('camac_dra', level='DEBUG')
logger.configure_file_handler('debug.log')
```

#### Use Interactive Debugging

```python
import pdb
from camac_dra.environment import Environment

config = ConfigLoader.load_config('configs/default.yaml')
env = Environment(config)

# Set breakpoint
pdb.set_trace()
obs = env.reset()
```

#### Profile Performance

```bash
python -m cProfile -s cumtime -o profile_stats train.py
python -m pstats profile_stats
```

---

## Contributing Guidelines

### Development Setup

1. **Fork the Repository**
   ```bash
   # On GitHub, click "Fork"
   git clone https://github.com/YOUR_USERNAME/CAMAC-DRA.git
   cd CAMAC-DRA
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

### Code Style

We follow PEP 8 and use Black for code formatting.

**Format code**:
```bash
black camac_dra/ tests/
isort camac_dra/ tests/  # Sort imports
```

**Lint code**:
```bash
flake8 camac_dra/ tests/
pylint camac_dra/
```

**Type checking**:
```bash
mypy camac_dra/
```

### Commit Guidelines

Use clear, descriptive commit messages:

```
[FEATURE] Add new attention mechanism to models
- Implement multi-head attention network
- Add attention weight visualization
- Include unit tests for attention layer

[BUGFIX] Fix message timeout in coordination
- Resolve race condition in message delivery
- Add timeout validation
- Update tests

[DOCS] Update README with API reference
[TEST] Add integration tests for resource allocation
[REFACTOR] Simplify environment reset logic
```

### Pull Request Process

1. **Ensure all tests pass**
   ```bash
   pytest tests/ -v
   ```

2. **Update documentation**
   - Add docstrings to new functions
   - Update README if adding features
   - Include usage examples

3. **Create descriptive PR**
   - Clear title and description
   - Reference related issues
   - Include testing evidence

4. **Code review**
   - Address reviewer feedback
   - Make requested changes
   - Re-request review

### Adding New Modules

When adding a new module:

1. Create module file in appropriate directory
2. Implement required base classes/interfaces
3. Add comprehensive docstrings
4. Create unit tests in `tests/`
5. Update documentation and examples
6. Add configuration options if needed

**Example: Adding Custom Model**

```python
# camac_dra/models/custom_model.py
from .base import BaseModel
import torch.nn as nn

class CustomModel(BaseModel):
    """Custom model implementation.
    
    Args:
        input_size: Input dimension
        output_size: Output dimension
    """
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)
    
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)
```

### Testing New Features

```python
# tests/test_custom_model.py
import unittest
from camac_dra.models import CustomModel

class TestCustomModel(unittest.TestCase):
    def test_forward_pass(self):
        model = CustomModel(64, 10)
        output = model.forward(torch.randn(1, 64))
        self.assertEqual(output.shape, (1, 10))
```

### Issue Reporting

When reporting issues, include:

1. **Description**: Clear problem statement
2. **Steps to Reproduce**: Exact reproduction steps
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Python version, OS, hardware
6. **Logs/Errors**: Full error messages and logs
7. **Minimal Example**: Code that reproduces issue

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Attribution

If you use CAMAC-DRA in your research or project, please cite:

```bibtex
@software{camac_dra_2024,
  author = {Mudd Sair Sharif},
  title = {CAMAC-DRA: Comprehensive Adaptive Multi-Agent Coordination in Distributed Resource Allocation},
  year = {2024},
  url = {https://github.com/muddsairsharif/CAMAC-DRA}
}
```

---

## Contact

### Project Maintainer

- **Name**: Mudd Sair Sharif
- **GitHub**: [@muddsairsharif](https://github.com/muddsairsharif)

### Support & Questions

- **GitHub Issues**: Report bugs and request features
- **Discussions**: General questions and ideas
- **Email**: Contact via GitHub profile

### Contributing

We welcome contributions! Please see [Contributing Guidelines](#contributing-guidelines) section above.

### Code of Conduct

Be respectful and professional in all interactions. We are committed to providing a welcoming and inclusive environment for all contributors.

---

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- OpenAI for RL best practices and research
- Contributors and maintainers of dependencies
- Community members for feedback and suggestions

---

**Last Updated**: 2024-12-16

For the latest updates and documentation, visit the [GitHub repository](https://github.com/muddsairsharif/CAMAC-DRA).
