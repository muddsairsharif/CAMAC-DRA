# CAMAC-DRA: Context-Aware Multi-Agent Coordination for Dynamic Resource Allocation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![GitHub Stars](https://img.shields.io/github/stars/yourusername/camac-dra?style=social)](https://github.com/yourusername/camac-dra)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Development Roadmap](#development-roadmap)
- [Week-by-Week Guide](#week-by-week-guide)
- [Documentation](#documentation)
- [Results](#results)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

**CAMAC-DRA** is a cutting-edge deep reinforcement learning framework for intelligent electric vehicle (EV) charging coordination. The system enables autonomous agents to coordinate resource allocation across large-scale EV networks (250+ vehicles, 45+ charging stations) while dynamically adapting to real-time environmental conditions.

### Key Achievements
- 🏆 **92% coordination success rate** (10% better than state-of-the-art)
- ⚡ **15% energy efficiency improvement**
- 💰 **10% operational cost reduction**
- 🔋 **20% grid strain decrease**
- 🚀 **2.3× faster convergence** than baseline methods

### Research Paper
This implementation is based on our paper:
> **"Context-Aware Multi-Agent Coordination Framework for Intelligent Electric Vehicle Charging Optimization: A Deep Reinforcement Learning Approach"**  
> Muddsair Sharif, Huseyin Seker, Yasir Javed  
> *IEEE Access*, 2025

[📄 Read Paper](link-to-paper) | [📊 View Results](docs/results.md) | [🎬 Watch Demo](link-to-demo)

---

## ✨ Key Features

### 🧠 Advanced AI Architecture
- **Graph Neural Networks (GNN)** for heterogeneous graph modeling
- **Multi-Head Attention** mechanisms (8 heads) for context processing
- **Multi-Stakeholder Q-Networks** balancing 5 agent objectives
- **Hierarchical Coordination** using PSO and GA algorithms

### 🌍 Context-Aware Decision Making
- Processes **20 contextual features**:
  - Weather conditions (temperature, solar irradiance)
  - Traffic patterns (congestion, travel times)
  - Grid status (load, voltage, frequency)
  - Electricity pricing (time-of-use tariffs)
  - Renewable energy availability

### 🤝 Multi-Stakeholder Optimization
Balances competing objectives across:
- **EV Users** (25%): Cost minimization, convenience
- **Grid Operators** (20%): Stability, load balancing
- **Station Operators** (20%): Utilization, revenue
- **Fleet Managers** (20%): Availability, planning
- **Environmental Entities** (15%): Sustainability, renewables

### 📊 Comprehensive Validation
- Tested on **441,077 real charging transactions**
- Validated across diverse operational conditions
- Comparison with 6 baseline algorithms (DQN, DDPG, A3C, PPO, GNN)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                               │
│  Weather | Traffic | Grid | Pricing | Spatial-Temporal     │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              CONTEXT ENCODING LAYER                          │
│  LSTM Networks + Feature Normalization + Embeddings        │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          GRAPH NEURAL NETWORK LAYER                          │
│  Heterogeneous Graph + Message Passing (3 layers)          │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           ATTENTION MECHANISM LAYER                          │
│  Multi-Head Attention (8 heads) + Context Weighting        │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│       MULTI-STAKEHOLDER Q-NETWORK LAYER                      │
│  5 Specialized Networks with Weighted Aggregation          │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              OPTIMAL ACTIONS                                 │
│  Charging assignments | Power allocation | Schedules        │
└─────────────────────────────────────────────────────────────┘
```

[View Detailed Architecture](docs/architecture.md)

---

## 📁 Repository Structure

```
CAMAC-DRA/
│
├── 📄 README.md                    # This file
├── 📄 LICENSE                      # MIT License
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package installation
├── 📄 .gitignore                   # Git ignore rules
│
├── 📂 config/                      # Configuration files
│   ├── hyperparameters.yaml        # Model hyperparameters
│   ├── environment.yaml            # Environment settings
│   └── training.yaml               # Training configurations
│
├── 📂 src/                         # Source code
│   ├── __init__.py
│   │
│   ├── 📂 models/                  # Neural network models
│   │   ├── __init__.py
│   │   ├── context_encoder.py     # LSTM-based encoder
│   │   ├── gnn.py                 # Graph Neural Network
│   │   ├── attention.py           # Multi-head attention
│   │   ├── q_network.py           # Multi-stakeholder Q-net
│   │   └── camac_agent.py         # Complete agent
│   │
│   ├── 📂 environment/             # Simulation environment
│   │   ├── __init__.py
│   │   ├── ev_env.py              # Main environment
│   │   ├── ev.py                  # EV class
│   │   ├── charging_station.py    # Station class
│   │   └── grid_simulator.py      # Grid simulation
│   │
│   ├── 📂 training/                # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py             # Training loop
│   │   ├── replay_buffer.py       # Experience replay
│   │   └── optimizer.py           # Custom optimizers
│   │
│   ├── 📂 coordination/            # Coordination algorithms
│   │   ├── __init__.py
│   │   ├── pso.py                 # Particle Swarm Optimization
│   │   └── ga.py                  # Genetic Algorithm
│   │
│   └── 📂 utils/                   # Utility functions
│       ├── __init__.py
│       ├── metrics.py             # Performance metrics
│       ├── visualization.py       # Plotting functions
│       ├── logger.py              # Logging utilities
│       └── data_loader.py         # Data processing
│
├── 📂 data/                        # Data directory
│   ├── 📂 raw/                     # Raw data
│   │   ├── weather_data.csv
│   │   ├── traffic_data.csv
│   │   ├── pricing_data.csv
│   │   └── charging_transactions.csv
│   │
│   ├── 📂 processed/               # Processed data
│   │   └── dataset_processed.pkl
│   │
│   └── 📂 samples/                 # Sample datasets
│       └── sample_data.csv
│
├── 📂 notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_environment_demo.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_results_analysis.ipynb
│   └── 05_visualization.ipynb
│
├── 📂 scripts/                     # Executable scripts
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   ├── demo.py                    # Quick demo
│   └── preprocess_data.py         # Data preprocessing
│
├── 📂 tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_environment.py
│   ├── test_training.py
│   └── test_utils.py
│
├── 📂 docs/                        # Documentation
│   ├── architecture.md            # Architecture details
│   ├── api_reference.md           # API documentation
│   ├── results.md                 # Experimental results
│   ├── weekly_guide.md            # Development guide
│   └── troubleshooting.md         # Common issues
│
├── 📂 experiments/                 # Experiment configs & results
│   ├── baseline_comparison/
│   ├── ablation_studies/
│   └── sensitivity_analysis/
│
├── 📂 models/                      # Saved models
│   ├── checkpoints/
│   │   ├── epoch_10.pth
│   │   ├── epoch_50.pth
│   │   └── epoch_150.pth
│   └── best_model.pth
│
└── 📂 results/                     # Results & outputs
    ├── 📂 figures/                 # Generated plots
    ├── 📂 logs/                    # Training logs
    └── 📂 reports/                 # Experiment reports
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB+ RAM recommended

### Option 1: Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/camac-dra.git
cd camac-dra

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### Option 2: Manual Install

```bash
# Clone the repository
git clone https://github.com/yourusername/camac-dra.git
cd camac-dra

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Development Install

```bash
# Clone repository
git clone https://github.com/yourusername/camac-dra.git
cd camac-dra

# Install in development mode with all dependencies
pip install -e ".[dev,docs,test]"
```

### Verify Installation

```bash
# Run tests
pytest tests/

# Check imports
python -c "from src.models import CAMACAgent; print('✓ Installation successful!')"
```

---

## ⚡ Quick Start

### 1. Run Demo (5 minutes)

```bash
# Run the quick demo
python scripts/demo.py

# Expected output:
# ✓ Environment initialized (250 EVs, 45 stations)
# ✓ Training started...
# ✓ Episode 10/50: Reward=15.3, Success=87%
# ✓ Training completed! Success rate: 89.5%
```

### 2. Train Full Model (2-3 hours)

```bash
# Train with default settings
python scripts/train.py

# Train with custom configuration
python scripts/train.py --config config/training.yaml --episodes 150

# Train on GPU
python scripts/train.py --device cuda --batch-size 128
```

### 3. Evaluate Model

```bash
# Evaluate trained model
python scripts/evaluate.py --model models/best_model.pth

# Compare with baselines
python scripts/evaluate.py --compare --baselines DQN,DDPG,PPO
```

### 4. Interactive Notebook

```bash
# Launch Jupyter
jupyter notebook notebooks/03_model_training.ipynb
```

---

## 📖 Usage

### Basic Training Example

```python
from src.models import CAMACAgent
from src.environment import EVChargingEnvironment
from src.training import Trainer

# Initialize environment
env = EVChargingEnvironment(num_evs=250, num_stations=45)

# Create agent
agent = CAMACAgent(
    state_dim=21,
    action_dim=100,
    context_dim=20
)

# Train
trainer = Trainer(agent, env)
results = trainer.train(num_episodes=150)

# Save model
agent.save('models/my_model.pth')
```

### Advanced Configuration

```python
# Load configuration
from src.utils import load_config

config = load_config('config/hyperparameters.yaml')

# Create agent with custom settings
agent = CAMACAgent(
    state_dim=config['state_dim'],
    action_dim=config['action_dim'],
    learning_rate=0.0001,
    gamma=0.99,
    epsilon_decay=0.995
)

# Custom training loop
for episode in range(150):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.train()
        state = next_state
```

### Evaluation Example

```python
from src.utils.metrics import calculate_metrics

# Evaluate agent
metrics = calculate_metrics(agent, env, num_episodes=10)

print(f"Coordination Success: {metrics['coordination_success']:.2f}%")
print(f"Energy Efficiency: {metrics['energy_efficiency']:.2f}%")
print(f"Cost Reduction: {metrics['cost_reduction']:.2f}%")
```

---

## 🗓️ Development Roadmap

### ✅ Phase 1: Core Framework (Completed)
- [x] Environment implementation
- [x] Graph Neural Network
- [x] Multi-head attention mechanism
- [x] Multi-stakeholder Q-network
- [x] Training pipeline
- [x] Basic visualization

### 🔄 Phase 2: Enhanced Features (In Progress)
- [x] Real data integration
- [ ] PSO/GA hierarchical coordination
- [ ] Advanced visualization dashboard
- [ ] Distributed training support
- [ ] Model compression

### 📅 Phase 3: Production Ready (Planned)
- [ ] REST API for deployment
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure)
- [ ] Real-time monitoring
- [ ] Web interface

### 🚀 Phase 4: Advanced Research (Future)
- [ ] Federated learning integration
- [ ] Quantum computing optimization
- [ ] Transfer learning across cities
- [ ] Multi-objective optimization
- [ ] Explainable AI features

---

## 📅 Week-by-Week Development Guide

### Week 1-2: Environment Setup & Familiarization

**Goals:**
- Understand the framework architecture
- Set up development environment
- Run basic examples

**Tasks:**
```bash
# Day 1-2: Installation and setup
git clone https://github.com/yourusername/camac-dra.git
pip install -e .
pytest tests/

# Day 3-4: Explore notebooks
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_environment_demo.ipynb

# Day 5-7: Run quick demo
python scripts/demo.py
python scripts/train.py --episodes 10  # Quick test
```

**Learning Resources:**
- [📄 Architecture Documentation](docs/architecture.md)
- [📹 Video Tutorial: Getting Started](link-to-video)
- [📝 Blog Post: Understanding CAMAC-DRA](link-to-blog)

**Deliverables:**
- ✓ Successful installation
- ✓ All tests passing
- ✓ Demo running successfully

---

### Week 3-4: Core Components Deep Dive

**Goals:**
- Understand Graph Neural Networks
- Implement attention mechanisms
- Customize Q-networks

**Tasks:**
```python
# Study GNN implementation
from src.models import HeterogeneousGNN
# Read: src/models/gnn.py
# Modify and experiment

# Implement custom attention
from src.models import MultiHeadAttention
# Experiment with different head counts

# Customize Q-network
from src.models import MultiStakeholderQNetwork
# Try different architectures
```

**Learning Resources:**
- [📖 GNN Tutorial](docs/gnn_tutorial.md)
- [📖 Attention Mechanism Guide](docs/attention_guide.md)
- [📚 Research Papers](docs/references.md)

**Exercises:**
1. Modify GNN to use 5 layers instead of 3
2. Implement 16-head attention mechanism
3. Add a new stakeholder to Q-network

**Deliverables:**
- ✓ Custom GNN implementation
- ✓ Modified attention mechanism
- ✓ Extended Q-network

---

### Week 5-6: Training Pipeline Mastery

**Goals:**
- Master training loop
- Implement advanced techniques
- Optimize hyperparameters

**Tasks:**
```python
# Hyperparameter tuning
python scripts/train.py --lr 0.0001 --gamma 0.99 --batch-size 128

# Implement early stopping
# Add learning rate scheduling
# Try different optimizers

# Monitor training
tensorboard --logdir results/logs/
```

**Key Concepts:**
- Experience replay
- Target network updates
- Exploration-exploitation balance
- Loss function optimization

**Experiments to Run:**
- [ ] Baseline: Default hyperparameters
- [ ] Experiment 1: Larger batch size (256)
- [ ] Experiment 2: Higher learning rate (0.001)
- [ ] Experiment 3: Different epsilon decay
- [ ] Experiment 4: Adjusted discount factor

**Deliverables:**
- ✓ Training runs logged
- ✓ Hyperparameter comparison report
- ✓ Best configuration identified

---

### Week 7-8: Advanced Features & Optimization

**Goals:**
- Implement hierarchical coordination (PSO/GA)
- Add real data integration
- Optimize performance

**Tasks:**
```python
# Implement PSO
from src.coordination import PSO
pso = PSO(num_particles=30, dimensions=10)
pso.optimize()

# Implement GA
from src.coordination import GeneticAlgorithm
ga = GeneticAlgorithm(population_size=50)
ga.evolve(generations=100)

# Integrate real data
from src.utils import load_real_data
data = load_real_data('data/raw/charging_transactions.csv')
```

**Performance Optimization:**
- Profile code: `python -m cProfile scripts/train.py`
- Optimize bottlenecks
- Implement multi-processing
- Add GPU acceleration

**Deliverables:**
- ✓ PSO implementation working
- ✓ GA implementation working
- ✓ Real data integrated
- ✓ 2x performance improvement

---

### Week 9-10: Evaluation & Validation

**Goals:**
- Comprehensive evaluation
- Compare with baselines
- Generate publication-ready results

**Tasks:**
```bash
# Run full evaluation
python scripts/evaluate.py --model models/best_model.pth --comprehensive

# Baseline comparison
python scripts/evaluate.py --compare-baselines --save-results

# Generate visualizations
python scripts/visualize_results.py --input results/ --output figures/

# Statistical analysis
python scripts/statistical_analysis.py
```

**Evaluation Metrics:**
- Coordination success rate (target: >90%)
- Energy efficiency improvement (target: >12%)
- Cost reduction (target: >8%)
- Training stability (target: >85%)
- Convergence speed (target: <20 episodes)

**Deliverables:**
- ✓ Complete evaluation report
- ✓ Baseline comparison tables
- ✓ Publication-quality figures
- ✓ Statistical significance tests

---

### Week 11-12: Documentation & Deployment

**Goals:**
- Complete documentation
- Prepare for deployment
- Write research paper

**Tasks:**
```markdown
# Documentation
- API reference complete
- Tutorials written
- Examples documented

# Deployment
docker build -t camac-dra .
docker run -p 8000:8000 camac-dra

# Paper writing
- Methods section complete
- Results section with figures
- Discussion and conclusions
```

**Final Deliverables:**
- ✓ Complete documentation
- ✓ Docker deployment ready
- ✓ Research paper draft
- ✓ Presentation slides

---

## 📊 Results

### Performance Comparison

| Algorithm | Coord. Success (%) | Energy Eff. (%) | Cost Red. (%) | Convergence (Eps) |
|-----------|-------------------|-----------------|---------------|-------------------|
| **CAMAC-DRL** | **92.0** | **15.0** | **10.0** | **15** |
| GNN Baseline | 82.0 | 11.0 | 7.0 | 25 |
| DQN | 78.0 | 8.0 | 5.0 | 35 |
| DDPG | 71.0 | 6.0 | 4.0 | 45 |
| A3C | 75.0 | 7.0 | 6.0 | 40 |
| PPO | 69.0 | 5.0 | 3.0 | 50 |

### Key Achievements
- 🏆 **10-23% improvement** over baselines
- ⚡ **2.3× faster convergence**
- 💰 **69% cost reduction** with renewable integration
- 🌍 **Net Present Cost: -$122,962**

[View Detailed Results](docs/results.md)

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### How to Contribute

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/camac-dra.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

4. **Run tests**
   ```bash
   pytest tests/
   python -m flake8 src/
   ```

5. **Submit pull request**

### Development Guidelines
- Follow PEP 8 style guide
- Write unit tests for new features
- Update documentation
- Add docstrings to all functions

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{sharif2025camac,
  title={Context-Aware Multi-Agent Coordination Framework for Intelligent Electric Vehicle Charging Optimization: A Deep Reinforcement Learning Approach},
  author={Sharif, Muddsair and Seker, Huseyin and Javed, Yasir},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Muddsair Sharif**  
📧 Email: muddsair.sharif@hft-stuttgart.de  
🏢 Stuttgart University of Applied Sciences  
🔗 [LinkedIn](your-linkedin) | [Google Scholar](your-scholar) | [ResearchGate](your-researchgate)

**Project Link:** [https://github.com/yourusername/camac-dra](https://github.com/yourusername/camac-dra)

---

## 🌟 Acknowledgments

- Stuttgart University of Applied Sciences (HfT Stuttgart)
- Birmingham City University
- University of Sharjah
- All contributors and supporters

---

## 📈 Project Statistics

![GitHub repo size](https://img.shields.io/github/repo-size/yourusername/camac-dra)
![GitHub contributors](https://img.shields.io/github/contributors/yourusername/camac-dra)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/camac-dra)
![GitHub issues](https://img.shields.io/github/issues/yourusername/camac-dra)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/camac-dra)

---

**⭐ If you find this project useful, please consider giving it a star!**

---

*Last updated: December 2025*# CAMAC-DRA
