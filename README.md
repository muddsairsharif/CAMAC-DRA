# CAMAC-DRA: Context-Aware Multi-Agent Coordination with Deep Reinforcement Learning for EV Charging

## Overview

CAMAC-DRA is a sophisticated multi-agent deep reinforcement learning framework designed for electric vehicle (EV) charging coordination. It combines graph neural networks, attention mechanisms, and multi-agent coordination to optimize charging schedules while managing grid constraints.

## Features

- **Multi-Agent RL**: Coordinated learning for multiple EV agents
- **Graph Neural Networks**: Context-aware representations of charging infrastructure
- **Attention Mechanisms**: Dynamic priority and resource allocation
- **Advanced Coordination**: PSO and GA-based optimization
- **Realistic Simulation**: Grid simulator with dynamic pricing and constraints
- **Comprehensive Analysis**: Metrics, visualization, and logging tools

## Quick Start

### Installation

```bash
git clone https://github.com/muddsairsharif/CAMAC-DRA.git
cd CAMAC-DRA
pip install -r requirements.txt
python setup.py install
```

### Training

```bash
python scripts/train.py --config config/training.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint models/checkpoints/best_model.pth
```

## Project Structure

```
CAMAC-DRA/
├── config/                 # Configuration files
├── src/                    # Source code
│   ├── models/            # Neural network models
│   ├── environment/       # EV charging environment
│   ├── training/          # Training pipeline
│   ├── coordination/      # Multi-agent coordination
│   └── utils/             # Utility functions
├── scripts/                # Executable scripts
├── notebooks/              # Jupyter notebooks
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── data/                   # Data storage
├── models/                 # Model checkpoints
├── results/                # Results and outputs
└── experiments/            # Experimental runs
```

## Documentation

- [Architecture](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Results](docs/results.md)
- [Weekly Guide](docs/weekly_guide.md)
- [Troubleshooting](docs/troubleshooting.md)

## License

MIT License - See LICENSE file for details

## Contact

Mudd Sair Sharif - [GitHub](https://github.com/muddsairsharif)