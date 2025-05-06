# Federated Learning for Smart Traffic Lights

Modern urban traffic management heavily relies on outdated centralized systems with fixed timers or centralized AI models requiring extensive data collection. These approaches struggle to adapt to dynamic, region-specific traffic conditions, leading to inefficiencies, privacy concerns, and scalability challenges. This project addresses these limitations by leveraging Reinforcement Learning (RL), specifically Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO), to optimize local traffic signals using real-time traffic data from the PeMS dataset. Our results demonstrate significant improvements in traffic flow, reduced queue lengths, and enhanced vehicle speeds. Building on these RL advancements, we propose integrating our successful local models into a Federated Learning (FL) infrastructure. This decentralized approach enables collaborative global model training across multiple urban intersections without exchanging sensitive data, thereby providing an adaptive, scalable, and privacy-preserving traffic management solution.

## Overview

Our approach:
- Uses PeMS district traffic data (flow, occupancy, speed) to simulate traffic conditions
- Implements a Dueling DQN agent with experience replay and double Q-learning
- Supports training across multiple districts for improved generalization
- Lays groundwork for federated learning in traffic control systems

## Setup

1. Install the required packages:
   ```bash
   pip install torch numpy matplotlib pandas
   ```

2. Make sure you have traffic data in the PeMS format (CSV/CSV.GZ files)

## Usage

### Training a new model

To train a new DQN agent on PeMS data:
```bash
python3 main_pems_dqn.py --data path/to/pems/data --mode train --episodes 50 --steps 200
```

### Evaluating a trained model

To evaluate a trained model:
```bash
python3 main_pems_dqn.py --data path/to/pems/data --mode eval --weights weights/dqn_traffic_50.pth --eval_episodes 5
```

### Training and evaluating in one run

To both train and evaluate:
```bash
python3 main_pems_dqn.py --data path/to/pems/data --mode both --episodes 50 --eval_episodes 5 --steps 200
```

## Command Line Arguments

- `--data`: Path to PeMS data directory (required)
- `--mode`: Operation mode - "train", "eval", or "both" (default: "train")
- `--episodes`: Number of training episodes (default: 100)
- `--eval_episodes`: Number of evaluation episodes (default: 5)
- `--weights`: Path to weights file for evaluation (required for eval mode)
- `--steps`: Maximum steps per episode (default: 100)

## Output

The training process will:
1. Create a `weights` directory with saved model weights
2. Create a `results` directory with performance plots

## Example

```bash
python3 main_pems_dqn.py --data preprocessed_data/PeMS --mode train --episodes 50 --steps 200
```

## Project Structure

- `main_pems_dqn.py`: Main script for training and evaluation
- `pems_traffic_env.py`: Environment that simulates traffic control using PeMS data
- `traffic_dqn_agent.py`: Implementation of the Dueling DQN agent

## Results

The DQN agent learns to effectively control traffic signals across different districts, with rewards typically ranging between 100-400 after training. The agent quickly adapts to different traffic patterns and demonstrates good generalization capabilities.

## Future Work

- Implement federated learning by training district-specific models and aggregating knowledge
- Incorporate more complex traffic metrics and scenarios
- Optimize hyperparameters for improved performance
- Deploy the system in a real-world traffic management scenario
