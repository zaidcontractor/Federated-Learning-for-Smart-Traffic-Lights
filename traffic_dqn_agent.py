import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from pathlib import Path
import copy
import time
from typing import List, Dict, Tuple, Optional, Union
import os

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Dueling DQN Architecture
class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture with separate value and advantage streams.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize the Dueling DQN.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
        """
        super(DuelingDQN, self).__init__()
        
        # Feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Current state tensor
            
        Returns:
            Q-values for each action
        """
        # Extract features
        features = self.feature_layer(state)
        
        # Calculate value and advantages
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages using dueling formula
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        qvals = value + advantages - advantages.mean(dim=1, keepdim=True)
        
        return qvals

# Define Experience Replay Buffer
class ReplayBuffer:
    """
    Experience replay buffer to store and sample transitions.
    """
    def __init__(self, buffer_size: int, batch_size: int):
        """
        Initialize replay buffer.
        
        Args:
            buffer_size: Maximum size of buffer
            batch_size: Size of training batches
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Add new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self) -> Tuple:
        """Randomly sample a batch of experiences from memory"""
        if len(self) < self.batch_size:
            return None
        
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # Convert to tensors and move to device
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self) -> int:
        """Return the current size of internal memory"""
        return len(self.memory)

# Traffic DQN Agent
class TrafficDQNAgent:
    """
    DQN Agent for traffic signal control using PeMS data.
    """
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 config: Dict = None):
        """
        Initialize the DQN Agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Default configuration
        self.config = {
            'buffer_size': 100000,      # Replay buffer size
            'batch_size': 64,           # Minibatch size
            'gamma': 0.99,              # Discount factor
            'tau': 0.001,               # Soft update parameter
            'lr': 3e-4,                 # Learning rate
            'update_every': 4,          # How often to update network
            'hidden_dim': 128,          # Hidden layer size
            'eps_start': 1.0,           # Starting epsilon for exploration
            'eps_end': 0.01,            # Minimum epsilon
            'eps_decay': 0.995,         # Epsilon decay rate
            'prioritized_replay': False, # Whether to use prioritized replay
            'double_dqn': True,         # Whether to use double DQN
        }
        
        # Update with custom config if provided
        if config:
            self.config.update(config)
        
        # Create Q-Networks
        self.qnetwork_local = DuelingDQN(state_dim, action_dim, self.config['hidden_dim']).to(device)
        self.qnetwork_target = DuelingDQN(state_dim, action_dim, self.config['hidden_dim']).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.config['lr'])
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(self.config['buffer_size'], self.config['batch_size'])
        
        # Initialize time step and training stats
        self.t_step = 0
        self.train_step = 0
        self.eps = self.config['eps_start']
        
        # For saving training metrics
        self.loss_history = []
        self.epsilon_history = []
    
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Store experience and learn if it's time.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Add experience to replay buffer
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.config['update_every']
        if self.t_step == 0 and len(self.memory) >= self.config['batch_size']:
            experiences = self.memory.sample()
            if experiences:
                loss = self.learn(experiences)
                self.loss_history.append(loss)
                
        # Decay epsilon
        self.train_step += 1
        self.eps = max(
            self.config['eps_end'], 
            self.config['eps_decay'] * self.eps
        )
        self.epsilon_history.append(self.eps)
    
    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Select action based on current policy.
        
        Args:
            state: Current state
            eval_mode: If True, use greedy policy (no exploration)
            
        Returns:
            Action index
        """
        # Convert state to tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Get action values
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if eval_mode or random.random() > self.eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_dim))
    
    def learn(self, experiences: Tuple) -> float:
        """
        Update value parameters using batch of experiences.
        
        Args:
            experiences: Tuple of (states, actions, rewards, next_states, dones)
            
        Returns:
            Loss value
        """
        states, actions, rewards, next_states, dones = experiences
        
        if self.config['double_dqn']:
            # Double DQN: get actions from local model
            local_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            # Get Q values from target model but for actions chosen by local model
            Q_targets_next = self.qnetwork_target(next_states).gather(1, local_actions)
        else:
            # Regular DQN: get maximum Q values from target model
            Q_targets_next = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.config['gamma'] * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        self._soft_update()
        
        return loss.item()
    
    def _soft_update(self):
        """Soft update target network parameters"""
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(
                self.config['tau'] * local_param.data + (1.0 - self.config['tau']) * target_param.data
            )
    
    def save(self, filepath: str):
        """Save model weights"""
        torch.save(self.qnetwork_local.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model weights"""
        self.qnetwork_local.load_state_dict(torch.load(filepath, map_location=device))
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        print(f"Model loaded from {filepath}")
    
    def plot_training_metrics(self, save_path=None):
        """Plot training metrics"""
        if not self.loss_history:
            print("No training data to plot")
            return
            
        steps = range(len(self.loss_history))
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(steps, self.loss_history)
        plt.title('Loss during Training')
        plt.xlabel('Learning Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.epsilon_history)), self.epsilon_history)
        plt.title('Epsilon during Training')
        plt.xlabel('Training Steps')
        plt.ylabel('Epsilon')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()