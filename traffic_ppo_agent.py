# traffic_ppo_agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
from typing import Dict, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(ActorCritic, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state: torch.Tensor):
        x = self.feature(state)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

class PPOBuffer:
    def __init__(self, buffer_size: int, state_dim: int):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.buffer_size = buffer_size
        self.state_dim = state_dim

    def add(self, state, action, logprob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def get(self):
        # Convert to tensors
        return (
            torch.tensor(np.array(self.states), dtype=torch.float32, device=device),
            torch.tensor(self.actions, dtype=torch.long, device=device),
            torch.tensor(self.logprobs, dtype=torch.float32, device=device),
            torch.tensor(self.rewards, dtype=torch.float32, device=device),
            torch.tensor(self.dones, dtype=torch.float32, device=device),
            torch.tensor(self.values, dtype=torch.float32, device=device)
        )

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

class TrafficPPOAgent:
    def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = {
            'hidden_dim': 128,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'lr': 3e-4,
            'batch_size': 64,
            'ppo_epochs': 4,
            'buffer_size': 2048,
            'max_grad_norm': 1.0
        }
        if config:
            self.config.update(config)

        self.model = ActorCritic(state_dim, action_dim, self.config['hidden_dim']).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        self.buffer = PPOBuffer(self.config['buffer_size'], state_dim)
        self.loss_history = []

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> Tuple[int, float, float]:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        logits, value = self.model(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        if eval_mode:
            action = torch.argmax(logits, dim=-1)
            logprob = dist.log_prob(action)
            return action.item(), logprob.item(), value.item()
        else:
            action = dist.sample()
            logprob = dist.log_prob(action)
            return action.item(), logprob.item(), value.item()

    def compute_gae(self, rewards, dones, values, next_value):
        gae = 0
        returns = []
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.config['gamma'] * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.config['gamma'] * self.config['gae_lambda'] * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self, next_state):
        states, actions, old_logprobs, rewards, dones, values = self.buffer.get()
        with torch.no_grad():
            _, next_value = self.model(torch.from_numpy(next_state).float().unsqueeze(0).to(device))
            next_value = next_value.item()
        returns = self.compute_gae(rewards.tolist(), dones.tolist(), values.tolist(), next_value)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages = returns - values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.config['ppo_epochs']):
            idxs = np.arange(len(states))
            np.random.shuffle(idxs)
            for start in range(0, len(states), self.config['batch_size']):
                end = start + self.config['batch_size']
                batch_idx = idxs[start:end]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_logprobs = old_logprobs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                logits, values_pred = self.model(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logprobs - batch_old_logprobs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config['clip_epsilon'], 1.0 + self.config['clip_epsilon']) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_pred.squeeze(-1), batch_returns)
                loss = policy_loss + self.config['value_coef'] * value_loss - self.config['entropy_coef'] * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()
                self.loss_history.append(loss.item())
        self.buffer.clear()

    def save(self, filepath: str):
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        self.model.load_state_dict(torch.load(filepath, map_location=device))
        print(f"Model loaded from {filepath}")

    def plot_training_metrics(self, save_path=None):
        if not self.loss_history:
            print("No training data to plot")
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.title('PPO Loss during Training')
        plt.xlabel('Update Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.show()
