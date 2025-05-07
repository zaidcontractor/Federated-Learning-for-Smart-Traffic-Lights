import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Any

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
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

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.feature(state)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

class PPOBuffer:
    def __init__(self, buffer_size: int):
        self.states, self.actions, self.logprobs = [], [], []
        self.rewards, self.dones, self.values = [], [], []
        self.buffer_size = buffer_size

    def add(self, state: Any, action: int, logprob: float, reward: float, done: bool, value: float):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def get(self):
        return (
            torch.tensor(np.array(self.states), dtype=torch.float32, device=device),
            torch.tensor(self.actions, dtype=torch.long, device=device),
            torch.tensor(self.logprobs, dtype=torch.float32, device=device),
            torch.tensor(self.rewards, dtype=torch.float32, device=device),
            torch.tensor(self.dones, dtype=torch.float32, device=device),
            torch.tensor(self.values, dtype=torch.float32, device=device)
        )

    def clear(self):
        self.states, self.actions, self.logprobs = [], [], []
        self.rewards, self.dones, self.values = [], [], []

class TrafficPPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict] = None,
        weights_path: Optional[str] = None
    ):
        # Hyperparameters and configuration
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

        # Create model, optimizer, and buffer
        self.model = ActorCritic(state_dim, action_dim, self.config['hidden_dim']).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        self.buffer = PPOBuffer(self.config['buffer_size'])
        self.loss_history = []

        # Optionally load pretrained weights
        if weights_path:
            self.load(weights_path)

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> Tuple[int, float, float]:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        logits, value = self.model(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        if eval_mode:
            action = torch.argmax(logits, dim=-1)
            logprob = dist.log_prob(action)
            return action.item(), logprob.item(), value.item()
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.item(), logprob.item(), value.item()

    def compute_gae(self, rewards, dones, values, next_value):
        gae, returns = 0, []
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.config['gamma'] * values[step+1] * (1 - dones[step]) - values[step]
            gae = delta + self.config['gamma'] * self.config['gae_lambda'] * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self, next_state: np.ndarray) -> None:
        states, actions, old_logprobs, rewards, dones, values = self.buffer.get()
        with torch.no_grad():
            _, next_value = self.model(
                torch.from_numpy(next_state).float().unsqueeze(0).to(device)
            )
            next_value = next_value.item()
        returns = torch.tensor(
            self.compute_gae(rewards.tolist(), dones.tolist(), values.tolist(), next_value),
            dtype=torch.float32,
            device=device
        )
        advantages = (returns - values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.config['ppo_epochs']):
            idxs = np.arange(len(states))
            np.random.shuffle(idxs)
            for start in range(0, len(states), self.config['batch_size']):
                batch_idxs = idxs[start:start + self.config['batch_size']]
                bs = states[batch_idxs]
                ba = actions[batch_idxs]
                bl = old_logprobs[batch_idxs]
                br = returns[batch_idxs]
                bd = dones[batch_idxs]
                adv = advantages[batch_idxs]

                logits, values_pred = self.model(bs)
                dist = torch.distributions.Categorical(logits=logits)
                logprobs = dist.log_prob(ba)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logprobs - bl)
                surr = torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1 - self.config['clip_epsilon'], 1 + self.config['clip_epsilon']) * adv
                )
                policy_loss = -surr.mean()
                value_loss = F.mse_loss(values_pred.squeeze(-1), br)
                loss = policy_loss + self.config['value_coef'] * value_loss - self.config['entropy_coef'] * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()
                self.loss_history.append(loss.item())

        self.buffer.clear()

    def save(self, filepath: str) -> None:
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        self.model.load_state_dict(torch.load(filepath, map_location=device))
        print(f"Model loaded from {filepath}")

    def act(self, state: np.ndarray) -> int:
        """Deterministic action in eval mode."""
        self.model.eval()
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            logits, _ = self.model(state_t)
            return int(torch.argmax(logits, dim=-1).item())

    def evaluate(
        self,
        env,
        n_episodes: int = 10,
        max_steps: Optional[int] = None,
        render: bool = False,
        weights_path: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate policy: load weights, run rollouts, return metrics."""
        if weights_path:
            self.load(weights_path)
        self.model.eval()
        returns = []
        for _ in range(n_episodes):
            state, done, total_r, steps = env.reset(), False, 0.0, 0
            while not done and (max_steps is None or steps < max_steps):
                action = self.act(state)
                state, reward, done, _ = env.step(action)
                total_r += reward
                steps += 1
                if render:
                    env.render()
            returns.append(total_r)
        mean_r, std_r = float(np.mean(returns)), float(np.std(returns))
        print(f"Eval over {n_episodes} eps: mean={mean_r:.2f} ± {std_r:.2f}")
        return {"mean_return": mean_r, "std_return": std_r}

# Example usage:
# agent = TrafficPPOAgent(state_dim, action_dim, config, weights_path='weights/ppo_traffic_100.pth')
# stats = agent.evaluate(env=validation_env, n_episodes=20)
