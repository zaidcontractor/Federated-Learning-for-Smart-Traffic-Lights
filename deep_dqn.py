###############################################
# deep_dqn.py – Drop‑in Deep Q‑Network module #
###############################################
from __future__ import annotations

import random
from collections import deque, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import trange

# ONFIG DATA‑CLASS
@dataclass
class DQNConfig:
    state_dim: int
    action_dim: int
    buffer_size: int = 100_000
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    tau: float = 5e-3          # soft target update rate
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 200_000   # steps over which eps decays
    device: str = "mps"        # "cuda", "mps", or "cpu"
    seed: int = 42


# 2.  NETWORK                                
class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.val = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_dim))

    def forward(self, x):
        x = self.backbone(x)
        v, a = self.val(x), self.adv(x)
        return v + a - a.mean(1, keepdim=True)


# 3.  REPLAY BUFFER                         
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.mem = deque(maxlen=capacity)

    def push(self, *exp):
        self.mem.append(Experience(*exp))

    def sample(self, k: int):
        batch = random.sample(self.mem, k)
        return Experience(*zip(*batch))

    def __len__(self):
        return len(self.mem)


# 4.  AGENT                                  
class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.cfg = cfg
        self.device = torch.device(
            cfg.device if (torch.cuda.is_available() or torch.backends.mps.is_available()) else "cpu"
        )

        self.q = DuelingDQN(cfg.state_dim, cfg.action_dim).to(self.device)
        self.q_target = DuelingDQN(cfg.state_dim, cfg.action_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.opt = AdamW(self.q.parameters(), lr=cfg.lr, weight_decay=1e-4)
        self.buf = ReplayBuffer(cfg.buffer_size)

        self.steps_done = 0
        self.eps = cfg.eps_start

    # ε‑greedy policy ----------------------------------------------------
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.eps:
            return random.randrange(self.cfg.action_dim)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q(state_t).argmax(1).item())

    # single optimisation step ------------------------------------------
    def learn(self):
        if len(self.buf) < self.cfg.batch_size:
            return
        batch = self.buf.sample(self.cfg.batch_size)

        # ❶  fast, vectorised tensor construction (kills the warning)
        s  = torch.from_numpy(np.stack(batch.state)).float().to(self.device)
        a  = torch.tensor(batch.action,  dtype=torch.int64,  device=self.device).unsqueeze(1)
        r  = torch.tensor(batch.reward,  dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.from_numpy(np.stack(batch.next_state)).float().to(self.device)
        d  = torch.tensor(batch.done,    dtype=torch.float32, device=self.device).unsqueeze(1)

        # ❷  standard DQN update
        q_sa = self.q(s).gather(1, a)
        with torch.no_grad():
            a2 = self.q(s2).argmax(1, keepdim=True)
            q_s2a2 = self.q_target(s2).gather(1, a2)
            target = r + self.cfg.gamma * q_s2a2 * (1 - d)

        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10)
        self.opt.step()

        # soft update
        for tp, p in zip(self.q_target.parameters(), self.q.parameters()):
            tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

    def update_eps(self):
        self.steps_done += 1
        frac = min(1.0, self.steps_done / self.cfg.eps_decay)
        self.eps = self.cfg.eps_start - frac * (self.cfg.eps_start - self.cfg.eps_end)


# 5.  TRAINING LOOP                          

def train_dqn(csv_path: str | Path, env_name: str = "sumo_rl", episodes: int = 500, device: str = "cpu", weight_file: Optional[str] = None):
    """High‑level training helper.

    Args:
        csv_path: Path to a pre‑processed PeMS file **or directory**.
        env_name: Module that provides ``make_demand_env``.
        episodes: Number of episodes.
        device:   "cpu", "cuda", or "mps".
        weight_file: Optional path to save model weights.
    """
    import importlib
    env_mod = importlib.import_module(env_name)
    env = env_mod.make_demand_env(csv_path)

    # --- derive dims dynamically --------------------------------------
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    cfg = DQNConfig(state_dim=state_dim, action_dim=action_dim, device=device)
    agent = DQNAgent(cfg)

    for ep in trange(episodes, desc="episodes", unit="ep"):
        s, _ = env.reset()
        done = False; ep_ret = 0
        while not done:
            a = agent.select_action(s)
            s2, r, done, _, _ = env.step(a)
            agent.buf.push(s, a, r, s2, done)
            s, ep_ret = s2, ep_ret + r
            agent.learn(); agent.update_eps()
        if ep % 10 == 0:
            print(f"Episode {ep}: return={ep_ret:.1f}, eps={agent.eps:.2f}")

    if weight_file:
        Path(weight_file).parent.mkdir(parents=True, exist_ok=True)
        torch.save(agent.q.state_dict(), weight_file)
        print(f"Weights saved to {weight_file}")


# 6.  CLI ENTRY‑POINT                        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--env", default="sumo_env")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save", default="weights/local_tl.pt")
    args = parser.parse_args()

    train_dqn(args.csv, env_name=args.env, episodes=args.episodes, device=args.device, weight_file=args.save)