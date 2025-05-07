#!/usr/bin/env python3
"""
main_pems_ppo.py

Run training and evaluation for PPO and baseline agents on PeMS traffic data.

Usage examples:
  # Train PPO only:
  python main_pems_ppo.py --data path/to/pems/data --mode train --episodes 100 --steps 288

  # Evaluate PPO and baselines on held-out data:
  python main_pems_ppo.py --data path/to/pems/data --mode eval \
      --weights weights/ppo_traffic_100.pth \
      --eval_episodes 5 --steps 288

  # Train then evaluate all (PPO + baselines) in one go:
  python main_pems_ppo.py --data path/to/pems/data --mode both \
      --episodes 100 --eval_episodes 5 --steps 288

By default, in 'eval' or 'both' modes, the script will automatically run:
  1) PPO evaluation using provided weights
  2) RandomAgent baseline
  3) FixedTimeAgent baseline

Results are saved under results/:
  learning_curve_ppo.png       # PPO training curve
  ppo_episode_*.png            # PPO per-episode metrics
  random_episode_*.png         # RandomAgent per-episode metrics
  fixedtime_episode_*.png      # FixedTimeAgent per-episode metrics
  eval_comparison.png          # Evaluation rewards comparison plot
"""
import argparse
from pathlib import Path
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import environment and agents
from pems_traffic_env import PeMSTrafficEnv
from traffic_ppo_agent import TrafficPPOAgent
from baseline_agents import RandomAgent, FixedTimeAgent


def train(data_path: str, episodes: int = 100, max_steps: int = 288, weights_dir: str = "weights"):
    """Train PPO agent and save interim weights and training curve."""
    Path(weights_dir).mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    env = PeMSTrafficEnv(data_path, time_window=5, max_steps=max_steps)
    state_dim = env.reset().shape[0]
    agent = TrafficPPOAgent(state_dim, env.action_space)

    rewards = []
    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, logprob, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.add(state, action, logprob, reward, done, value)
            state = next_state
            total_reward += reward
            if len(agent.buffer.states) >= agent.config['buffer_size']:
                agent.update(next_state)
        rewards.append(total_reward)
        print(f"Episode {ep}/{episodes} | PPO Reward: {total_reward:.2f}")
        if ep % 10 == 0 or ep == episodes:
            agent.save(f"{weights_dir}/ppo_traffic_{ep}.pth")

    # Plot and save PPO training curve
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='PPO Reward')
    running_avg = [np.mean(rewards[max(0, i-9):i+1]) for i in range(len(rewards))]
    plt.plot(running_avg, 'r--', label='10-episode avg')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO Training Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/learning_curve_ppo.png', dpi=200)
    print("Saved training curve to results/learning_curve_ppo.png")

    return agent, rewards


def evaluate(agent, data_path: str, episodes: int = 5, max_steps: int = 288,
             weights_path: str = None, label: str = "PPO"):
    """Evaluate an agent (PPO or baseline) and return per-episode rewards."""
    env = PeMSTrafficEnv(data_path, time_window=5, max_steps=max_steps)
    # Load PPO weights if needed
    if isinstance(agent, TrafficPPOAgent) and weights_path:
        agent.load(weights_path)
    rewards = []
    for ep in range(1, episodes + 1):
        state = env.reset()
        # reset baseline pointer
        if hasattr(agent, 'pointer'):
            agent.pointer = 0
        total_reward = 0.0
        done = False
        while not done:
            if isinstance(agent, TrafficPPOAgent):
                action, _, _ = agent.select_action(state, eval_mode=True)
            else:
                action, _, _ = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        print(f"{label} Eval Ep {ep}/{episodes} | Reward: {total_reward:.2f}")
        env.plot_metrics(save_path=f"results/{label.lower()}_episode_{ep}.png")

    mean_r, std_r = np.mean(rewards), np.std(rewards)
    print(f"{label} Mean ± Std: {mean_r:.2f} ± {std_r:.2f}\n")
    return rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PeMS Traffic PPO & Baselines")
    parser.add_argument('--data', required=True, help="Path to PeMS data (file or directory)")
    parser.add_argument('--mode', choices=['train', 'eval', 'both'], default='train')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--eval_episodes', type=int, default=5)
    parser.add_argument('--weights', type=str, help="PPO weights for evaluation")
    parser.add_argument('--steps', type=int, default=288)
    args = parser.parse_args()

    print(f"Python {sys.version.split()[0]} | PyTorch {torch.__version__}")
    print(f"Data={args.data} | Mode={args.mode}")

    # Train PPO if requested
    ppo_agent, ppo_train_rewards = (None, [])
    if args.mode in ['train', 'both']:
        ppo_agent, ppo_train_rewards = train(args.data, episodes=args.episodes, max_steps=args.steps)

    # Evaluate policies if requested
    if args.mode in ['eval', 'both']:
        # PPO evaluation
        print("\nEvaluating PPO policy...")
        ppo_eval_rewards = evaluate(
            ppo_agent if args.mode=='both' else TrafficPPOAgent( PeMSTrafficEnv(args.data, max_steps=args.steps).reset().shape[0],
                                                               PeMSTrafficEnv(args.data, max_steps=args.steps).action_space ),
            args.data, episodes=args.eval_episodes, max_steps=args.steps,
            weights_path=args.weights if args.mode=='eval' else None,
            label='PPO'
        )
        # Random baseline evaluation
        print("\nEvaluating Random baseline...")
        random_agent = RandomAgent(action_space=PeMSTrafficEnv(args.data, max_steps=args.steps).action_space)
        random_rewards = evaluate(random_agent, args.data, episodes=args.eval_episodes,
                                  max_steps=args.steps, label='Random')
        # Fixed-time baseline evaluation
        print("\nEvaluating Fixed-Time baseline...")
        fixed_agent = FixedTimeAgent(action_space=PeMSTrafficEnv(args.data, max_steps=args.steps).action_space)
        fixed_rewards = evaluate(fixed_agent, args.data, episodes=args.eval_episodes,
                                 max_steps=args.steps, label='FixedTime')

        # Combined evaluation comparison plot
        ppo_mean   = np.mean(ppo_eval_rewards)
        random_mean = np.mean(random_rewards)
        fixed_mean  = np.mean(fixed_rewards)

        print(f"PPO   average reward over {len(ppo_eval_rewards)} episodes:   {ppo_mean:.2f}")
        print(f"Random average reward over {len(random_rewards)} episodes: {random_mean:.2f}")
        print(f"FixedTime average reward over {len(fixed_rewards)} episodes: {fixed_mean:.2f}")

        # 2) Plot bar chart of averages
        agents = ['PPO', 'Random', 'FixedTime']
        means  = [ppo_mean, random_mean, fixed_mean]

        plt.figure(figsize=(8,6))
        plt.bar(agents, means)
        plt.ylabel('Average Total Reward')
        plt.title('Average Evaluation Reward by Agent')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('results/avg_eval_comparison.png', dpi=200)
        print("Saved average comparison to results/avg_eval_comparison.png")
