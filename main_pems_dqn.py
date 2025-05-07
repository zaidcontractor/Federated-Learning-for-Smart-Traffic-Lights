#!/usr/bin/env python3
# main_pems_dqn.py

import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

from pems_traffic_env import PeMSTrafficEnv
from traffic_dqn_agent import TrafficDQNAgent
from baseline_agents import RandomAgent, FixedTimeAgent


def train(data_path, episodes=100, max_steps=288, weights_dir="weights"):
    """Train a DQN agent on PeMS traffic data."""
    Path(weights_dir).mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    env = PeMSTrafficEnv(data_path, time_window=5, max_steps=max_steps)
    initial_state = env.reset()
    print(f"Initial state shape: {initial_state.shape}")
    
    state_dim = initial_state.shape[0]
    agent = TrafficDQNAgent(
        state_dim=state_dim,
        action_dim=env.action_space,
        config={
            'buffer_size': 50000,
            'batch_size': 32,
            'gamma': 0.99,
            'eps_start': 1.0,
            'eps_end': 0.01,
            'eps_decay': 0.995,
            'lr': 3e-4
        }
    )
    print(f"Agent initialized with state_dim={state_dim}, action_dim={env.action_space}")
    
    rewards = []
    for episode in range(1, episodes+1):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                break
        
        rewards.append(episode_reward)
        avg_reward = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
        print(f"Episode {episode}/{episodes} | Reward: {episode_reward:.2f} | Avg(10): {avg_reward:.2f} | Epsilon: {agent.eps:.4f}")
        
        if episode % 10 == 0 or episode == episodes:
            torch.save(agent.qnetwork_local.state_dict(), f"{weights_dir}/dqn_traffic_{episode}.pth")
    
    # Plot training curve
    plt.figure(figsize=(12, 8))
    plt.plot(rewards, marker='o', alpha=0.7, label='Episode Reward')
    running_avg = [np.mean(rewards[max(0, i-9):i+1]) for i in range(len(rewards))]
    plt.plot(running_avg, 'r-', linewidth=2, label='10-Episode Avg')
    plt.title(f"DQN Training Rewards over {episodes} Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/learning_curve_dqn.png", dpi=300)
    print("Saved training curve to results/learning_curve_dqn.png")
    
    return agent


def evaluate(agent, data_path, episodes=5, max_steps=288, weights_path=None, label="DQN"):
    """Evaluate a trained agent (DQN or baseline) and return per-episode rewards."""
    env = PeMSTrafficEnv(data_path, time_window=5, max_steps=max_steps)
    
    if agent is None:
        init_state = env.reset()
        state_dim  = init_state.shape[0]
        agent = TrafficDQNAgent(state_dim, env.action_space)
        if weights_path:
            agent.load(weights_path)
    
    if isinstance(agent, TrafficDQNAgent):
        if weights_path:
            state_dict = torch.load(weights_path, map_location='cpu')
            agent.qnetwork_local.load_state_dict(state_dict)
            agent.qnetwork_target.load_state_dict(state_dict)
            print(f"{label}: Loaded weights from {weights_path}")
    else:
        # for RandomAgent or FixedTimeAgent
        pass
    
    rewards = []
    for ep in range(1, episodes+1):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if isinstance(agent, TrafficDQNAgent):
                action = agent.act(state, eval_mode=True)
            else:
                action, _ , _ = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        
        rewards.append(total_reward)
        print(f"{label} Eval Ep {ep}/{episodes} | Reward: {total_reward:.2f}")
        env.plot_metrics(save_path=f"results/{label.lower()}_episode_{ep}.png")
    
    mean_r = np.mean(rewards)
    std_r  = np.std(rewards)
    print(f"{label} Mean ± Std over {episodes} episodes: {mean_r:.2f} ± {std_r:.2f}\n")
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PeMS Traffic Control with DQN")
    parser.add_argument("--data", required=True, help="Path to PeMS data directory")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="train")
    parser.add_argument("--episodes", type=int, default=100, help="Training episodes")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Evaluation episodes")
    parser.add_argument("--weights", type=str, help="Path to DQN weights for evaluation")
    parser.add_argument("--steps", type=int, default=288, help="Max steps per episode")
    args = parser.parse_args()
    
    print(f"Python {sys.version.split()[0]} | PyTorch {torch.__version__}")
    print(f"Data={args.data} | Mode={args.mode}")
    
    # Train if requested
    dqn_agent = None
    if args.mode in ["train", "both"]:
        dqn_agent = train(args.data, episodes=args.episodes, max_steps=args.steps)
    
    # Evaluate if requested
    if args.mode in ["eval", "both"]:
        # DQN evaluation
        print("\nEvaluating DQN policy...")
        dqn_rewards = evaluate(
            dqn_agent if args.mode=="both" else None,
            args.data,
            episodes=args.eval_episodes,
            max_steps=args.steps,
            weights_path=args.weights,
            label="DQN"
        )
        
        # Random baseline
        print("Evaluating Random baseline...")
        random_agent = RandomAgent(action_space=PeMSTrafficEnv(args.data, max_steps=args.steps).action_space)
        random_rewards = evaluate(random_agent, args.data, episodes=args.eval_episodes,
                                  max_steps=args.steps, label="Random")
        
        # Fixed-Time baseline
        print("Evaluating FixedTime baseline...")
        fixed_agent = FixedTimeAgent(action_space=PeMSTrafficEnv(args.data, max_steps=args.steps).action_space)
        fixed_rewards = evaluate(fixed_agent, args.data, episodes=args.eval_episodes,
                                 max_steps=args.steps, label="FixedTime")
        
        # Plot comparison over episodes
        idx = np.arange(1, args.eval_episodes+1)
        plt.figure(figsize=(10,6))
        plt.plot(idx, dqn_rewards,    marker='o', label='DQN')
        plt.plot(idx, random_rewards, marker='s', label='Random')
        plt.plot(idx, fixed_rewards,  marker='^', label='FixedTime')
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Evaluation Rewards Comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("results/dqn_eval_comparison.png", dpi=200)
        print("Saved episode-by-episode comparison to results/dqn_eval_comparison.png")
        
        # Compute and print averages
        dqn_mean    = np.mean(dqn_rewards)
        random_mean = np.mean(random_rewards)
        fixed_mean  = np.mean(fixed_rewards)
        print(f"DQN   avg over {len(dqn_rewards)} eps:   {dqn_mean:.2f}")
        print(f"Random avg over {len(random_rewards)} eps: {random_mean:.2f}")
        print(f"FixedTime avg over {len(fixed_rewards)} eps: {fixed_mean:.2f}")
        
        # Bar chart of averages
        agents = ["DQN", "Random", "FixedTime"]
        means  = [dqn_mean, random_mean, fixed_mean]
        plt.figure(figsize=(8,6))
        plt.bar(agents, means)
        plt.ylabel("Average Total Reward")
        plt.title("Average Evaluation Reward by Agent")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig("results/dqn_avg_eval_comparison.png", dpi=200)
        print("Saved average comparison to results/dqn_avg_eval_comparison.png")
