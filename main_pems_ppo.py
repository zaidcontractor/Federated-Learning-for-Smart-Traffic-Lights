#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from pems_traffic_env import PeMSTrafficEnv
from traffic_ppo_agent import TrafficPPOAgent

def train(data_path, episodes=100, max_steps=288, weights_dir="weights"):
    Path(weights_dir).mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    env = PeMSTrafficEnv(data_path, time_window=5, max_steps=max_steps)
    initial_state = env.reset()
    print(f"Initial state shape: {initial_state.shape}")

    state_dim = initial_state.shape[0]
    agent = TrafficPPOAgent(
        state_dim=state_dim,
        action_dim=env.action_space,
        config={
            'buffer_size': 2048,
            'batch_size': 64,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'lr': 3e-4,
            'ppo_epochs': 4,
            'max_grad_norm': 1.0
        }
    )

    print(f"Agent initialized with state_dim={state_dim}, action_dim={env.action_space}")

    rewards = []
    districts = []
    district_rewards = {}

    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0

        current_district = env.district_ids[env.current_file_idx]
        districts.append(current_district)
        if current_district not in district_rewards:
            district_rewards[current_district] = []

        for step in range(max_steps):
            action, logprob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.buffer.add(state, action, logprob, reward, done, value)
            state = next_state
            episode_reward += reward

            if len(agent.buffer.states) >= agent.config['buffer_size']:
                agent.update(next_state)

            if done:
                break

        rewards.append(episode_reward)
        district_rewards[current_district].append(episode_reward)
        avg_reward = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)

        print(f"Episode {episode}/{episodes} | District: {current_district} | "
              f"Reward: {episode_reward:.2f} | Avg (10): {avg_reward:.2f}")

        if episode % 10 == 0 or episode == episodes:
            agent.save(f"{weights_dir}/ppo_traffic_{episode}.pth")

    plt.figure(figsize=(12, 10))
    plt.plot(rewards, marker='o', linestyle='-', alpha=0.7, label='Reward for Each Episode')
    plt.title(f"Training Rewards over {episodes} Episodes - PPO", fontsize=14)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.grid(True)
    running_avg = [np.mean(rewards[max(0, i-9):i+1]) for i in range(len(rewards))]
    plt.plot(running_avg, 'r-', linewidth=2, label='10-Episode Average')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/learning_curve_ppo.png", dpi=300)
    print(f"Learning curve saved to results/learning_curve_ppo.png")

    return agent

def evaluate(agent, data_path, episodes=10, max_steps=288, weights_path=None):
    env = PeMSTrafficEnv(data_path, time_window=5, max_steps=max_steps)

    if agent is None:
        initial_state = env.reset()
        state_dim = initial_state.shape[0]
        agent = TrafficPPOAgent(state_dim, env.action_space)
    if weights_path:
        try:
            agent.load(weights_path)
        except Exception as e:
            print(f"Error loading weights: {e}")
            return []

    rewards = []
    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action, logprob, value = agent.select_action(state, eval_mode=True)
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward)
        print(f"Evaluation Episode {episode}/{episodes} | Reward: {episode_reward:.2f}")
        env.plot_metrics(save_path=f"results/eval_ppo_episode_{episode}.png")

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")

    return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PeMS Traffic Control with PPO")
    parser.add_argument("--data", required=True, help="Path to PeMS data directory")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="train",
                        help="Operation mode: train, eval, or both")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--weights", type=str, help="Path to weights file for evaluation")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode (default: 100)")

    args = parser.parse_args()

    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Running with data from: {args.data}")

    if args.mode in ["train", "both"]:
        print(f"\n{'='*50}")
        print(f"Starting PPO training for {args.episodes} episodes")
        print(f"{'='*50}")
        agent = train(args.data, episodes=args.episodes, max_steps=args.steps)

        if args.mode == "both":
            print(f"\n{'='*50}")
            print(f"Starting PPO evaluation")
            print(f"{'='*50}")
            evaluate(agent, args.data, episodes=args.eval_episodes, max_steps=args.steps)

    elif args.mode == "eval":
        if not args.weights:
            print("Error: Must provide weights file for evaluation")
            exit(1)

        print(f"\n{'='*50}")
        print(f"Starting PPO evaluation using weights from {args.weights}")
        print(f"{'='*50}")

        evaluate(None, args.data, episodes=args.eval_episodes,
                 max_steps=args.steps, weights_path=args.weights)
