#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from pems_traffic_env import PeMSTrafficEnv
from traffic_dqn_agent import TrafficDQNAgent

def train(data_path, episodes=100, max_steps=288, weights_dir="weights"):
    """Train a DQN agent on PeMS traffic data"""
    # Create directories
    Path(weights_dir).mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    # Initialize environment
    env = PeMSTrafficEnv(data_path, time_window=5, max_steps=max_steps)
    
    # Debug the state shape
    initial_state = env.reset()
    print(f"Initial state shape: {initial_state.shape}")
    
    # Initialize agent with the EXACT state dimension from the environment
    state_dim = initial_state.shape[0]
    agent = TrafficDQNAgent(
        state_dim=state_dim,  # Use the actual state dimension
        action_dim=env.action_space,
        config={
            'buffer_size': 50000,  # Reduced for faster training
            'batch_size': 32,      # Smaller batch for stability
            'gamma': 0.99,
            'eps_start': 1.0,
            'eps_end': 0.01,
            'eps_decay': 0.995,
            'lr': 3e-4
        }
    )
    
    print(f"Agent initialized with state_dim={state_dim}, action_dim={env.action_space}")
    
    # Training loop
    rewards = []
    for episode in range(1, episodes+1):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience and learn
            agent.step(state, action, reward, next_state, done)
            
            # Update state and stats
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Track progress
        rewards.append(episode_reward)
        avg_reward = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
        
        # Print status
        print(f"Episode {episode}/{episodes} | Reward: {episode_reward:.2f} | "
              f"Avg (10): {avg_reward:.2f} | Epsilon: {agent.eps:.4f}")
        
        # Save model periodically
        if episode % 10 == 0 or episode == episodes:
            torch.save(agent.qnetwork_local.state_dict(), f"{weights_dir}/dqn_traffic_{episode}.pth")
    
    # Plot learning curve
    # Plot learning curve and save it
    plt.figure(figsize=(12, 10))

    # Plot episode rewards
    plt.subplot(2, 1, 1)
    plt.plot(rewards, marker='o', linestyle='-', alpha=0.7)
    plt.title(f"Training Rewards over {episodes} Episodes", fontsize=14)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.grid(True)

    # Plot running average (smoother curve)
    running_avg = [np.mean(rewards[max(0, i-9):i+1]) for i in range(len(rewards))]
    plt.plot(running_avg, 'r-', linewidth=2, label='10-Episode Average')
    plt.legend()

    # Plot rewards by district
    if len(rewards) > 5:  # Only if we have enough episodes
        plt.subplot(2, 1, 2)
        
        # We need to track which episodes were for which districts
        # This needs to be collected during training, so add to the main loop:
        # districts = []  # Add this at start of train function
        # districts.append(env.district_ids[env.current_file_idx])  # Add inside episode loop
        
        # For now, just show a placeholder
        plt.text(0.5, 0.5, "District-specific performance will be shown here\nwhen tracking is added to the code", 
                ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
        plt.title("Performance by District", fontsize=14)
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Average Reward", fontsize=12)
        
        plt.tight_layout()
        plt.savefig("results/learning_curve.png", dpi=300)  # Higher DPI for better quality
        print(f"Learning curve saved to results/learning_curve.png")

    return agent

def evaluate(agent, data_path, episodes=10, max_steps=288, weights_path=None):
    """Evaluate a trained agent"""
    # Initialize environment
    env = PeMSTrafficEnv(data_path, time_window=5, max_steps=max_steps)
    
    # Create a new agent if needed
    if agent is None:
        # Get state dimension from the environment
        initial_state = env.reset()
        state_dim = initial_state.shape[0]
        agent = TrafficDQNAgent(state_dim, env.action_space)
    
    # Load weights if path provided
    if weights_path:
        try:
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            agent.qnetwork_local.load_state_dict(state_dict)
            agent.qnetwork_target.load_state_dict(state_dict)
            print(f"Loaded weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            return []
    
    # Evaluation loop
    rewards = []
    for episode in range(1, episodes+1):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action (no exploration)
            action = agent.act(state, eval_mode=True)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update state and stats
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Save metrics
        rewards.append(episode_reward)
        print(f"Evaluation Episode {episode}/{episodes} | Reward: {episode_reward:.2f}")
        
        # Plot performance for this episode
        env.plot_metrics(save_path=f"results/eval_episode_{episode}.png")
    
    # Overall evaluation results
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    
    return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PeMS Traffic Control with DQN")
    parser.add_argument("--data", required=True, help="Path to PeMS data directory")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="train", 
                        help="Operation mode: train, eval, or both")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--weights", type=str, help="Path to weights file for evaluation")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode (default: 100)")
    
    args = parser.parse_args()
    
    # Print your Python and PyTorch versions for debugging
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Running with data from: {args.data}")
    
    if args.mode in ["train", "both"]:
        print(f"\n{'='*50}")
        print(f"Starting training for {args.episodes} episodes")
        print(f"{'='*50}")
        agent = train(args.data, episodes=args.episodes, max_steps=args.steps)
        
        if args.mode == "both":
            print(f"\n{'='*50}")
            print(f"Starting evaluation")
            print(f"{'='*50}")
            evaluate(agent, args.data, episodes=args.eval_episodes, max_steps=args.steps)
    
    elif args.mode == "eval":
        if not args.weights:
            print("Error: Must provide weights file for evaluation")
            exit(1)
            
        print(f"\n{'='*50}")
        print(f"Starting evaluation using weights from {args.weights}")
        print(f"{'='*50}")
        
        # Create agent for evaluation
        evaluate(None, args.data, episodes=args.eval_episodes, 
                 max_steps=args.steps, weights_path=args.weights)