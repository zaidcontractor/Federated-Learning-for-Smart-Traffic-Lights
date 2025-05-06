#!/usr/bin/env python3
"""
Simple Federated Learning orchestration for your PeMS-based traffic control agents.
Supports both DQN and PPO with FedAvg and FedProx, or loading a pretrained global model.
Produces comparison plots (local vs. global) in `results/`.
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pems_traffic_env import PeMSTrafficEnv
from traffic_dqn_agent import TrafficDQNAgent
from traffic_ppo_agent import TrafficPPOAgent
from fl_utils import fed_avg, fed_prox


def local_train_dqn(data_path, global_state, episodes, max_steps):
    env = PeMSTrafficEnv(data_path, time_window=5, max_steps=max_steps)
    state = env.reset()
    state_dim = state.shape[0]
    agent = TrafficDQNAgent(
        state_dim, env.action_space,
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
    if global_state is not None:
        agent.qnetwork_local.load_state_dict(global_state)
        agent.qnetwork_target.load_state_dict(global_state)

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
    return agent.qnetwork_local.state_dict()


def local_train_ppo(data_path, global_state, episodes, max_steps):
    env = PeMSTrafficEnv(data_path, time_window=5, max_steps=max_steps)
    state = env.reset()
    state_dim = state.shape[0]
    action_dim = env.action_space
    agent = TrafficPPOAgent(
        state_dim, action_dim,
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
    if global_state is not None:
        agent.model.load_state_dict(global_state)

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action, logp, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.add(state, action, logp, reward, done, value)
            state = next_state
            if len(agent.buffer.states) >= agent.config['buffer_size']:
                agent.update(next_state)
    return agent.model.state_dict()


def evaluate_model(algo, state_dict, client_dirs, eval_episodes, max_steps):
    results = {}
    for path in client_dirs:
        env = PeMSTrafficEnv(path, time_window=5, max_steps=max_steps)
        rewards = []
        if algo == 'dqn':
            state = env.reset()
            agent = TrafficDQNAgent(
                state.shape[0], env.action_space,
                config={'buffer_size':1,'batch_size':1,'gamma':1.0,'eps_start':0.0,'eps_end':0.0,'eps_decay':1.0,'lr':1e-4}
            )
            if state_dict is not None:
                agent.qnetwork_local.load_state_dict(state_dict)
            for _ in range(eval_episodes):
                state = env.reset()
                total = 0
                done = False
                while not done:
                    action = agent.act(state, eval_mode=True)
                    state, reward, done, _ = env.step(action)
                    total += reward
                rewards.append(total)
        else:
            state = env.reset()
            agent = TrafficPPOAgent(
                state.shape[0], env.action_space,
                config={'buffer_size':1,'batch_size':1,'gamma':1.0,'gae_lambda':1.0,'clip_epsilon':0.0,'entropy_coef':0.0,'value_coef':0.0,'lr':1e-4,'ppo_epochs':1,'max_grad_norm':1.0}
            )
            if state_dict is not None:
                agent.model.load_state_dict(state_dict)
            for _ in range(eval_episodes):
                state = env.reset()
                total = 0
                done = False
                while not done:
                    action, _, _ = agent.select_action(state, eval_mode=True)
                    state, reward, done, _ = env.step(action)
                    total += reward
                rewards.append(total)
        results[path] = np.mean(rewards)
    return results


def plot_comparison(local_metrics, global_metrics, algo, strategy):
    clients = list(local_metrics.keys())
    loc_vals = [local_metrics[c] for c in clients]
    glob_vals = [global_metrics[c] for c in clients]
    x = np.arange(len(clients))
    w = 0.35
    plt.figure(figsize=(10,6))
    plt.bar(x - w/2, loc_vals, w, label='Local')
    plt.bar(x + w/2, glob_vals, w, label='Global')
    plt.xticks(x, clients, rotation=45, ha='right')
    plt.xlabel('Client'); plt.ylabel('Avg Reward')
    plt.title(f'Local vs Global ({algo.upper()}, {strategy.upper()})')
    plt.legend()
    Path('results').mkdir(exist_ok=True)
    out = f'results/{algo}_{strategy}_comparison.png'
    plt.tight_layout(); plt.savefig(out, dpi=300)
    print(f"Saved comparison plot to {out}")


def main():
    parser = argparse.ArgumentParser(description="Federated PeMS traffic learning")
    parser.add_argument('--algo', choices=['dqn','ppo'], required=True)
    parser.add_argument('--strategy', choices=['fedavg','fedprox'], default='fedavg')
    parser.add_argument('--clients', nargs='+', required=True, help='Paths to per-client CSV(s) or directory')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--steps', type=int, default=288)
    parser.add_argument('--mu', type=float, default=0.1)
    parser.add_argument('--global_model', type=str, default=None,
                        help='Load pretrained global .pth and skip training')
    args = parser.parse_args()

    # expand directory of CSVs or .csv.gz
    if len(args.clients) == 1 and Path(args.clients[0]).is_dir():
        args.clients = sorted(str(p) for p in Path(args.clients[0]).glob('*.csv*'))

    if args.global_model:
        print(f"Loading pretrained global model from {args.global_model}")
        global_state = torch.load(args.global_model, map_location='cpu')
    else:
        global_state = None
        for r in range(1, args.rounds+1):
            print(f"\n=== FL Round {r}/{args.rounds} ({args.strategy.upper()}) ===")
            local_states = []
            for c in args.clients:
                print(f"Training local on {c}...")
                fn = local_train_dqn if args.algo == 'dqn' else local_train_ppo
                sd = fn(c, global_state, args.episodes, args.steps)
                local_states.append(sd)
            # first round: always use FedAvg to initialize
            if global_state is None:
                global_state = fed_avg(local_states)
            else:
                if args.strategy == 'fedavg':
                    global_state = fed_avg(local_states)
                else:
                    global_state = fed_prox(local_states, global_state, mu=args.mu)
            print(f"Completed round {r}")
        Path('results').mkdir(exist_ok=True)
        outp = f'results/global_{args.algo}_{args.strategy}.pth'
        torch.save(global_state, outp)
        print(f"Saved new global model to {outp}")

    print("Evaluating local-only models...")
    local_metrics = evaluate_model(args.algo, None, args.clients, args.eval_episodes, args.steps)
    print("Evaluating global model...")
    global_metrics = evaluate_model(args.algo, global_state, args.clients, args.eval_episodes, args.steps)
    plot_comparison(local_metrics, global_metrics, args.algo, args.strategy)

if __name__ == '__main__':
    main()
