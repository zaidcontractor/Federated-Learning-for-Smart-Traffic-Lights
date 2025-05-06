#pems_traffic_env.py

import numpy as np
import pandas as pd
import torch
import random
import gzip
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

class PeMSTrafficEnv:
    """
    A traffic environment that simulates traffic control based on PeMS district data.
    Uses real traffic patterns to simulate traffic light control scenarios.
    """
    def __init__(self, 
                 data_path: str, 
                 time_window: int = 5, 
                 max_steps: int = 288):  # 288 = 24 hours of 5-minute intervals
        """
        Initialize the environment.
        
        Args:
            data_path: Path to PeMS district data file (.csv.gz) or directory
            time_window: Number of timesteps to include in state representation
            max_steps: Maximum steps per episode
        """
        self.data_path = Path(data_path)
        self.time_window = time_window
        self.max_steps = max_steps
        
        # Load all data files if directory is provided
        if self.data_path.is_dir():
            self.data_files = sorted(list(self.data_path.glob("*.csv.gz")))
            if not self.data_files:
                raise ValueError(f"No .csv.gz files found in {data_path}")
            
            # For district identification
            self.district_ids = [file.stem.split('.')[0] for file in self.data_files]
            print(f"Found {len(self.data_files)} district files: {self.district_ids}")
            
            # Start with a random district
            self.current_file_idx = random.randint(0, len(self.data_files) - 1)
            self.current_file = self.data_files[self.current_file_idx]
        else:
            self.data_files = [self.data_path]
            self.district_ids = [self.data_path.stem.split('.')[0]]
            self.current_file_idx = 0
            self.current_file = self.data_path
        
        # Load initial data
        self.data = self._load_data(self.current_file)
        
        # Define action space: 4 actions representing different traffic light phases
        # 0: North-South Green (60s), East-West Red
        # 1: North-South Yellow (5s), East-West Red
        # 2: North-South Red, East-West Green (60s)
        # 3: North-South Red, East-West Yellow (5s)
        self.action_space = 4
        
        # Define observation space 
        # Each state includes:
        # - Traffic metrics for time_window steps (flow, occupancy, speed)
        # - Current phase (one-hot encoded - 4 values)
        # - Phase duration (normalized)
        self.state_dim = (self.time_window * 3) + 4 + 1
        
        # Initialize traffic light state
        self.current_phase = 0  # Start with N-S green
        self.phase_duration = 0  # How long current phase has been active
        
        # Traffic state
        self.queue_length = 0
        self.prev_queue_length = 0
        
        # The current position in the data
        self.current_step = 0
        self.start_idx = 0
        
        # For tracking performance
        self.total_reward = 0
        self.rewards = []
        self.queue_lengths = []
        self.avg_speeds = []
        
        print(f"Initialized PeMS Traffic Environment with state dimension: {self.state_dim}")
    
    def _load_data(self, file_path: Path) -> pd.DataFrame:
        """Load and preprocess a PeMS data file"""
        print(f"Loading data from {file_path}")
        
        # Read gzipped CSV file
        try:
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Create synthetic data if file loading fails
            print("Creating synthetic data instead")
            return self._create_synthetic_data()
        
        # Basic data validation
        if len(df) == 0:
            print(f"Warning: Empty dataframe from {file_path}")
            return self._create_synthetic_data()
        
        # Check required columns
        required_cols = ['Total Flow', 'Avg Occupancy', 'Avg Speed']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns in {file_path}: {missing_cols}")
            
            # Try to find similar columns
            if 'Total Flow' in missing_cols and 'flow' in df.columns:
                df['Total Flow'] = df['flow']
            elif 'Total Flow' in missing_cols:
                # Look for lane flow columns and sum them
                lane_flow_cols = [col for col in df.columns if 'Lane' in col and 'Flow' in col]
                if lane_flow_cols:
                    df['Total Flow'] = df[lane_flow_cols].sum(axis=1)
                else:
                    df['Total Flow'] = np.random.randint(10, 200, size=len(df))
            
            if 'Avg Occupancy' in missing_cols and 'occupancy' in df.columns:
                df['Avg Occupancy'] = df['occupancy']
            elif 'Avg Occupancy' in missing_cols:
                # Look for lane occupancy columns and average them
                lane_occ_cols = [col for col in df.columns if 'Lane' in col and 'Occ' in col]
                if lane_occ_cols:
                    df['Avg Occupancy'] = df[lane_occ_cols].mean(axis=1)
                else:
                    df['Avg Occupancy'] = np.random.uniform(0.01, 0.20, size=len(df))
            
            if 'Avg Speed' in missing_cols and 'speed' in df.columns:
                df['Avg Speed'] = df['speed']
            elif 'Avg Speed' in missing_cols:
                # Look for lane speed columns and average them
                lane_speed_cols = [col for col in df.columns if 'Lane' in col and 'Speed' in col]
                if lane_speed_cols:
                    df['Avg Speed'] = df[lane_speed_cols].mean(axis=1)
                else:
                    df['Avg Speed'] = np.random.uniform(40, 70, size=len(df))
        
        # Handle NaN values - Use newer pandas methods to avoid FutureWarning
        df = df.ffill().bfill().fillna(0)
        
        # Make sure numeric columns are actually numeric
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Normalize columns for neural network
        for col in required_cols:
            max_val = df[col].max()
            if max_val > 0:  # Avoid division by zero
                df[col] = df[col] / max_val
            
        return df
    
    def _create_synthetic_data(self, rows=1000):
        """Create synthetic traffic data for testing"""
        print("Creating synthetic traffic data")
        df = pd.DataFrame()
        
        # Generate time-based flow pattern (higher in morning and evening peaks)
        time_indices = np.arange(rows)
        morning_peak = np.exp(-0.01 * (time_indices - 100)**2)  # Morning peak around index 100
        evening_peak = np.exp(-0.01 * (time_indices - 400)**2)  # Evening peak around index 400
        base_flow = 0.2 + 0.3 * np.sin(2 * np.pi * time_indices / rows)  # Base sinusoidal pattern
        
        # Combine patterns and add noise
        flow = 0.3 * base_flow + 0.4 * morning_peak + 0.4 * evening_peak
        flow = flow + 0.1 * np.random.random(rows)
        
        # Create occupancy (correlated with flow)
        occupancy = 0.6 * flow + 0.4 * np.random.random(rows)
        
        # Create speed (inversely correlated with occupancy)
        speed = 1.0 - 0.7 * occupancy + 0.2 * np.random.random(rows)
        
        # Create dataframe
        df['Total Flow'] = flow
        df['Avg Occupancy'] = occupancy
        df['Avg Speed'] = speed
        
        return df
    
    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial state"""
        # Randomly select a new data file 30% of the time
        if len(self.data_files) > 1 and random.random() < 0.3:
            prev_idx = self.current_file_idx
            while self.current_file_idx == prev_idx:  # Ensure we pick a different file
                self.current_file_idx = random.randint(0, len(self.data_files) - 1)
            
            self.current_file = self.data_files[self.current_file_idx]
            self.data = self._load_data(self.current_file)
        
        # Pick a random starting point in the data
        self.start_idx = random.randint(0, max(0, len(self.data) - self.max_steps - 1))
        self.current_step = 0
        
        # Reset traffic light state
        self.current_phase = 0  # Start with N-S green
        self.phase_duration = 0
        
        # Reset traffic metrics
        self.queue_length = self._calculate_queue()
        self.prev_queue_length = self.queue_length
        
        # Reset tracking variables
        self.total_reward = 0
        self.rewards = []
        self.queue_lengths = []
        self.avg_speeds = []
        
        # Get initial state
        state = self._get_state()
        return state
    
    def _get_data_row(self, offset=0):
        """Get data at current step + offset, with bounds checking"""
        idx = self.start_idx + self.current_step + offset
        if 0 <= idx < len(self.data):
            return self.data.iloc[idx]
        
        # If out of bounds, return zeros
        dummy = pd.Series({
            'Total Flow': 0.0,
            'Avg Occupancy': 0.0,
            'Avg Speed': 0.0
        })
        return dummy
    
    def _calculate_queue(self) -> float:
        """
        Calculate queue length based on current traffic metrics.
        This is a simplified but effective model based on the relationship
        between occupancy and speed in traffic flow theory.
        """
        try:
            # Get current traffic metrics
            metrics = self._get_data_row()
            
            flow = metrics['Total Flow']
            occupancy = metrics['Avg Occupancy']
            speed = metrics['Avg Speed']
            
            # Higher occupancy + lower speed = longer queue
            # This formula approximates queue length based on these metrics
            if speed > 0.01:  # Avoid division by near-zero
                queue = (occupancy / speed) * flow * 100
            else:
                queue = occupancy * flow * 200  # High queue when speed is near zero
                
            # Add some randomness based on current phase
            if self.current_phase in [0, 2]:  # Green phases reduce queue
                queue = max(0, queue * (0.8 + 0.2 * random.random()))
            else:  # Yellow phases increase queue slightly
                queue = queue * (1.05 + 0.1 * random.random())
                
            return float(queue)
        except Exception as e:
            print(f"Error calculating queue: {e}")
            return 0.0
    
    def _get_state(self) -> np.ndarray:
        """Construct the current state representation"""
        # Get traffic metrics for time window
        flow_window = []
        occupancy_window = []
        speed_window = []
        
        for i in range(self.time_window):
            metrics = self._get_data_row(-i)  # Look at recent history
            flow_window.append(metrics['Total Flow'])
            occupancy_window.append(metrics['Avg Occupancy'])
            speed_window.append(metrics['Avg Speed'])
        
        # Normalize traffic light phase (one-hot encoding)
        phase_onehot = np.zeros(4)
        phase_onehot[self.current_phase] = 1
        
        # Normalize phase duration
        norm_duration = min(1.0, self.phase_duration / 60.0)
        
        # Create state vector by flattening all components
        state = np.concatenate([
            flow_window,
            occupancy_window,
            speed_window,
            phase_onehot,
            [norm_duration]
        ]).astype(np.float32)
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take action in the environment and return next state, reward, etc.
        
        Args:
            action: Traffic light phase to set (0-3)
            
        Returns:
            tuple of (next_state, reward, done, info)
        """
        # Validate action
        assert 0 <= action < self.action_space, f"Invalid action: {action}"
        
        # Save previous state
        old_phase = self.current_phase
        
        # Update traffic light state
        self.current_phase = action
        
        # Update phase duration - reset if phase changed, otherwise increment
        if old_phase == action:
            self.phase_duration += 1
        else:
            self.phase_duration = 0
        
        # Store previous metrics
        self.prev_queue_length = self.queue_length
        
        # Move to next timestep
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Calculate new traffic metrics
        self.queue_length = self._calculate_queue()
        current_speed = self._get_data_row()['Avg Speed']
        
        # Track metrics
        self.queue_lengths.append(self.queue_length)
        self.avg_speeds.append(current_speed)
        
        # Calculate reward
        reward = self._calculate_reward(old_phase)
        self.total_reward += reward
        self.rewards.append(reward)
        
        # Get next state
        next_state = self._get_state()
        
        # Prepare info dict
        info = {
            'queue_length': self.queue_length,
            'prev_queue': self.prev_queue_length,
            'phase': self.current_phase,
            'phase_duration': self.phase_duration,
            'district': self.district_ids[self.current_file_idx],
            'flow': self._get_data_row()['Total Flow'],
            'speed': current_speed,
            'occupancy': self._get_data_row()['Avg Occupancy'],
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, old_phase: int) -> float:
        """Calculate reward based on changes in traffic metrics and light phases"""
        # Calculate queue difference (negative = queue decreased = good)
        queue_diff = self.queue_length - self.prev_queue_length
        
        # Calculate weighted reward components
        queue_reward = -2.0 * queue_diff  # Negative queue diff = positive reward
        
        # Penalty for unnecessary phase changes (short phases are inefficient)
        phase_change_penalty = 0.0
        if old_phase != self.current_phase:  # Phase was changed
            if old_phase % 2 == 0 and self.phase_duration < 10:  # Green phase too short
                phase_change_penalty = -3.0
            elif old_phase % 2 == 1 and self.phase_duration < 1:  # Yellow phase too short
                phase_change_penalty = -1.0
        
        # Bonuses and penalties based on state
        empty_queue_bonus = 2.0 if self.queue_length < 0.1 else 0.0
        long_queue_penalty = -0.1 * self.queue_length if self.queue_length > 0.5 else 0.0
        
        # Speed reward - higher speeds are better
        speed = self._get_data_row()['Avg Speed'] 
        speed_reward = 1.0 * speed
        
        # Invalid phase sequence penalty (should go green->yellow->red, not green->red)
        invalid_sequence_penalty = 0.0
        if (old_phase == 0 and self.current_phase == 2) or (old_phase == 2 and self.current_phase == 0):
            invalid_sequence_penalty = -2.0  # Penalty for skipping yellow
        
        # Combine reward components
        reward = (
            queue_reward + 
            phase_change_penalty + 
            empty_queue_bonus + 
            long_queue_penalty + 
            speed_reward +
            invalid_sequence_penalty
        )
        
        # Scale reward to reasonable range
        reward = max(-10.0, min(10.0, reward))
        
        return reward
    
    def render(self):
        """Display current environment state"""
        metrics = self._get_data_row()
        
        print(f"\nStep: {self.current_step}/{self.max_steps}")
        print(f"District: {self.district_ids[self.current_file_idx]}")
        print(f"Traffic Light: Phase {self.current_phase}, Duration {self.phase_duration}s")
        print(f"Traffic Metrics: Flow={metrics['Total Flow']:.2f}, " 
              f"Occupancy={metrics['Avg Occupancy']:.2f}, Speed={metrics['Avg Speed']:.2f}")
        print(f"Queue Length: {self.queue_length:.2f}")
        print(f"Reward: {self.rewards[-1] if self.rewards else 0:.2f}")
        print(f"Total Reward: {self.total_reward:.2f}")
    
    def plot_metrics(self, save_path=None):
        """Plot performance metrics"""
        if not self.rewards:
            print("No data to plot yet")
            return
            
        steps = range(len(self.rewards))
        
        plt.figure(figsize=(12, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(steps, self.rewards)
        plt.title(f'Rewards over Time - {self.district_ids[self.current_file_idx]}')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(steps, self.queue_lengths)
        plt.title('Queue Length over Time')
        plt.xlabel('Step')
        plt.ylabel('Queue Length')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(steps, self.avg_speeds)
        plt.title('Average Speed over Time')
        plt.xlabel('Step')
        plt.ylabel('Speed')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)