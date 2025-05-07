# baseline_agents.py
# Two simple baseline policies for PeMSTrafficEnv: RandomAgent and FixedTimeAgent.

import numpy as np
from typing import Tuple

class RandomAgent:
    """
    A policy that selects actions uniformly at random.
    """
    def __init__(self, action_space: int):
        self.action_space = action_space

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> Tuple[int, float, float]:
        # ignore state, just random
        action = np.random.randint(0, self.action_space)
        # return dummy logprob and value
        return action, 0.0, 0.0

class FixedTimeAgent:
    """
    A fixed-time traffic signal cycle policy.
    Cycles: NS green, NS yellow, EW green, EW yellow, repeating.
    Each phase lasts a fixed number of environment steps.
    """
    def __init__(
        self,
        action_space: int,
        ns_green_steps: int = 3,
        ns_yellow_steps: int = 1,
        ew_green_steps: int = 3,
        ew_yellow_steps: int = 1
    ):
        assert action_space == 4, "Action space must be size 4"
        # build the sequence of actions
        self.cycle = (
            [0] * ns_green_steps +  # NS green
            [1] * ns_yellow_steps + # NS yellow
            [2] * ew_green_steps +  # EW green
            [3] * ew_yellow_steps   # EW yellow
        )
        self.pointer = 0

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> Tuple[int, float, float]:
        action = self.cycle[self.pointer]
        self.pointer = (self.pointer + 1) % len(self.cycle)
        return action, 0.0, 0.0
