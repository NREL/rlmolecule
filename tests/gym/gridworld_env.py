import logging
from typing import Tuple

import numpy as np

import gym

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ACTION_MAP = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
OBSTACLE_CHANNEL = 0
GOAL_CHANNEL = 1
PLAYER_CHANNEL = 2


class GridWorldEnv(gym.Env):
    
    def __init__(self,
                 grid: np.ndarray = None,
                 max_episode_steps: int = None):
        
        self.size = grid.shape[1]
        self.max_episode_steps = max_episode_steps if max_episode_steps is not None \
            else 4 * (self.size - 1)
        self.episode_steps = None

        self.start = tuple([x[0] for x in np.where(grid[:, :, PLAYER_CHANNEL])])
        self.goal = tuple([x[0] for x in np.where(grid[:, :, GOAL_CHANNEL])])
        self.initial_grid = grid.copy()

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=self.initial_grid.shape, dtype=np.float64)
        self.action_space = gym.spaces.Discrete(4)

    def reset(self) -> np.ndarray:
        self.episode_steps = 0
        self.player = list(self.start)
        self.grid = self.initial_grid.copy()
        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        return self.grid.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:

        # Zero out the previous player cell
        self.grid[self.player[0], self.player[1], PLAYER_CHANNEL] = 0.
        
        # Update the new player cell, keeping it within the grid boundaries
        action = ACTION_MAP[action]
        _row = min(self.size-1, max(0, self.player[0] + action[0]))
        _col = min(self.size-1, max(0, self.player[1] + action[1]))

        # If the new position is open, update player position, else do nothing
        if self.grid[_row, _col, OBSTACLE_CHANNEL] == 0:
            self.player = (_row, _col)
        self.grid[self.player[0], self.player[1], PLAYER_CHANNEL] = 1.

        # Look for termination
        goal_reached = tuple(self.player) == self.goal        
        self.episode_steps += 1
        max_steps_reached = self.episode_steps == self.max_episode_steps
        done = goal_reached or max_steps_reached

        # Compute reward
        reward = -1/self.size if not done else self.get_terminal_reward()

        return self.get_obs(), reward, done, {}

    def get_terminal_reward(self) -> float:
        if np.all(np.array(self.player) == np.array(self.goal)):
            return 2*(self.size-1) / self.episode_steps  # best possible divided by actual steps
        else:
            return -np.sum(np.abs(np.array(self.player) - np.array(self.goal))) / self.size
