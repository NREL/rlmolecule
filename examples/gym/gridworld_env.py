import logging
from typing import Tuple

import numpy as np
import gym


ACTION_MAP = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
OBSTACLE_CHANNEL = 0
GOAL_CHANNEL = 1
PLAYER_CHANNEL = 2


class GridEnv(gym.Env):
    
    def __init__(self,
                 grid: np.ndarray,
                 max_episode_steps: int = 16):
        
        self.size = grid.shape[1]
        self.start = tuple([x[0] for x in np.where(grid[PLAYER_CHANNEL, :, :])])
        self.goal = tuple([x[0] for x in np.where(grid[GOAL_CHANNEL, :, :])])
        self.initial_grid = grid.copy()

        self.max_episode_steps = max_episode_steps
        self.episode_steps = None

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
        self.grid[PLAYER_CHANNEL, self.player[0], self.player[1]] = 0.
        
        # Update the new player cell, keeping it within the grid boundaries
        action = ACTION_MAP[action]
        self.player[0] = min(self.size-1, max(0, self.player[0] + action[0]))
        self.player[1] = min(self.size-1, max(0, self.player[1] + action[1]))
        self.grid[PLAYER_CHANNEL, self.player[0], self.player[1]] = 1.

        goal_reached = tuple(self.player) == self.goal        
        self.episode_steps += 1
        max_steps_reached = self.episode_steps == self.max_episode_steps
        done = goal_reached or max_steps_reached

        reward = -1 if not done else self.get_terminal_reward()

        return self.get_obs(), reward, done, {}

    def get_terminal_reward(self) -> float:
        if np.all(np.array(self.player) == np.array(self.goal)):
            return self.size
        else:
            return -np.sum(np.abs(np.array(self.player) - np.array(self.goal)))


if __name__ == "__main__":
    
    grid = np.zeros((3, 3, 3), dtype=int)
    grid[PLAYER_CHANNEL, 0, 0] = 1
    grid[GOAL_CHANNEL, -1, -1] = 1

    env = GridEnv(grid, max_episode_steps=6)

    env.reset()
    done, rew, step = False, 0., 0
    while not done:
        action = env.action_space.sample()
        obs, r, done, _ = env.step(action)
        rew += r
        step += 1
        print("\nstep {}, reward {}, done {}".format(step, r, done))
        print("action", action)
        print("obs\n", obs[PLAYER_CHANNEL, :, :].squeeze())
        
    print("final reward", rew)
