import logging

import numpy as np
import gym

logger = logging.getLogger(__name__)

ACTION_MAP = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
OBSTACLE_CHANNEL = 0
GOAL_CHANNEL = 1
PLAYER_CHANNEL = 2


class GridEnv(gym.Env):
    
    def __init__(self,
                 grid: np.ndarray,
                 max_episode_steps: int = 16,
                 goal_reward: float = 0.):
        
        self.size = grid.shape[1]
        self.start = tuple([x[0] for x in np.where(grid[PLAYER_CHANNEL, :, :])])
        self.goal = tuple([x[0] for x in np.where(grid[GOAL_CHANNEL, :, :])])
        self.initial_grid = grid.copy()

        self.goal_reward = goal_reward
        self.max_episode_steps = max_episode_steps
        self.episode_steps = None

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.initial_grid.shape, dtype=int)
        self.action_space = gym.spaces.Discrete(4)

    def reset(self):
        self.episode_steps = 0
        self.player = list(self.start)
        self.grid = self.initial_grid.copy()
        return self.get_obs()

    def get_obs(self):
        return (255 * self.grid.copy()).astype(np.int64)

    def step(self, action: int) -> None:

        # Zero out the previous player cell
        self.grid[PLAYER_CHANNEL, self.player[0], self.player[1]] = 0
        
        # Update the new player cell, keeping it within the grid boundaries
        action = ACTION_MAP[action]
        self.player[0] = min(self.size-1, max(0, self.player[0] + action[0]))
        self.player[1] = min(self.size-1, max(0, self.player[1] + action[1]))
        self.grid[PLAYER_CHANNEL, self.player[0], self.player[1]] = 1

        goal_reached = tuple(self.player) == self.goal        
        self.episode_steps += 1
        max_steps_reached = self.episode_steps == self.max_episode_steps
        logger.debug("goal reached: {}, max steps reached: {}".format(goal_reached, max_steps_reached))
        done = goal_reached or max_steps_reached

        return self.get_obs(), self.get_reward(), done, {}

    def get_reward(self):
        return -1 if np.all(self.player == self.goal) else self.goal_reward


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    grid = np.zeros((3, 3, 3), dtype=int)
    grid[PLAYER_CHANNEL, 0, 0] = 1
    grid[GOAL_CHANNEL, -1, -1] = 1

    print(grid)

    env = GridEnv(grid, max_episode_steps=100)
    env.reset()
    done, rew, step = False, 0., 0
    while not done:
        action = env.action_space.sample()
        obs, r, done, _ = env.step(action)
        rew += r
        step += 1
        print("\nstep, reward, done", step, r, done)
        print("action", action)
        print("obs", obs)
    print("final reward", rew)
