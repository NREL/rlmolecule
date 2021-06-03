import logging
import random
import sys
from copy import copy
from enum import IntEnum
from typing import Tuple

import numpy as np

import gym

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CellType(IntEnum):
    OPEN = 0
    GOAL = 1
    START = 2
    OBSTACLE = 3


class GridWorldEnv(gym.Env):
    """Class implementing an n-dimensional gridworld with 2n discrete action space.

    The observation is modeled as an 3-channel array with separate channels for 
    obstacles (0), goal (1), and player (2).  The env is instantiated by 
    passing a n-dimensional numpy array of CellType integer codes.
    """

    def __init__(self,
                 grid: np.ndarray,
                 max_episode_steps: int = None,
                 sparse_rewards: bool = False,
                 observation_type: str = 'scalar',
                 ) -> None:
        self.observation_type: str = observation_type

        self.grid: np.ndarray = grid.copy()
        self.goals: [(int, ...)] = tuple(np.argwhere(self.grid == CellType.GOAL))

        self.max_episode_steps: int = max_episode_steps if max_episode_steps is not None \
            else 4 * (np.prod(self.shape) - 1)
        self.sparse_rewards: bool = sparse_rewards

        self.episode_steps: int = 0
        self.cumulative_reward: float = 0.0

        self.player_position: np.ndarray = np.zeros(len(self.shape))

        num_dims = len(self.shape)

        self.action_map: [(int, ...)] = []
        for d in range(num_dims):
            self.action_map.append(tuple((1 * (i == d) for i in range(num_dims))))
            self.action_map.append(tuple((-1 * (i == d) for i in range(num_dims))))

        low = 0
        high = 1
        dtype = np.int64
        obs_shape: (int, ...) = (1,)
        if self.observation_type == 'index':  # a single flattened index of the position
            obs_shape = (1,)
            high = np.prod(self.shape) - 1
        elif self.observation_type == 'scalar':  # (row, col)
            obs_shape = (num_dims,)
            low = np.array([0] * num_dims)
            high = np.array([d - 1 for d in self.shape])
        elif self.observation_type == 'rgb':  # binary matrices of [obstacle, goal, start]
            obs_shape = tuple(list(self.shape) + [3])
            dtype = np.float64
        elif self.observation_type == 'grayscale':  # float matrix, 0 for open, 1/3 for obstacle, 2/3 for goal and 1 for player
            obs_shape = self.shape
            dtype = np.float64
        else:
            assert False, f'Unknown observation_type {observation_type}.'

        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=low, high=high, shape=obs_shape, dtype=dtype)

        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(4)

    def make_next(self, action: int) -> ('GridWorldEnv', bool):
        next = copy(self)
        valid_position = next.apply_action(action)
        return next, valid_position

    @property
    def shape(self) -> tuple:
        return self.grid.shape

    def reset(self) -> np.ndarray:
        self.episode_steps = 0
        self.cumulative_reward = 0.
        self.player_position = np.array(random.choice(list(np.argwhere(self.grid == CellType.START))))
        return self.make_observation()

    def step(self, action: int) -> (np.ndarray, float, bool, {}):
        self.apply_action(action)

        # Look for termination
        goal_reached = self.grid[tuple(self.player_position)] == CellType.GOAL
        self.episode_steps += 1
        max_steps_reached = self.episode_steps == self.max_episode_steps
        done = goal_reached or max_steps_reached

        # Compute reward
        reward = -1
        if done:
            reward += self.get_terminal_reward()
        self.cumulative_reward += reward

        # If using sparse rewards, only return cumulative reward if done, else 0.
        if self.sparse_rewards:
            reward = self.cumulative_reward if done else 0.

        return self.make_observation(), reward, done, {}

    def get_terminal_reward(self) -> float:
        # Terminal reward results in max cumulative reward being 0.
        distance = min((self.distance_between_cells(self.player_position, goal) for goal in self.goals))
        return -distance + 2 * (np.prod(self.shape) - 1)

    @staticmethod
    def distance_between_cells(a: np.ndarray, b: np.ndarray) -> int:
        return int(np.sum(np.abs(a - b)))

    def nearest_goal(self, position: np.ndarray) -> (int, np.ndarray):
        return max([(env.distance_between_cells(goal, position), goal) for goal in self.goals],
                   key=lambda tup: tup[0])

    def make_observation(self) -> np.ndarray:
        shape = self.shape
        observation = None
        observation_type = self.observation_type
        if observation_type == 'index':
            idx = 0
            for i, p in enumerate(self.player_position):
                idx = idx * shape[i] + p
            observation = [idx]
        elif observation_type == 'scalar':
            observation = list(self.player_position)
        elif observation_type == 'grayscale':
            observation = np.zeros(shape)
            observation += (1.0 / 3.0) * (self.grid == CellType.OBSTACLE)
            observation += (2.0 / 3.0) * (self.grid == CellType.GOAL)
            observation[tuple(self.player_position)] = 1.0
        elif observation_type == 'rgb':
            player_location_grid = np.zeros(self.shape)
            player_location_grid[tuple(self.player_position)] = 1.0
            observation = np.stack([
                self.grid == CellType.OBSTACLE,
                self.grid == CellType.GOAL,
                player_location_grid
            ])
        else:
            assert False, f'Unknown observation_type {observation_type}.'

        return observation

    def get_next_position(
            self,
            player_position: np.ndarray,
            grid: np.ndarray,
            action: int,
    ) -> (np.ndarray, bool):
        # for make_next to work without additional trickery, this needs to be a new object
        new_position: np.ndarray = player_position + self.action_map[action]

        for i, p in enumerate(new_position):
            if p < 0 or p >= grid.shape[i]:
                return player_position, False

        if grid[tuple(new_position)] == CellType.OBSTACLE:
            return player_position, False

        return new_position, True

    def apply_action(self, action: int) -> bool:
        self.player_position, valid_move = self.get_next_position(self.player_position, self.grid, action)
        return valid_move


def make_empty_grid(size=5):
    """Helper function for creating empty (no obstacles) of given size."""
    grid = np.zeros((size, size))
    grid[0, 0] = CellType.START
    grid[-1, -1] = CellType.GOAL
    return grid


def make_doorway_grid():
    """Example where wall blocks all but 2 pixels for the player to pass through."""
    size = 10
    wall_offset = 4
    grid = np.zeros((size, size))

    grid[wall_offset, :] = CellType.OBSTACLE
    grid[wall_offset, 4] = CellType.OPEN
    grid[wall_offset, 5] = CellType.OPEN

    grid[0, 0] = CellType.START
    grid[-1, -1] = CellType.GOAL

    return grid


def policy(env):
    """An optimal policy for empty gridworld: find the vector pointing towards
    the goal, and choose the first non-zero direction.  Total episode reward
    should be 0 when running this."""
    distance, nearest_goal = env.nearest_goal(env.player_position)

    goal_direction = nearest_goal - env.player_position
    print("goal direction", goal_direction)
    action = np.where(goal_direction.squeeze() != 0)[0][0]
    if action == 0:
        if goal_direction[action] > 0:
            return 1
        return 3
    if goal_direction[action] > 0:
        return 0
    return 2


if __name__ == "__main__":

    # from tf_model import gridworld_image_embed_policy
    # model = gridworld_image_embed_policy(
    #     size=32,
    #     filters=[4, 8, 16],
    #     kernel_size=[8, 2, 2],
    #     strides=[8, 2, 1]
    # )

    size = 64
    grid = make_empty_grid(size=size)
    env = GridWorldEnv(grid, observation_type="scalar", max_episode_steps=2 * size + 2)
    observation = env.reset()

    print("observation", observation)
    # print("PREDICT", model.predict(observation.reshape(1, 1)))
    done, rew, step = False, 0., 0
    while not done:
        # action = env.action_space.sample()
        # action = 2
        action = policy(env)
        observation, r, done, _ = env.step(action)
        rew += r
        step += 1
        print("\nstep {}, reward {}, done {}".format(step, r, done))
        print("action", action)
        print("observation", observation)
        # print("policy", model.predict(observation.reshape(1, 1)))

    print("final reward", rew)
