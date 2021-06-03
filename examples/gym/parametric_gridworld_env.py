import logging
from typing import Tuple, Union

import numpy as np

import gym
from gym.spaces import Box

from examples.gym import gridworld_env
from examples.gym.gridworld_env import GridWorldEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class ParametricGridWorldEnv(gym.Env):
    """

    """

    def __init__(self,
                 delegate: GridWorldEnv,
                 ) -> None:

        self.delegate: GridWorldEnv = delegate

        num_actions = len(ACTION_MAP)
        self.observation_space = ray.utils.Dict({
            'action_mask': Box(0, 1, shape=(num_actions,)),
            'available_actions': Box(-10, 10, shape=(num_actions, 2)),
            'internal_observation': self.delegate.observation_space
        })
        self.action_space = delegate.action_space

    def reset(self) -> {str: np.ndarray}:
        return self._make_observation(self.delegate.reset())


    def step(self, action: int) -> Tuple[Union[np.ndarray, {str: np.ndarray}], float, bool, dict]:

        return self.get_obs(), reward, done, {}

    def get_terminal_reward(self) -> float:
        # Terminal reward results in max cumulative reward being 0.
        return -self._tuple_distance(self.player, self.goal) + 2 * (self.size - 1)

    def _tuple_distance(self, t1, t2) -> float:
        """Compute the manhattan distance between two tuples."""
        return np.sum(np.abs((np.array(t1) - np.array(t2))))


    def _make_observation(self, internal_observation : np.ndarray) -> {str: np.ndarray}:
        grid = self.delegate.grid

        action_mask = []
        action_observations = []
        for action in gridworld_env.ACTION_MAP:
            new_grid = grid.copy()

        return gym.spaces.Dict({
            'action_mask': Box(0, 1, shape=(num_actions,)),
            'available_actions': Box(-10, 10, shape=(num_actions, 2)),
            'internal_observation': internal_observation
        })

