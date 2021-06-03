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

        num_actions = len(self.delegate.action_map)
        self.observation_space = gym.spaces.Dict({
            'action_mask': Box(0, 1, shape=(num_actions,)),
            'available_actions': Box(-10, 10, shape=(num_actions, 2)),
            'internal_observation': self.delegate.observation_space
        })
        self.action_space = delegate.action_space

    def reset(self) -> {str: np.ndarray}:
        self.delegate.reset()
        return self.make_observation()

    def step(self, action: int) -> Tuple[Union[np.ndarray, {str: np.ndarray}], float, bool, dict]:
        # this could be more efficient if we cached or elided delegate observation generation
        delegate_observation, reward, is_terminal, info = self.delegate.step(action)
        return self.make_observation(), reward, is_terminal, info

    def make_observation(self) -> {str: np.ndarray}:
        delegate = self.delegate
        successors = []
        for action in range(delegate.action_map):
            next, valid_action = delegate.make_next(action)
            successors.append(next if valid_action else None)

        # I split this out for clarity and modularity...
        delegate_observation_space = delegate.observation_space
        action_mask = []
        action_observations = []
        for next in successors:
            mask = 0
            action_observation = None
            if next is None:
                action_observation = np.zeros(delegate_observation_space.shape, dtype=delegate_observation_space.dtype)
            else:
                mask = 1
                action_observation = next.make_observation()
            action_mask.append(mask)
            action_observations.append(action_observation)

        return {
            'action_mask': action_mask,
            'action_observations': action_observations,
        }

