import logging
from typing import Tuple, Dict

import gym
import numpy as np

from examples.gym.graph_gridworld.gridworld_graph_state import GridWorldGraphState
from examples.gym.gridworld_env import GridWorldEnv
from rlmolecule.graph_gym.graph_problem import GraphProblem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GridWorldGraphProblem(GraphProblem):
    """

    """

    def __init__(self,
                 delegate: GridWorldEnv,
                 ) -> None:
        self.delegate: GridWorldEnv = delegate
        self._null_observation = self.observation_space.sample() * 0

    @property
    def observation_space(self) -> gym.Space:
        return self.delegate.observation_space

    @property
    def null_observation(self) -> any:
        return self._null_observation

    @property
    def invalid_action_result(self) -> (float, bool, {}):
        return -1.0, True, {}

    @property
    def action_space(self) -> gym.Space:
        return self.delegate.action_space

    @property
    def max_num_actions(self) -> int:
        return len(self.delegate.action_map)

    def make_observation(self, state: GridWorldGraphState) -> np.ndarray:
        return state.delegate.make_observation()

    def get_initial_state(self) -> GridWorldGraphState:
        return GridWorldGraphState(self.delegate)

    def step(self, state: GridWorldGraphState) -> (float, bool, dict):
        is_terminal = len(state.get_next_actions()) == 0
        reward = self.delegate.get_terminal_reward() if is_terminal else 0.0
        return reward, is_terminal, {}
