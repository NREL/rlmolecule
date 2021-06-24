import logging
from typing import Tuple, Dict

import gym
import numpy as np
from gym.spaces import Box

from rlmolecule.graph_gym.graph_problem import GraphProblem
from rlmolecule.tree_search.graph_search_state import GraphSearchState


class GraphGymEnv(gym.Env):
    """

    """

    def __init__(self,
                 problem: GraphProblem,
                 ) -> None:
        super().__init__()
        self.problem: GraphProblem = problem
        self.state: GraphSearchState = self.problem.get_initial_state()
        self.action_space = problem.action_space
        self.observation_space: gym.Space = gym.spaces.Dict({
            'action_mask': Box(False, True, shape=(problem.max_num_actions,), dtype=np.bool),
            'action_observations': gym.spaces.Tuple((problem.observation_space,) * problem.max_num_actions),
        })

    def reset(self) -> {str: np.ndarray}:
        self.state = self.problem.get_initial_state()
        return self.make_observation()

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        self.state = self.state.get_next_actions()[action]  # assumes get_next_actions is indexable
        reward, is_terminal, info = self.problem.step(self.state)
        return self.make_observation(), reward, is_terminal, info

    def make_observation(self) -> {str: np.ndarray}:
        action_mask = []
        action_observations = []

        for successor in self.state.get_next_actions():
            action_mask.append(True)
            action_observations.append(self.problem.make_observation(successor))

        null_observation = self.problem.null_observation
        for _ in range(len(action_mask), self.problem.max_num_actions):
            action_mask.append(False)
            action_observations.append(null_observation)

        return {
            'action_mask': np.array(action_mask, dtype=np.bool),
            'action_observations': tuple(action_observations),
        }
