from typing import Tuple, Dict

import gym
import numpy as np
import tensorflow as tf
from gym.spaces import Box

from rlmolecule.graph_gym.graph_problem import GraphProblem
from rlmolecule.tree_search.graph_search_state import GraphSearchState


class FlatGraphGymEnv(gym.Env):
    """

    """

    def __init__(self,
                 problem: GraphProblem,
                 ) -> None:
        super().__init__()
        self.problem: GraphProblem = problem
        self.state: GraphSearchState = self.problem.get_initial_state()
        self.action_space = problem.action_space

        max_num_actions = problem.max_num_actions
        subspaces = {key: gym.spaces.Box(low=np.repeat(value.low, max_num_actions, axis=0),
                                         high=np.repeat(value.high, max_num_actions, axis=0),
                                         shape=(max_num_actions * value.shape[0], *value.shape[1:]),
                                         dtype=value.dtype)
                     for key, value in problem.observation_space.spaces.items()}
        subspaces['action_mask'] = Box(False, True, shape=(max_num_actions,), dtype=np.bool)

        self.observation_space: gym.Space = gym.spaces.Dict(subspaces)

        # self.observation_space: gym.Space = gym.spaces.Dict({
        #     'action_mask': Box(False, True, shape=(problem.max_num_actions,), dtype=np.bool),
        #     'action_observations': gym.spaces.Tuple((problem.observation_space,) * problem.max_num_actions),
        # })

    def reset(self) -> {str: np.ndarray}:
        self.state = self.problem.get_initial_state()
        return self.make_observation()

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        next_actions = self.state.get_next_actions()

        reward, is_terminal, info = self.problem.invalid_action_result
        if action < len(next_actions):
            self.state = next_actions[action]  # assumes get_next_actions is indexable
            reward, is_terminal, info = self.problem.step(self.state)

        result = (self.make_observation(), reward, is_terminal, info)
        return result

    def make_observation(self) -> {str: np.ndarray}:
        max_num_actions = self.problem.max_num_actions
        action_mask = [False] * max_num_actions
        action_observations = [self.problem.null_observation] * max_num_actions

        for i, successor in enumerate(self.state.get_next_actions()):
            if i >= max_num_actions:
                break
            action_mask[i] = True
            action_observations[i] = self.problem.make_observation(successor)

        flat_action_observations = {
            'action_mask': np.array(action_mask, dtype=np.bool),
        }
        for key in action_observations[0].keys():
            action_observations_sublist = [action_observation[key] for action_observation in action_observations]
            flat_action_observations[key] = tf.concat(action_observations_sublist, axis=0)
        return flat_action_observations
