from copy import deepcopy
from typing import Sequence

import numpy as np
import gym

from rlmolecule.tree_search.graph_search_state import GraphSearchState

from alphazero_gym import AlphaZeroGymEnv


class GymEnvState(GraphSearchState):
    """Gyn env state implementation that maps the gym API to the GraphSearchState
    interface.  Should work generically for any gym env."""

    def __init__(self, 
                 env: AlphaZeroGymEnv,
                 step_reward: float,
                 cumulative_reward: float,
                 done: bool) -> None:
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.env = deepcopy(env)
        self.step_reward = step_reward
        self.cumulative_reward = cumulative_reward
        self.done = done

    def __repr__(self) -> str:
        return self.env.get_obs().__repr__()

    def equals(self, other: any) -> bool:
        return type(self) == type(other) and \
               np.isclose(self._env.get_obs(), other._env.get_obs())   # legit?

    def hash(self) -> int:
        return hash(self.__repr__())

    def get_next_actions(self) -> Sequence[GraphSearchState]:
        next_actions = []
        if not self.done:
            for action in range(self.env.action_space.n):
                env_copy = deepcopy(self.env)
                _, step_rew, done, _ = env_copy.step(action)
                cumulative_rew = self.cumulative_reward + step_rew
                next_actions.append(
                    GymEnvState(env_copy, step_rew, cumulative_rew, done))
        return next_actions

