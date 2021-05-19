import logging
from copy import deepcopy
from typing import Sequence

import gym
import numpy as np

from rlmolecule.gym.alphazero_gym import AlphaZeroGymEnv
from rlmolecule.sql import hash_to_integer
from rlmolecule.tree_search.graph_search_state import GraphSearchState

logger = logging.getLogger(__name__)


class GymEnvState(GraphSearchState):
    """Gyn env state implementation that maps the gym API to the GraphSearchState
    interface.  Should work generically for any gym env."""

    def __init__(self,
                 env: AlphaZeroGymEnv,
                 step_count: int,
                 step_reward: float,
                 cumulative_reward: float,
                 done: bool,
                 meta: dict = {}) -> None:
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.env = deepcopy(env)
        self.step_count = step_count
        self.step_reward = step_reward
        self.cumulative_reward = cumulative_reward
        self.done = done
        self.meta = meta

    def __repr__(self) -> str:
        return (self.env.get_obs(), self.step_count).__repr__()

    def equals(self, other: any) -> bool:
        are_close = np.all(np.isclose(self.env.get_obs(), other.env.get_obs()))
        same_time = (self.step_count == other.step_count)
        return are_close and same_time

    def hash(self) -> int:
        return hash_to_integer(self.env.get_obs().tobytes()) ^ hash_to_integer(bytes(13 * self.step_count))

    def get_next_actions(self) -> Sequence[GraphSearchState]:
        next_actions = []
        if not self.done:
            for action in range(self.env.action_space.n):
                env_copy = deepcopy(self.env)
                _, step_rew, done, meta = env_copy.step(action)
                cumulative_rew = self.cumulative_reward + step_rew
                next_actions.append(GymEnvState(env_copy, self.step_count + 1, step_rew, cumulative_rew, done, meta))
        return next_actions
