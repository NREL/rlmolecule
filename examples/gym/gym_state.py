import base64
from copy import deepcopy
import logging
import pickle
from typing import Sequence


import numpy as np
import gym

from rlmolecule.tree_search.graph_search_state import GraphSearchState

from alphazero_gym import AlphaZeroGymEnv

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
        return hash(self.__repr__())

    def get_next_actions(self) -> Sequence[GraphSearchState]:
        next_actions = []
        if not self.done:
            for action in range(self.env.action_space.n):
                env_copy = deepcopy(self.env)
                _, step_rew, done, meta = env_copy.step(action)
                cumulative_rew = self.cumulative_reward + step_rew
                next_actions.append(
                    GymEnvState(env_copy, self.step_count+1, step_rew, cumulative_rew, done, meta))
        return next_actions


class AtariGymEnvState(GymEnvState):
    """TODO: Still needs testing."""
    def __init__(self, 
                 env: AlphaZeroGymEnv,
                 step_reward: float,
                 cumulative_reward: float,
                 done: bool) -> None:
        super().__init__(env, step_reward, cumulative_reward, done)

    def serialize(self) -> str:
        self_data = (
            self.step_reward,
            self.cumulative_reward,
            self.done,
            self.env.clone_full_state())
        return base64.b64encode(pickle.dumps(self_data)).decode('utf-8')

    def deserialize(self, data: str) -> 'AtariGymEnvState':     # how to return self class?
        data = pickle.loads(base64.b64decode(data))
        step_reward = data[0]
        cumulative_reward = data[1]
        done = data[2]
        self.env.restore_full_state(data[3])
        logger.debug("DESERIALIZING", step_reward, cumulative_reward, done)
        return AtariGymEnvState(self.env, step_reward, cumulative_reward, done)
