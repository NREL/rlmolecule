from typing import Sequence
from copy import deepcopy

import gym

from rlmolecule.tree_search.graph_search_state import GraphSearchState


class GymEnvState(GraphSearchState):

    def __init__(self, env: gym.Env, reward: float, done: bool) -> None:
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self._env = deepcopy(env)
        self._reward = reward
        self._done = done

    def __repr__(self) -> str:
        return self._env.get_obs().__repr__()

    def equals(self, other: any) -> bool:
        return type(self) == type(other) and \
               self.get_obs() == other.get_obs()   # legit?

    def hash(self) -> int:
        return hash(self.__repr__())

    def get_next_actions(self) -> Sequence[GraphSearchState]:
        next_actions = []
        if not self._done:
            for action in range(self.env.action_space.n):
                env_copy = deepcopy(self._env)
                _, rew, done, _ = env_copy.step(action)
                next_actions.append(GymEnvState(env_copy, rew, done))
        return next_actions

    def get_reward(self) -> float:
        return self._reward

    @property
    def env(self) -> gym.Env:
        return self._env

    @property
    def reward(self) -> float:
        return self._reward
