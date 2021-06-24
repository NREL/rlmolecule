from abc import ABC, abstractmethod

import gym

from rlmolecule.tree_search.graph_search_state import GraphSearchState


class GraphProblem(ABC):

    @property
    @abstractmethod
    def observation_space(self) -> gym.Space:
        pass

    @property
    @abstractmethod
    def null_observation(self) -> any:
        pass

    @property
    @abstractmethod
    def action_space(self) -> gym.Space:
        pass

    @property
    @abstractmethod
    def max_num_actions(self) -> int:
        pass

    @abstractmethod
    def make_observation(self, state: GraphSearchState) -> any:
        pass

    @abstractmethod
    def get_initial_state(self) -> GraphSearchState:
        pass

    @abstractmethod
    def step(self, state: GraphSearchState) -> (float, bool, dict):
        pass
